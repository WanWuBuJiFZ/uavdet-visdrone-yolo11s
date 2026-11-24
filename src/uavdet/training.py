"""Training of one YOLO11s detector on the audited VisDrone dataset view.

Rules enforced here (see ``docs/INTERFACE_CONTRACT.md`` §3.1 and
验收清单 B01-B04, B08):

* The start weights are checked before anything else. A missing file, or a file
  that is recognisably one of *our* trained checkpoints (``best.pt`` /
  ``last.pt``, a checkpoint storing the 10 VisDrone class names, or a checkpoint
  carrying optimizer state) is a hard :class:`~uavdet.types.WeightError` unless
  ``train.allow_trained_init`` is true. Training never falls back to a random
  initialisation, and never silently continues from an old result.
* Every training parameter is passed to :meth:`ultralytics.engine.model.Model.train`
  explicitly, so an Ultralytics default can never apply by accident
  (``project``/``name`` included, which the legacy code forgot).
* The trainer's *actual* ``save_dir`` is used to copy ``args.yaml``,
  ``results.csv``, ``weights/`` and ``train.log`` into ``<run>/train``. The run
  directory is the only output location this module writes to.
* ``resume`` only accepts a checkpoint inside the same run's ``train_dir/weights``
  and the configured epoch target is never added to the checkpoint's epoch
  counter. After training the merged ``results.csv`` epoch column must be
  strictly increasing. If fewer epochs finished than configured, the artefacts
  are still copied and then :class:`~uavdet.types.RunError` is raised: an
  unfinished run must never look complete.

Heavy dependencies (torch, ultralytics) are imported inside the functions that
need them, so ``import uavdet.training`` works on a machine without them.
"""

from __future__ import annotations

import csv
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .artifacts import now_iso, record_event, sha256_file
from .devices import DeviceChoice, device_summary, resolve_device
from .types import CLASS_NAMES, NUM_CLASSES, RunContext, RunError, UAVDetError, WeightError

__all__ = [
    "check_init_weights",
    "train_model",
    "training_arguments",
    "default_yolo_factory",
    "resolve_data_yaml",
    "read_results_epochs",
    "TRAINED_WEIGHT_NAMES",
    "WEIGHT_STEMS",
]

#: File names Ultralytics uses for its own training checkpoints.
TRAINED_WEIGHT_NAMES = frozenset({"best.pt", "last.pt"})

#: Warmup values fixed by the frozen protocol (not part of the config schema).
FIXED_WARMUP_MOMENTUM = 0.8
FIXED_WARMUP_BIAS_LR = 0.1

#: Files copied from the trainer's save_dir into ``<run>/train``.
COPIED_ARTIFACTS: tuple[str, ...] = ("args.yaml", "results.csv", "train.log")
#: Checkpoint stems (``<stem>.pt``) that must exist after training.
WEIGHT_STEMS: tuple[str, ...] = ("best", "last")
COPIED_WEIGHTS: tuple[str, ...] = tuple(f"{stem}.pt" for stem in WEIGHT_STEMS)


# ---------------------------------------------------------------------------
# factories / helpers
# ---------------------------------------------------------------------------
def default_yolo_factory(weights: Path) -> Any:
    """Build the real Ultralytics model (imported lazily, on purpose)."""
    from ultralytics import YOLO  # noqa: PLC0415 - import-time cost + optional dependency

    return YOLO(str(weights))


def resolve_data_yaml(
    config: Mapping[str, Any],
    run: RunContext,
    data_yaml: str | Path | None = None,
) -> Path:
    """Locate the data.yaml of this run's dataset view.

    Order: explicit argument, the run manifest's ``dataset_view`` record, then a
    fresh ``prepare_dataset_view`` from the stored audit (the CLI normally does
    that before calling a stage function, so the last step is a fallback that
    still refuses to invent a dataset).
    """
    if data_yaml is not None:
        path = Path(data_yaml)
        if not path.is_file():
            raise UAVDetError(f"data.yaml not found: {path}")
        return path

    recorded = (run.manifest.get("dataset_view") or {}).get("data_yaml")
    if recorded:
        path = Path(str(recorded))
        if path.is_file():
            return path

    from .data import load_audit, prepare_dataset_view  # noqa: PLC0415 - avoid a cycle

    try:
        loaded = load_audit(run)
    except Exception as exc:  # noqa: BLE001 - re-raised with a clear message
        raise RunError(
            f"run '{run.run_id}' has no dataset view and no readable audit "
            f"({type(exc).__name__}: {exc}); run the 'check' command first"
        ) from exc
    audit = dict(loaded.get("manifest") or {})
    # The persisted audit manifest keeps the per-image list; the split counters
    # live in dataset_stats.json. prepare_dataset_view wants both.
    audit.setdefault("splits", (loaded.get("stats") or {}).get("splits") or {})
    return Path(prepare_dataset_view(audit, run, config))


def _read_checkpoint_metadata(path: Path) -> dict[str, Any]:
    """Best-effort metadata of an Ultralytics ``.pt`` file (never raises).

    The caller decides what to do with the facts; an unreadable file is not
    "assume COCO" but "unknown", which is recorded as a warning.
    """
    info: dict[str, Any] = {
        "names": None,
        "num_classes": None,
        "epoch": None,
        "has_optimizer": False,
        "error": None,
    }
    try:
        import torch  # noqa: PLC0415 - optional dependency
    except Exception as exc:  # noqa: BLE001 - torch missing is a warning, not a crash
        info["error"] = f"torch is not importable ({exc}); the checkpoint metadata is unverified"
        return info

    try:
        checkpoint = torch.load(str(path), map_location="cpu")
    except Exception as exc:  # noqa: BLE001 - corrupt/foreign file
        info["error"] = (
            f"cannot read {path.name} as a torch checkpoint with torch "
            f"{getattr(torch, '__version__', '?')}: {type(exc).__name__}: {exc}"
        )
        return info

    if not isinstance(checkpoint, Mapping):
        info["error"] = f"{path.name} does not contain a checkpoint dictionary"
        return info

    info["epoch"] = checkpoint.get("epoch")
    info["has_optimizer"] = checkpoint.get("optimizer") is not None

    model = checkpoint.get("model")
    raw_names = getattr(model, "names", None)
    if raw_names is None:
        yaml_block = getattr(model, "yaml", None)
        if isinstance(yaml_block, Mapping):
            raw_names = yaml_block.get("names")
    names = _normalise_names(raw_names)
    info["names"] = names
    if names is not None:
        info["num_classes"] = len(names)
    else:
        model_yaml = getattr(model, "yaml", None)
        num_classes = model_yaml.get("nc") if isinstance(model_yaml, Mapping) else None
        if isinstance(num_classes, int):
            info["num_classes"] = num_classes
    return info


def _normalise_names(raw_names: Any) -> tuple[str, ...] | None:
    """Turn Ultralytics' ``names`` (dict or list) into a tuple in class-id order."""
    if raw_names is None:
        return None
    if isinstance(raw_names, Mapping):
        try:
            ordered = sorted(((int(key), str(value)) for key, value in raw_names.items()))
        except (TypeError, ValueError):
            return None
        if [index for index, _ in ordered] != list(range(len(ordered))):
            return None
        return tuple(name for _, name in ordered)
    if isinstance(raw_names, (list, tuple)):
        return tuple(str(name) for name in raw_names)
    return None


# ---------------------------------------------------------------------------
# start weights
# ---------------------------------------------------------------------------
def check_init_weights(
    config: Mapping[str, Any],
    *,
    allow_trained: bool | None = None,
) -> dict[str, Any]:
    """Verify ``train.init_weights`` before a training run starts.

    Returns ``{"path", "sha256", "kind", "num_classes", "warnings"}``.

    Raises :class:`~uavdet.types.WeightError` when the file is missing/empty, or
    when it is an already-trained checkpoint and ``train.allow_trained_init`` is
    false (an explicit ``allow_trained`` argument wins, for a re-validation CLI
    path that must not depend on the config flag).
    """
    train_cfg = dict(config.get("train") or {})
    raw_path = train_cfg.get("init_weights")
    if not raw_path:
        raise WeightError(
            "train.init_weights is empty; a new formal run must start from the official "
            "YOLO11s COCO pretrained file (e.g. weights/yolo11s.pt)"
        )
    path = Path(str(raw_path))
    if not path.exists():
        raise WeightError(
            f"training start weights not found: {path}. Refusing to silently start from a "
            f"random initialisation - download the official YOLO11s pretrained weights and "
            f"point train.init_weights at them."
        )
    if not path.is_file():
        raise WeightError(f"train.init_weights is not a file: {path}")
    if path.stat().st_size <= 0:
        raise WeightError(f"training start weights are empty: {path}")

    allow = bool(train_cfg.get("allow_trained_init", False)) if allow_trained is None else bool(allow_trained)
    metadata = _read_checkpoint_metadata(path)
    warnings: list[str] = []
    if metadata.get("error"):
        warnings.append(str(metadata["error"]))

    names = metadata.get("names")
    num_classes = metadata.get("num_classes")
    name_flag = path.name.lower() in TRAINED_WEIGHT_NAMES
    class_flag = names is not None and tuple(names) == CLASS_NAMES
    looks_trained = bool(name_flag or class_flag or metadata.get("has_optimizer"))

    if looks_trained and not allow:
        reasons: list[str] = []
        if name_flag:
            reasons.append(f"the file name '{path.name}' is an Ultralytics training checkpoint name")
        if class_flag:
            reasons.append(f"it stores the {NUM_CLASSES} VisDrone class names {list(CLASS_NAMES)}")
        if metadata.get("has_optimizer"):
            reasons.append("it stores an optimizer state, so it is a resumable training checkpoint")
        raise WeightError(
            "refusing to start a new training run from an already trained checkpoint: "
            + "; ".join(reasons)
            + f". Point train.init_weights at the official COCO YOLO11s file, or set "
            f"train.allow_trained_init=true if continuing from '{path.name}' is really intended."
        )

    if looks_trained:
        warnings.append(
            f"starting from the trained checkpoint {path.name} because "
            f"allow_trained_init/allow_trained is true"
        )

    kind = "trained" if looks_trained else "pretrained-coco"
    if num_classes is not None and not looks_trained and num_classes != NUM_CLASSES:
        warnings.append(
            f"the checkpoint declares {num_classes} classes; the detection head is "
            f"re-initialised for the {NUM_CLASSES} VisDrone classes"
        )
    if num_classes is None:
        warnings.append(
            f"the class metadata of {path.name} could not be verified; the file name and the "
            f"trainer will decide (no metadata is fabricated here)"
        )

    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "kind": kind,
        "num_classes": num_classes,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# results.csv
# ---------------------------------------------------------------------------
def read_results_epochs(path: str | Path) -> list[int]:
    """Read the ``epoch`` column of an Ultralytics ``results.csv``."""
    target = Path(path)
    if not target.is_file():
        raise RunError(f"results.csv not found: {target}")
    with open(target, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise RunError(f"{target} is empty; cannot verify the completed epochs") from exc
        labels = [str(cell).strip().strip('"').lower() for cell in header]
        if "epoch" in labels:
            index = labels.index("epoch")
        elif labels and labels[0] in {"", "index", "unnamed: 0"}:
            # Renderers strip a leading DataFrame index column; mirror that here.
            index = 0
        else:
            raise RunError(
                f"{target} has no 'epoch' column (header: {header}); cannot verify the "
                f"completed epochs"
            )
        epochs: list[int] = []
        for line_number, row in enumerate(reader, start=2):
            if not row or all(not str(cell).strip() for cell in row):
                continue
            if index >= len(row):
                raise RunError(f"{target}:{line_number} has no epoch value")
            raw_value = str(row[index]).strip()
            try:
                epochs.append(int(float(raw_value)))
            except ValueError as exc:
                raise RunError(
                    f"{target}:{line_number} has a non-numeric epoch value {raw_value!r}"
                ) from exc
    return epochs


def _assert_strictly_increasing(epochs: Sequence[int], path: Path) -> None:
    """Reject a merged CSV that repeats or moves backwards."""
    if not epochs:
        raise RunError(f"{path} has no epoch rows; the trainer reported no finished epoch")
    duplicates = sorted({value for value in epochs if list(epochs).count(value) > 1})
    if duplicates:
        raise RunError(
            f"{path} repeats the epoch(s) {duplicates} after training/resume; the same epoch "
            f"was trained twice, so the run is not a clean continuation"
        )
    for previous, current in zip(epochs, epochs[1:]):
        if current <= previous:
            raise RunError(
                f"{path} epoch column is not strictly increasing ({previous} -> {current}); "
                f"resume re-trained an epoch that was already recorded"
            )


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def training_arguments(
    config: Mapping[str, Any],
    run: RunContext,
    data_yaml: Path,
    device: DeviceChoice,
) -> dict[str, Any]:
    """Every configured training parameter, spelled out (B01)."""
    train = dict(config["train"])
    evaluation = dict(config["eval"])
    arguments: dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": int(train["epochs"]),
        "imgsz": int(train["imgsz"]),
        "batch": int(train["batch"]),
        "workers": int(train["workers"]),
        "device": str(device.effective),
        "optimizer": str(train["optimizer"]),
        "lr0": float(train["lr0"]),
        "lrf": float(train["lrf"]),
        "momentum": float(train["momentum"]),
        "weight_decay": float(train["weight_decay"]),
        "warmup_epochs": float(train["warmup_epochs"]),
        "warmup_momentum": FIXED_WARMUP_MOMENTUM,
        "warmup_bias_lr": FIXED_WARMUP_BIAS_LR,
        "cos_lr": bool(train["cos_lr"]),
        "seed": int(train["seed"]),
        "deterministic": bool(train["deterministic"]),
        "amp": bool(train["amp"]),
        "mosaic": float(train["mosaic"]),
        "close_mosaic": int(train["close_mosaic"]),
        "hsv_h": float(train["hsv_h"]),
        "hsv_s": float(train["hsv_s"]),
        "hsv_v": float(train["hsv_v"]),
        "translate": float(train["translate"]),
        "scale": float(train["scale"]),
        "fliplr": float(train["fliplr"]),
        "flipud": float(train["flipud"]),
        "mixup": float(train["mixup"]),
        "copy_paste": float(train["copy_paste"]),
        # Ultralytics expects a fraction here; the config keeps the on/off flag.
        "multi_scale": float(bool(train["multi_scale"])),
        "patience": int(train["patience"]),
        "save_period": int(train["save_period"]),
        "plots": bool(train["plots"]),
        # Where the trainer writes. The run directory is the single source.
        "project": str(run.run_dir),
        "name": "train",
        "exist_ok": True,
        # Validation during training uses the frozen eval protocol thresholds.
        "val": True,
        "iou": float(evaluation["iou"]),
        "max_det": int(evaluation["max_det"]),
        "verbose": True,
    }
    return arguments


def _trainer_save_dir(model: Any, results: Any, arguments: Mapping[str, Any]) -> Path:
    """The directory the trainer really wrote to (never a guessed nested path)."""
    trainer = getattr(model, "trainer", None)
    for candidate in (getattr(trainer, "save_dir", None), getattr(results, "save_dir", None)):
        if candidate:
            return Path(str(candidate))
    return Path(str(arguments["project"])) / str(arguments["name"])


def _copy_artifacts(save_dir: Path, train_dir: Path, warnings: list[str]) -> dict[str, bool]:
    """Copy trainer products into ``<run>/train`` and report what exists."""
    train_dir.mkdir(parents=True, exist_ok=True)
    present: dict[str, bool] = {}
    for name in COPIED_ARTIFACTS:
        source = save_dir / name
        target = train_dir / name
        if not source.is_file():
            warnings.append(f"the trainer did not produce {name} in {save_dir}")
            present[name] = target.is_file()
            continue
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        present[name] = target.is_file()
    weights_target = train_dir / "weights"
    weights_target.mkdir(parents=True, exist_ok=True)
    for name in COPIED_WEIGHTS:
        source = save_dir / "weights" / name
        target = weights_target / name
        if not source.is_file():
            warnings.append(f"the trainer did not produce weights/{name} in {save_dir}")
            present[f"weights/{name}"] = target.is_file()
            continue
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        present[f"weights/{name}"] = target.is_file()
    return present


def _append_train_log(train_dir: Path, lines: Sequence[str]) -> Path:
    """Append our own bookkeeping lines to ``train/train.log`` (never truncates)."""
    target = train_dir / "train.log"
    payload = "".join(f"{line}\n" for line in lines)
    with open(target, "a", encoding="utf-8") as handle:
        handle.write(payload)
    return target


def _check_resume_checkpoint(run: RunContext, resume: Path) -> Path:
    """Resume must be a checkpoint of *this* run (same run layout)."""
    weights_dir = (run.train_dir / "weights").resolve()
    candidate = Path(resume)
    if not candidate.exists():
        raise RunError(
            f"resume checkpoint not found: {candidate}; pass the train/weights/last.pt of "
            f"run '{run.run_id}'"
        )
    resolved = candidate.resolve()
    if weights_dir != resolved.parent:
        raise RunError(
            f"resume checkpoint {resolved} is not inside this run's weights directory "
            f"({weights_dir}); resuming across runs would mix training protocols - start a "
            f"new run instead"
        )
    return resolved


def train_model(
    config: Mapping[str, Any],
    run: RunContext,
    resume: Path | None = None,
    *,
    yolo_factory: Callable[[Path], Any] | None = None,
) -> dict[str, Any]:
    """Train (or resume) YOLO11s and copy the real trainer products into the run."""
    started = time.time()
    started_at = now_iso()
    factory = yolo_factory or default_yolo_factory
    warnings: list[str] = []

    init = check_init_weights(config)
    warnings.extend(init["warnings"])

    choice = resolve_device(config["train"]["device"])
    warnings.extend(choice.warnings)
    if choice.warnings:
        record_event(
            run,
            "device_fallback",
            f"training device '{choice.requested}' is not available; using '{choice.effective}'",
            warnings=list(choice.warnings),
        )

    data_yaml = resolve_data_yaml(config, run, None)
    arguments = training_arguments(config, run, data_yaml, choice)
    epochs_target = int(arguments["epochs"])

    resumed_from: Path | None = None
    if resume is not None:
        resumed_from = _check_resume_checkpoint(run, Path(resume))
        model_weights = resumed_from
        # Native Ultralytics semantics: ``resume=True`` continues *this* checkpoint
        # to the epoch target stored in it. The target is never extended by the
        # number of finished epochs (B04).
        arguments["resume"] = True
        warnings.append(
            "resume: Ultralytics restores the checkpoint's own training arguments; the "
            "configured values are still recorded in config_resolved.yaml"
        )
    else:
        model_weights = Path(init["path"])

    train_dir = run.train_dir
    train_dir.mkdir(parents=True, exist_ok=True)
    record_event(
        run,
        "train_start",
        f"training started from {model_weights.name} for {epochs_target} epoch(s)",
        init_weights=init["path"],
        init_sha256=init["sha256"],
        device=device_summary(choice),
        resume=bool(resumed_from),
    )

    try:
        model = factory(model_weights)
    except Exception as exc:  # noqa: BLE001 - surfaced as a WeightError
        raise WeightError(
            f"cannot load the training start weights {model_weights}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    try:
        results = model.train(**arguments)
    except BaseException as exc:
        # The native trainer already writes into save_dir; keep our own line too so
        # a failed run never loses the reason.
        _append_train_log(
            train_dir,
            [
                f"[uavdet] FAILED at {now_iso()} after {round(time.time() - started, 3)}s: "
                f"{type(exc).__name__}: {exc}",
            ],
        )
        record_event(run, "train_failed", f"training failed: {type(exc).__name__}: {exc}")
        raise
    save_dir = _trainer_save_dir(model, results, arguments)

    present = _copy_artifacts(save_dir, train_dir, warnings)
    results_csv = train_dir / "results.csv"
    epochs: list[int] = []
    if present.get("results.csv") and results_csv.is_file():
        epochs = read_results_epochs(results_csv)
        _assert_strictly_increasing(epochs, results_csv)
    epochs_done = len(epochs)

    duration = round(time.time() - started, 3)
    weights_sha = {
        stem: (sha256_file(train_dir / "weights" / f"{stem}.pt")
               if (train_dir / "weights" / f"{stem}.pt").is_file() else None)
        for stem in WEIGHT_STEMS
    }
    log_lines = [
        f"[uavdet] started_at={started_at} finished_at={now_iso()} duration_s={duration}",
        f"[uavdet] init_weights={init['path']} sha256={init['sha256']} kind={init['kind']}",
        f"[uavdet] data={data_yaml}",
        f"[uavdet] device requested={choice.requested} effective={choice.effective}",
        f"[uavdet] trainer_save_dir={save_dir}",
        f"[uavdet] epochs_done={epochs_done} epochs_target={epochs_target}",
        f"[uavdet] resume={bool(resumed_from)} resumed_from={resumed_from or '-'}",
    ]
    log_lines.extend(f"[uavdet] warning: {text}" for text in warnings)
    _append_train_log(train_dir, log_lines)

    outcome: dict[str, Any] = {
        "status": "ok",
        "save_dir": run.rel(save_dir),
        "best_weights": (run.rel(train_dir / "weights" / "best.pt")
                         if present.get("weights/best.pt") else None),
        "last_weights": (run.rel(train_dir / "weights" / "last.pt")
                         if present.get("weights/last.pt") else None),
        "results_csv": (run.rel(results_csv) if present.get("results.csv") else None),
        "args_yaml": (run.rel(train_dir / "args.yaml") if present.get("args.yaml") else None),
        "train_log": run.rel(train_dir / "train.log"),
        "epochs_done": epochs_done,
        "epochs_target": epochs_target,
        "weights_sha256": weights_sha,
        "duration_s": duration,
        "resume": bool(resumed_from),
        "resumed_from": run.rel(resumed_from) if resumed_from is not None else None,
        "device": device_summary(choice),
        "warnings": warnings,
        "init_weights": {
            "path": init["path"],
            "sha256": init["sha256"],
            "kind": init["kind"],
        },
        "data_yaml": str(data_yaml),
    }

    if epochs_done < epochs_target:
        record_event(
            run,
            "train_incomplete",
            f"training stopped after {epochs_done} of {epochs_target} epoch(s)",
            epochs_done=epochs_done,
            epochs_target=epochs_target,
        )
        raise RunError(
            f"training finished only {epochs_done} of the configured {epochs_target} epoch(s); "
            f"the run is NOT complete. Artefacts were kept in {train_dir} - resume with "
            f"--resume {train_dir / 'weights' / 'last.pt'} once the cause is fixed."
        )

    record_event(
        run,
        "train_finished",
        f"training finished {epochs_done} epoch(s)",
        epochs_done=epochs_done,
        duration_s=duration,
        best_weights=outcome["best_weights"],
    )
    return outcome
