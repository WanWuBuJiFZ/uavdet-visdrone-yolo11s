"""Command line entry point: ``python -m uavdet --config <yaml> <command>``.

Only configuration/data bookkeeping is imported eagerly; torch, Ultralytics,
OpenCV and Qt are imported inside the command that needs them so that
``--help`` and ``check``/``plot`` work on a CPU-only machine without a GUI.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import __version__
from .artifacts import (
    begin_stage,
    complete_stage,
    fail_stage,
    interrupt_stage,
    missing_stage_products,
    open_run,
    record_environment,
    record_event,
    run_summary,
    stage_products_ok,
)
from .config import (
    default_config_path,
    load_config,
    parse_overrides,
    validate_run_id,
)
from .types import (
    DataError,
    EvaluationError,
    RunContext,
    RunError,
    UAVDetError,
    WeightError,
)

__all__ = ["main", "build_parser"]

PROG = "uavdet"

#: Exit code for a figure run that produced only part of the applicable set.
EXIT_PARTIAL = 7


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------
def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", metavar="PATH", default=argparse.SUPPRESS,
                        help="YAML config (defaults to configs/project.yaml)")
    parser.add_argument("--override", metavar="SECTION.KEY=VALUE", action="append",
                        default=argparse.SUPPRESS,
                        help="dotted config override, repeatable (wins over the file)")
    parser.add_argument("--data-root", metavar="PATH", default=None,
                        help="override data.root (dataset mount point)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="VisDrone UAV small-object detection: audit, train, evaluate, "
                    "benchmark, plot and infer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python -m uavdet --config configs/project.yaml check --run-id preflight_001\n"
            "  python -m uavdet --config configs/smoke.yaml pipeline --run-id smoke_001 --device 0\n"
            "  python -m uavdet --config configs/project.yaml plot --run-id formal_001\n"
            "  python -m uavdet.app   # desktop GUI (needs PyQt6)\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"{PROG} {__version__}")
    _add_common(parser)
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # -- check --------------------------------------------------------------
    check = sub.add_parser("check", help="audit data/environment, never trains or downloads")
    check.add_argument("--run-id", required=True)
    check.add_argument("--device", default=None, help="device recorded in the audit (not used)")
    check.add_argument("--allow-fail", action="store_true",
                       help="exit 0 even when the audit fails (for inspection only)")
    _add_common(check)

    # -- train --------------------------------------------------------------
    train = sub.add_parser("train", help="train YOLO11s from the configured pretrained weights")
    train.add_argument("--run-id", required=True)
    train.add_argument("--device", default=None)
    train.add_argument("--resume", metavar="LAST_PT", default=None,
                       help="resume the same run from its last.pt checkpoint")
    train.add_argument("--epochs", type=int, default=None, help="override train.epochs")
    train.add_argument("--batch", type=int, default=None, help="override train.batch")
    _add_common(train)

    # -- val ----------------------------------------------------------------
    val = sub.add_parser("val", help="run the native Ultralytics validation and export raw arrays")
    val.add_argument("--run-id", required=True)
    val.add_argument("--weights", default=None,
                     help="weights to evaluate (default: this run's train/weights/best.pt; "
                          "last.pt is used only with a logged warning when best.pt is absent)")
    val.add_argument("--split", choices=("val", "test"), default=None)
    val.add_argument("--device", default=None)
    val.add_argument("--parent-run-id", default=None,
                     help="run whose training produced --weights (revalidation audits)")
    val.add_argument("--force", action="store_true",
                     help="re-run a completed evaluation in the same run (default: refuse; "
                          "prefer a new --run-id with --parent-run-id)")
    _add_common(val)

    # -- benchmark ----------------------------------------------------------
    bench = sub.add_parser("benchmark", help="timing benchmark with a fixed image list")
    bench.add_argument("--run-id", required=True)
    bench.add_argument("--weights", default=None)
    bench.add_argument("--device", default=None)
    bench.add_argument("--images", type=int, default=None, help="override benchmark.num_images")
    bench.add_argument("--force", action="store_true",
                       help="re-run a completed benchmark in the same run (default: refuse)")
    _add_common(bench)

    # -- plot ---------------------------------------------------------------
    plot = sub.add_parser("plot", help="regenerate report figures from saved results only")
    plot.add_argument("--run-id", required=True)
    plot.add_argument("--figure-tag", default=None, help="sub-directory under figures/")
    _add_common(plot)

    # -- predict ------------------------------------------------------------
    predict = sub.add_parser("predict", help="headless image inference (deployment check)")
    predict.add_argument("--weights", required=True)
    predict.add_argument("--source", required=True, help="image file or directory")
    predict.add_argument("--out", default=None, help="output directory")
    predict.add_argument("--device", default=None)
    predict.add_argument("--conf", type=float, default=None)
    predict.add_argument("--iou", type=float, default=None)
    predict.add_argument("--max-det", type=int, default=None)
    predict.add_argument("--imgsz", type=int, default=None)
    _add_common(predict)

    # -- pipeline -----------------------------------------------------------
    pipe = sub.add_parser("pipeline", help="check -> train -> val -> benchmark -> plot")
    pipe.add_argument("--run-id", required=True)
    pipe.add_argument("--device", default=None)
    pipe.add_argument("--figure-tag", default=None)
    pipe.add_argument("--continue", dest="continue_", action="store_true",
                      help="skip stages that are complete AND whose products exist")
    pipe.add_argument("--epochs", type=int, default=None)
    pipe.add_argument("--batch", type=int, default=None)
    _add_common(pipe)

    # -- fetch-weights ------------------------------------------------------
    fetch = sub.add_parser("fetch-weights",
                           help="explicitly download official YOLO11 pretrained weights")
    fetch.add_argument("--name", default="yolo11s.pt")
    fetch.add_argument("--dir", default=None, help="target directory (default: <root>/weights)")
    fetch.add_argument("--force", action="store_true", help="overwrite an existing file")
    _add_common(fetch)

    # -- info ---------------------------------------------------------------
    info = sub.add_parser("info", help="print resolved config, paths and installed versions")
    _add_common(info)

    return parser


# ---------------------------------------------------------------------------
# config / run plumbing
# ---------------------------------------------------------------------------
def _resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    config_path = getattr(args, "config", None)
    path = Path(config_path) if config_path else default_config_path()
    overrides = parse_overrides(getattr(args, "override", None))
    data_root = getattr(args, "data_root", None)
    if data_root:
        overrides["data.root"] = data_root
    device = getattr(args, "device", None)
    if device:
        for section in ("train", "eval", "predict", "benchmark"):
            overrides[f"{section}.device"] = device
    epochs = getattr(args, "epochs", None)
    if epochs is not None:
        overrides["train.epochs"] = int(epochs)
    batch = getattr(args, "batch", None)
    if batch is not None:
        overrides["train.batch"] = int(batch)
    for attribute, key in (("conf", "predict.conf"), ("iou", "predict.iou"),
                           ("imgsz", "predict.imgsz")):
        value = getattr(args, attribute, None)
        if value is not None:
            overrides[key] = value
    max_det = getattr(args, "max_det", None)
    if max_det is not None:
        overrides["predict.max_det"] = int(max_det)
    images = getattr(args, "images", None)
    if images is not None:
        overrides["benchmark.num_images"] = int(images)
    return load_config(path, overrides)


def _run_id(args: argparse.Namespace) -> str:
    return validate_run_id(str(getattr(args, "run_id")))


def _log(message: str) -> None:
    print(message, flush=True)


def _audit_or_fail(config: Mapping[str, Any], run: RunContext) -> dict[str, Any]:
    """Run the dataset audit and persist it; raises DataError on failure."""
    from .data import audit_dataset, save_audit

    audit = audit_dataset(config, progress=_log)
    save_audit(audit, run)
    if audit["status"] != "ok":
        details = "; ".join(audit.get("failures", [])[:5])
        raise DataError(
            f"dataset audit failed with {len(audit.get('failures') or [])} problem(s): "
            f"{details}{' …' if len(audit.get('failures') or []) > 5 else ''}. "
            f"See {run.audit_dir / 'issues.csv'}"
        )
    return audit


def _audit_reuse_problems(saved: Mapping[str, Any], config: Mapping[str, Any]) -> list[str]:
    """Why a saved audit cannot be reused for the current data configuration.

    A completed audit is only valid for the data root and content it was made
    from. Reusing it after a mount/root change is exactly how a wrong cloud path
    would slip through a gate, so the root, the expected counts, the class order
    and a deterministic sample of file hashes are re-checked here.
    """
    from .artifacts import sha256_file

    problems: list[str] = []
    if Path(str(saved.get("data_root") or "")).resolve() != Path(config["data"]["root"]).resolve():
        problems.append(
            f"data root changed: audit={saved.get('data_root')} now={config['data']['root']}"
        )
    expected = saved.get("expected_images") or {}
    for split_key, split_name in (("expected_train_images", config["data"]["train_split"]),
                                  ("expected_val_images", config["data"]["val_split"])):
        if str(config["data"][split_key]) != str(expected.get(str(split_name))):
            problems.append(
                f"expected {split_name} image count changed: audit={expected.get(split_name)} "
                f"now={config['data'][split_key]}"
            )
    if list(saved.get("class_names") or []) != list(config["data"]["class_names"]):
        problems.append("class order changed since the audit")
    entries = list(saved.get("images") or [])
    if entries:
        by_split: dict[str, list[Mapping[str, Any]]] = {}
        for entry in entries:
            by_split.setdefault(str(entry.get("split")), []).append(entry)
        for split, split_entries in by_split.items():
            ordered = sorted(split_entries, key=lambda item: str(item.get("image_id")))
            if not ordered:
                continue
            sample = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
            for entry in sample:
                for key, rel in (("image_sha256", entry.get("image_rel")),
                                 ("label_sha256", entry.get("label_rel"))):
                    digest = entry.get(key)
                    if not digest or not rel:
                        continue
                    path = Path(config["data"]["root"]) / str(rel)
                    if not path.is_file():
                        problems.append(f"{split}/{entry.get('image_id')}: {rel} is missing")
                    elif sha256_file(path) != digest:
                        problems.append(f"{split}/{entry.get('image_id')}: {rel} content changed")
    return problems


def _require_audit(config: Mapping[str, Any], run: RunContext) -> dict[str, Any]:
    """Reuse a complete audit only if it still matches the data, else re-audit."""
    from .data import load_audit

    if run.stage_status("audit") == "complete":
        try:
            saved = load_audit(run)
            problems = _audit_reuse_problems(saved, config) if saved["status"] == "ok" else [
                f"saved audit status is {saved['status']!r}"
            ]
            if not problems:
                return saved
            record_event(run, "audit_reuse_rejected",
                         "saved audit does not match the current data; re-auditing",
                         problems=problems[:10])
            _log("[audit] saved audit rejected: " + "; ".join(problems[:3]))
        except (DataError, OSError, ValueError) as exc:
            _log(f"[audit] cannot reuse the saved audit ({exc}); re-auditing")
    begin_stage(run, "audit", allow_repeat=True)
    try:
        audit = _audit_or_fail(config, run)
        record_environment(run, config["train"]["device"])
        complete_stage(
            run, "audit",
            status="ok",
            manifest_digest=audit["manifest_digest"],
            num_images=audit["totals"]["num_images"],
            num_instances=audit["totals"]["num_instances"],
            splits={split: block["num_images"] for split, block in audit["splits"].items()},
        )
        return audit
    except BaseException as exc:  # includes KeyboardInterrupt -> interrupted
        if isinstance(exc, KeyboardInterrupt):
            interrupt_stage(run, "audit", "interrupted by the operator")
        else:
            fail_stage(run, "audit", str(exc))
        raise


def _prepare_view(config: Mapping[str, Any], run: RunContext, audit: Mapping[str, Any]) -> Path:
    from .data import prepare_dataset_view

    data_yaml = prepare_dataset_view(audit, run, config)
    record_event(run, "dataset_view", "writable dataset view ready", data_yaml=str(data_yaml))
    return data_yaml


def _refuse_existing_figure_tag(run: RunContext, figure_tag: str) -> None:
    """Check the tag BEFORE any stage transition (a typo must not flip the stage)."""
    manifest = run.figures_dir(figure_tag) / "figure_manifest.json"
    if manifest.exists():
        raise RunError(
            f"figures/{figure_tag} already has a figure_manifest.json; refusing to overwrite "
            f"it. Use a new --figure-tag (a style/font change also needs a new tag)."
        )


def _ensure_split_available(config: Mapping[str, Any], split: str) -> None:
    """Fail fast when the requested split is not defined by the dataset."""
    train_split = str(config["data"]["train_split"])
    val_split = str(config["data"]["val_split"])
    if str(split) in (train_split, val_split):
        return
    import yaml as _yaml

    data_yaml_path = Path(config["data"]["root"]) / "data.yaml"
    available: list[str] = []
    if data_yaml_path.exists():
        payload = _yaml.safe_load(data_yaml_path.read_text(encoding="utf-8")) or {}
        available = [key for key in ("train", "val", "test") if key in payload]
    raise DataError(
        f"split {split!r} is not configured for this dataset (available: {available or ['train', 'val']}). "
        f"The shipped VisDrone copy has train/val only; add a 'test' entry to data.yaml, set "
        f"data.val_split, and document the purpose before validating on it."
    )


def _default_weights(run: RunContext) -> Path:
    """This run's ``best.pt``; ``last.pt`` only with an explicit, logged warning."""
    best = run.train_dir / "weights" / "best.pt"
    if best.exists():
        return best
    last = run.train_dir / "weights" / "last.pt"
    if last.exists():
        message = ("best.pt is missing, so last.pt is used for evaluation; this is NOT the "
                   "checkpoint selection rule of the formal protocol")
        _log(f"[val] warning: {message}")
        record_event(run, "weights_fallback", message, weights=str(last))
        return last
    raise WeightError(
        f"run '{run.run_id}' has no trained weights yet; pass --weights explicitly "
        f"or run the 'train' command first"
    )


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def cmd_check(args: argparse.Namespace) -> int:
    config = _resolve_config(args)
    run_id = _run_id(args)
    run = open_run(config, run_id)
    from .devices import resolve_device

    choice = resolve_device(config["train"]["device"])
    for warning in choice.warnings:
        _log(f"[check] warning: {warning}")
        record_event(run, "device_fallback", warning,
                     requested=choice.requested, effective=choice.effective)
    _log(f"[check] run={run_id} scope={config['scope']} data={config['data']['root']}")
    try:
        _require_audit(config, run)
    except DataError as exc:
        if args.allow_fail:
            _log(f"[check] audit failed (allowed): {exc}")
            return 0
        raise
    summary = run_summary(run)
    _log(f"[check] ok — stages: {summary['stages']}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .training import train_model

    config = _resolve_config(args)
    run_id = _run_id(args)
    resume = Path(args.resume).resolve() if args.resume else None
    run = open_run(config, run_id)
    _log(f"[train] run={run_id} scope={config['scope']} resume={resume or 'no'}")
    audit = _require_audit(config, run)
    _prepare_view(config, run, audit)
    begin_stage(run, "train", allow_repeat=bool(resume))
    try:
        result = train_model(config, run, resume=resume)
        complete_stage(run, "train", **result)
        _log(f"[train] complete — best={result.get('best_weights')} epochs={result.get('epochs_done')}")
        return 0
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            interrupt_stage(run, "train", "interrupted by the operator")
        else:
            fail_stage(run, "train", str(exc))
        raise


def cmd_val(args: argparse.Namespace) -> int:
    from .evaluation import evaluate_model

    config = _resolve_config(args)
    run_id = _run_id(args)
    run = open_run(config, run_id, parent_run_id=getattr(args, "parent_run_id", None))
    if run.stage_complete("evaluation") and not getattr(args, "force", False):
        raise RunError(
            f"run '{run_id}' already has a complete evaluation. Re-validate in a NEW run "
            f"(python -m uavdet ... val --run-id reval_002 --weights ... "
            f"--parent-run-id {run_id}); pass --force only for an explicitly recorded re-run."
        )
    if args.split:
        config = load_config(
            Path(config["meta"]["config_path"]),
            _collect_overrides(args, extra={"eval.split": args.split}),
        )
    _ensure_split_available(config, config["eval"]["split"])
    audit = _require_audit(config, run)
    data_yaml = _prepare_view(config, run, audit)
    weights = Path(args.weights).resolve() if args.weights else _default_weights(run)
    _log(f"[val] run={run_id} weights={weights} split={config['eval']['split']}")
    begin_stage(run, "evaluation", allow_repeat=bool(getattr(args, "force", False)))
    try:
        result = evaluate_model(config, run, weights, split=config["eval"]["split"],
                                data_yaml=data_yaml)
        complete_stage(run, "evaluation", **result)
        _log(f"[val] done — mAP50={result.get('map50'):.4f} mAP50-95={result.get('map50_95'):.4f} "
             f"images={result.get('num_images')}")
        return 0
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            interrupt_stage(run, "evaluation", "interrupted by the operator")
        else:
            fail_stage(run, "evaluation", str(exc))
        raise


def cmd_benchmark(args: argparse.Namespace) -> int:
    from .benchmark import run_benchmark

    config = _resolve_config(args)
    run_id = _run_id(args)
    run = open_run(config, run_id)
    if run.stage_complete("benchmark") and not getattr(args, "force", False):
        raise RunError(
            f"run '{run_id}' already has a complete benchmark; use a new --run-id (or --force "
            f"for an explicitly recorded re-run)."
        )
    audit = _require_audit(config, run)
    data_yaml = _prepare_view(config, run, audit)
    weights = Path(args.weights).resolve() if args.weights else _default_weights(run)
    _log(f"[benchmark] run={run_id} weights={weights}")
    begin_stage(run, "benchmark", allow_repeat=bool(getattr(args, "force", False)))
    try:
        result = run_benchmark(config, run, weights, data_yaml=data_yaml)
        complete_stage(run, "benchmark", **result)
        _log(f"[benchmark] mean={result.get('mean_ms'):.2f} ms "
             f"p95={result.get('p95_ms'):.2f} ms fps={result.get('fps'):.1f}")
        return 0
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            interrupt_stage(run, "benchmark", "interrupted by the operator")
        else:
            fail_stage(run, "benchmark", str(exc))
        raise


def cmd_plot(args: argparse.Namespace) -> int:
    from .report import build_figures

    config = _resolve_config(args)
    run_id = _run_id(args)
    run = open_run(config, run_id)
    figure_tag = args.figure_tag or config["figures"]["tag"]
    _log(f"[plot] run={run_id} tag={figure_tag}")
    _refuse_existing_figure_tag(run, figure_tag)
    begin_stage(run, "figures", allow_repeat=True)
    try:
        result = build_figures(run, figure_tag=figure_tag)
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            interrupt_stage(run, "figures", "interrupted by the operator")
        else:
            fail_stage(run, "figures", str(exc))
        raise
    status = result.get("status")
    if status == "ok":
        complete_stage(run, "figures", **{k: v for k, v in result.items() if k != "figures"})
        _log(f"[plot] {len(result.get('produced') or [])} figure(s) written to "
             f"{run.figures_dir(figure_tag)}")
        return 0
    fail_stage(run, "figures", f"figure status={status}; missing={result.get('missing')}")
    _log(f"[plot] incomplete: status={status}")
    for item in result.get("missing") or []:
        _log(f"  missing {item.get('figure_id')}: {item.get('reason')}")
    return EXIT_PARTIAL


def cmd_predict(args: argparse.Namespace) -> int:
    from .inference import Detector, save_predictions

    config = _resolve_config(args)
    weights = Path(args.weights).resolve()
    options_overrides = {
        "conf": args.conf, "iou": args.iou, "max_det": args.max_det, "imgsz": args.imgsz,
        "device": config["predict"]["device"],
    }
    from .types import InferenceOptions

    options = InferenceOptions.from_config(
        config, "predict", **{k: v for k, v in options_overrides.items() if v is not None}
    )
    out_dir = Path(args.out) if args.out else Path(config["project"]["output_root"]) / "predict" / time.strftime("%Y%m%dT%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    source = Path(args.source)
    if not source.exists():
        raise UAVDetError(f"--source does not exist: {source}")
    images = [source] if source.is_file() else sorted(
        p for p in source.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    )
    if not images:
        raise UAVDetError(f"no images found under {source}")
    _log(f"[predict] weights={weights} images={len(images)} conf={options.conf} "
         f"iou={options.iou} max_det={options.max_det} imgsz={options.imgsz}")
    detector = Detector(weights, device=options.device)
    try:
        results = save_predictions(detector, images, out_dir, options)
    finally:
        detector.close()
    _log(f"[predict] wrote {len(results)} result(s) to {out_dir}")
    return 0


def cmd_pipeline(args: argparse.Namespace) -> int:
    from .benchmark import run_benchmark
    from .evaluation import evaluate_model
    from .report import build_figures
    from .training import train_model

    config = _resolve_config(args)
    run_id = _run_id(args)
    run = open_run(config, run_id)
    figure_tag = args.figure_tag or config["figures"]["tag"]
    _log(f"[pipeline] run={run_id} scope={config['scope']} continue={args.continue_}")

    planned = ["audit", "train", "evaluation", "benchmark", "figures"]
    if args.continue_:
        for stage in planned:
            if run.stage_status(stage) != "complete":
                continue
            missing = missing_stage_products(run, stage, figure_tag)
            if not missing:
                _log(f"[pipeline] stage '{stage}' already complete with all products — skipping")
                continue
            raise RunError(
                f"stage '{stage}' is marked complete but its product(s) {missing} are missing. "
                f"--continue never rewrites a completed stage: restore the file(s), or start a "
                f"new run for that stage."
            )

    # 1. audit -------------------------------------------------------------
    if not (args.continue_ and stage_products_ok(run, "audit")):
        _require_audit(config, run)
        _log("[pipeline] audit complete")
    audit = _require_audit(config, run)
    data_yaml = _prepare_view(config, run, audit)

    # 2. train -------------------------------------------------------------
    if not (args.continue_ and stage_products_ok(run, "train")):
        begin_stage(run, "train")
        try:
            result = train_model(config, run)
            complete_stage(run, "train", **result)
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                interrupt_stage(run, "train", "interrupted by the operator")
            else:
                fail_stage(run, "train", str(exc))
            raise
        _log("[pipeline] train complete")
    else:
        _log("[pipeline] train stage reused")

    weights = run.train_dir / "weights" / "best.pt"
    if not weights.exists():
        raise WeightError(f"training finished without {weights}")

    # 3. evaluation --------------------------------------------------------
    if not (args.continue_ and stage_products_ok(run, "evaluation")):
        begin_stage(run, "evaluation")
        try:
            result = evaluate_model(config, run, weights, split=config["eval"]["split"],
                                    data_yaml=data_yaml)
            expected = int(config["data"]["expected_val_images"])
            if config["scope"] == "formal" and int(result.get("num_images") or 0) != expected:
                raise EvaluationError(
                    f"formal pipeline requires a complete validation over {expected} images, but "
                    f"{result.get('num_images')} were evaluated; the stage is not complete"
                )
            complete_stage(run, "evaluation", **result)
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                interrupt_stage(run, "evaluation", "interrupted by the operator")
            else:
                fail_stage(run, "evaluation", str(exc))
            raise
        _log("[pipeline] evaluation complete")
    else:
        _log("[pipeline] evaluation stage reused")

    # 4. benchmark ---------------------------------------------------------
    if not (args.continue_ and stage_products_ok(run, "benchmark")):
        begin_stage(run, "benchmark")
        try:
            result = run_benchmark(config, run, weights, data_yaml=data_yaml)
            requested = int(result.get("requested_images") or result.get("num_images") or 0)
            available = int(result.get("available_images") or result.get("num_images") or 0)
            if config["scope"] == "formal" and available < requested:
                raise UAVDetError(
                    f"formal pipeline requires the fixed benchmark list ({requested} images) but "
                    f"the dataset view only has {available}; the stage is not complete"
                )
            complete_stage(run, "benchmark", **result)
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                interrupt_stage(run, "benchmark", "interrupted by the operator")
            else:
                fail_stage(run, "benchmark", str(exc))
            raise
        _log("[pipeline] benchmark complete")
    else:
        _log("[pipeline] benchmark stage reused")

    # 5. figures -----------------------------------------------------------
    if args.continue_ and stage_products_ok(run, "figures", figure_tag):
        _log(f"[pipeline] figures stage reused (tag '{figure_tag}')")
        _log("[pipeline] complete")
        return 0
    _refuse_existing_figure_tag(run, figure_tag)
    begin_stage(run, "figures", allow_repeat=True)
    try:
        result = build_figures(run, figure_tag=figure_tag)
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            interrupt_stage(run, "figures", "interrupted by the operator")
        else:
            fail_stage(run, "figures", str(exc))
        raise
    if result.get("status") != "ok":
        fail_stage(run, "figures", f"status={result.get('status')}")
        for item in result.get("missing") or []:
            _log(f"[pipeline] missing figure {item.get('figure_id')}: {item.get('reason')}")
        return EXIT_PARTIAL
    complete_stage(run, "figures", **{k: v for k, v in result.items() if k != "figures"})
    _log(f"[pipeline] complete — {len(result.get('produced') or [])} figure(s) in "
         f"{run.figures_dir(figure_tag)}")
    return 0


def cmd_fetch_weights(args: argparse.Namespace) -> int:
    config = _resolve_config(args)
    name = str(args.name)
    if not name.endswith(".pt"):
        name = f"{name}.pt"
    target_dir = Path(args.dir) if args.dir else Path(config["project"]["root"]) / "weights"
    target = target_dir / name
    if target.exists() and not args.force:
        from .artifacts import sha256_file

        _log(f"[fetch-weights] already present: {target} sha256={sha256_file(target)}")
        return 0
    target_dir.mkdir(parents=True, exist_ok=True)
    downloaded = _download_pretrained(name, target_dir)
    if downloaded is None:
        raise WeightError(
            f"could not download {name}. No file was created, so no random-initialised "
            f"'pretrained' run can happen by accident. Place the official weights at "
            f"{target} manually and re-run."
        )
    from .artifacts import sha256_file

    _log(f"[fetch-weights] downloaded {downloaded} sha256={sha256_file(downloaded)}")
    return 0


def _download_pretrained(name: str, target_dir: Path) -> Path | None:
    """Download official YOLO11 weights via Ultralytics' own asset resolver."""
    try:
        from ultralytics.utils.downloads import attempt_download_asset
    except Exception as exc:  # pragma: no cover - needs ultralytics
        _log(f"[fetch-weights] ultralytics download helper unavailable: {exc}")
        return None
    try:
        path = attempt_download_asset(name)
    except Exception as exc:
        _log(f"[fetch-weights] download failed: {exc}")
        return None
    source = Path(str(path))
    if not source.exists():
        return None
    target = target_dir / name
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
        # Ultralytics downloads next to the cwd when it cannot find the file;
        # do not leave a second 19 MB copy of the same weights behind.
        if source.parent.resolve() == Path.cwd().resolve():
            source.unlink(missing_ok=True)
    return target


def cmd_info(args: argparse.Namespace) -> int:
    from .artifacts import tools_summary

    config = _resolve_config(args)
    payload = {
        "uavdet_version": __version__,
        "scope": config["scope"],
        "config": config,
        "tools": tools_summary(),
        "runs": None,
    }
    from .artifacts import list_runs

    payload["runs"] = list_runs(config["project"]["output_root"])
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
_COMMANDS = {
    "check": cmd_check,
    "train": cmd_train,
    "val": cmd_val,
    "benchmark": cmd_benchmark,
    "plot": cmd_plot,
    "predict": cmd_predict,
    "pipeline": cmd_pipeline,
    "fetch-weights": cmd_fetch_weights,
    "info": cmd_info,
}


def _collect_overrides(args: argparse.Namespace, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    overrides = parse_overrides(getattr(args, "override", None))
    if getattr(args, "data_root", None):
        overrides["data.root"] = args.data_root
    if getattr(args, "device", None):
        for section in ("train", "eval", "predict", "benchmark"):
            overrides[f"{section}.device"] = args.device
    for key, value in (extra or {}).items():
        overrides[key] = value
    return overrides


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    handler = _COMMANDS.get(args.command)
    if handler is None:  # pragma: no cover - argparse guards this
        parser.error(f"unknown command {args.command}")
    if not getattr(args, "config", None):
        args.config = str(default_config_path())
    try:
        return int(handler(args) or 0)
    except UAVDetError as exc:
        print(f"[error] {type(exc).__name__}: {exc}", file=sys.stderr)
        return int(getattr(exc, "exit_code", 1))
    except FileNotFoundError as exc:
        print(f"[error] missing file: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("[error] interrupted", file=sys.stderr)
        return 130
