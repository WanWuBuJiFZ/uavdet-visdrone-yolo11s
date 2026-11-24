"""Fixed-protocol inference speed measurement.

Protocol (设计规范 §6.5, 验收清单 C06):

* a deterministic list of ``benchmark.num_images`` validation images: the view's
  ``val`` entries sorted by image id, then a seeded sample, then re-sorted;
* every image is decoded **before** the timer starts (disk read and JPEG decode
  are not part of the number that gets reported);
* the model is loaded once, outside the timer, followed by ``warmup`` untimed
  calls;
* then ``repeats`` passes over the list, batch size 1, FP32;
* the timer wraps exactly ``preprocess -> model -> postprocess -> structured
  result`` (one ``Detector.predict`` call) and nothing else: no weight loading,
  no decoding, no drawing, no file writing;
* GPU/MPS runs synchronise around each timed call so the measured interval is the
  real device work; CPU runs do not need it.

The summary keeps the raw per-image timings in ``timings.csv`` so mean/P50/P95
can be recomputed, and it records the exact hardware and thresholds it used.
"""

from __future__ import annotations

import csv
import io
import platform
import random
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .artifacts import now_iso, record_event, sha256_file, write_json_atomic, write_text_atomic
from .devices import resolve_device
from .training import resolve_data_yaml
from .types import (
    InferenceOptions,
    RunContext,
    RunError,
    UAVDetError,
    WeightError,
)

__all__ = ["run_benchmark", "default_detector_factory"]

#: What the measured interval contains, stored verbatim in summary.json.
TIMING_SCOPE = (
    "decoded original image -> preprocess -> model -> postprocess -> structured result; "
    "excludes weight loading, disk read, drawing and saving"
)

#: Benchmark always measures batch 1; the field exists so the artefact is explicit.
BENCHMARK_BATCH = 1


def default_detector_factory(weights: Path, device: str) -> Any:
    """The real engine (imported lazily so ``import uavdet.benchmark`` stays cheap)."""
    from .inference import Detector  # noqa: PLC0415 - optional/heavy dependency

    return Detector(weights, device=device)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _sync_device(device_kind: str) -> None:
    """Synchronise the accelerator so the timed interval is real device work."""
    if device_kind == "cuda":
        import torch  # noqa: PLC0415 - optional dependency

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    elif device_kind == "mps":
        import torch  # noqa: PLC0415 - optional dependency

        synchronise = getattr(getattr(torch, "mps", None), "synchronize", None)
        if synchronise is not None:
            synchronise()


def _decode_images(view_root: Path, entries: Sequence[Mapping[str, Any]]) -> list[np.ndarray]:
    import cv2  # noqa: PLC0415 - optional/heavy dependency

    decoded: list[np.ndarray] = []
    for entry in entries:
        path = view_root / str(entry["image_rel"])
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise UAVDetError(f"cannot decode benchmark image: {path}")
        decoded.append(image)
    return decoded


def _select_entries(
    entries: Sequence[Mapping[str, Any]],
    *,
    limit: int,
    seed: int,
    split: str,
) -> list[Mapping[str, Any]]:
    """Deterministic subset: sorted ids, seeded sample, re-sorted."""
    ordered = sorted(entries, key=lambda entry: str(entry["image_id"]))
    if limit >= len(ordered):
        return ordered
    identifiers = [str(entry["image_id"]) for entry in ordered]
    generator = random.Random(f"{seed}:{split}:benchmark")
    picked = set(generator.sample(identifiers, int(limit)))
    return [entry for entry in ordered if str(entry["image_id"]) in picked]


def _stats(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "fps": 0.0}
    mean = float(array.mean())
    return {
        "mean_ms": mean,
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "fps": float(1000.0 / mean) if mean > 0 else 0.0,
    }


def _hardware(device: str) -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "device": device,
    }
    try:
        import torch  # noqa: PLC0415 - optional dependency

        info["torch"] = str(torch.__version__)
        info["torch_threads"] = int(torch.get_num_threads())
        if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            info["cuda_device"] = str(torch.cuda.get_device_name(0))
        mps = getattr(torch.backends, "mps", None)
        info["mps_available"] = bool(mps is not None and mps.is_available())
    except Exception:  # noqa: BLE001 - a missing torch must not break reporting
        info["torch"] = None
    return info


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------
def run_benchmark(
    config: Mapping[str, Any],
    run: RunContext,
    weights: Path,
    *,
    data_yaml: Path | None = None,
    detector_factory: Callable[[Path, str], Any] | None = None,
) -> dict[str, Any]:
    """Measure the fixed protocol and write ``benchmark/{image_list,timings,summary}``."""
    weights = Path(weights)
    if not weights.is_file():
        raise WeightError(f"weights not found: {weights}")
    data_yaml_path = resolve_data_yaml(config, run, data_yaml)
    view_root = Path(data_yaml_path).resolve().parent

    benchmark_cfg = dict(config["benchmark"])
    split = str(config["data"]["val_split"])
    from .data import view_split_entries  # noqa: PLC0415 - avoid an import cycle

    entries = view_split_entries(data_yaml_path, split)
    if not entries:
        raise RunError(
            f"the dataset view at {view_root} lists no '{split}' image; run 'check' to build a "
            f"view before benchmarking"
        )
    requested_images = int(benchmark_cfg["num_images"])
    available_images = len(entries)
    chosen = _select_entries(
        entries,
        limit=requested_images,
        seed=int(benchmark_cfg["seed"]),
        split=split,
    )
    if not chosen:
        raise RunError("the benchmark image selection is empty")

    choice = resolve_device(benchmark_cfg["device"])
    warnings: list[str] = list(choice.warnings)
    if available_images < requested_images:
        # A smaller measurement set is a protocol change: say it loudly instead of
        # reporting "100 images" while only 10 were timed.
        shortfall = (
            f"benchmark.num_images={requested_images} but the '{split}' split of this dataset "
            f"view has only {available_images} image(s); {len(chosen)} image(s) were measured, "
            f"so this benchmark is smaller than the configured protocol and is not directly "
            f"comparable to a full run"
        )
        warnings.append(shortfall)
        record_event(
            run, "benchmark_shortfall", shortfall,
            requested_images=requested_images, available_images=available_images,
            measured_images=len(chosen),
        )
        print(f"[benchmark] warning: {shortfall}", flush=True)
    decoded = _decode_images(view_root, chosen)

    benchmark_dir = run.benchmark_dir
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        benchmark_dir / "image_list.json",
        {
            "schema": 1,
            "created_at": now_iso(),
            "run_id": run.run_id,
            "split": split,
            "seed": int(benchmark_cfg["seed"]),
            "num_images": len(chosen),
            "selection": (
                f"view split '{split}' entries sorted by image_id, then a random.Random("
                f"'{int(benchmark_cfg['seed'])}:{split}:benchmark') sample, then re-sorted"
            ),
            "images": [
                {
                    "image_id": str(entry["image_id"]),
                    "image_rel": str(entry["image_rel"]),
                    "source": str(view_root / str(entry["image_rel"])),
                    "width": entry.get("width"),
                    "height": entry.get("height"),
                    "num_objects": entry.get("num_objects"),
                }
                for entry in chosen
            ],
        },
    )

    factory = detector_factory or default_detector_factory
    device_kind = choice.kind
    options = _options(config, choice.effective)
    warmup = int(benchmark_cfg["warmup"])
    repeats = int(benchmark_cfg["repeats"])

    factory_calls = 0
    raw: list[dict[str, Any]] = []
    detector = None
    try:
        factory_calls += 1
        detector = factory(weights, choice.effective)
        loader = getattr(detector, "load", None)
        if callable(loader):
            loader()  # weight loading stays outside every timed interval

        for index in range(max(0, warmup)):
            entry = chosen[index % len(chosen)]
            detector.predict(
                decoded[index % len(decoded)], options,
                image_id=str(entry["image_id"]),
                source=str(view_root / str(entry["image_rel"])),
            )

        for repeat in range(repeats):
            for index, (entry, image) in enumerate(zip(chosen, decoded)):
                _sync_device(device_kind)
                started = time.perf_counter()
                result = detector.predict(
                    image,
                    options,
                    image_id=str(entry["image_id"]),
                    source=str(view_root / str(entry["image_rel"])),
                )
                _sync_device(device_kind)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                if result is None or not hasattr(result, "detections"):
                    raise UAVDetError(
                        "the detector returned no structured PredictionResult; the timed "
                        "interval would not cover preprocessing/model/postprocessing"
                    )
                raw.append(
                    {
                        "repeat": repeat,
                        "index": index,
                        "image_id": str(entry["image_id"]),
                        "ms": elapsed_ms,
                    }
                )
    finally:
        closer = getattr(detector, "close", None) if detector is not None else None
        if callable(closer):
            closer()

    if factory_calls != 1:
        raise UAVDetError("the benchmark must build the detector exactly once")
    if not raw:
        raise RunError("no timed iteration ran; check benchmark.repeats/num_images")

    values = [row["ms"] for row in raw]
    overall = _stats(values)
    per_repeat = []
    for repeat in range(repeats):
        repeat_values = [row["ms"] for row in raw if row["repeat"] == repeat]
        repeat_stats = _stats(repeat_values)
        per_repeat.append(
            {
                "repeat": repeat,
                "mean_ms": repeat_stats["mean_ms"],
                "p50_ms": repeat_stats["p50_ms"],
                "p95_ms": repeat_stats["p95_ms"],
            }
        )

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["repeat", "index", "image_id", "ms"])
    for row in raw:
        writer.writerow([row["repeat"], row["index"], row["image_id"], f"{row['ms']:.6f}"])
    write_text_atomic(benchmark_dir / "timings.csv", buffer.getvalue())

    summary: dict[str, Any] = {
        "status": "ok",
        "run_id": run.run_id,
        "weights_sha256": sha256_file(weights),
        "device": str(benchmark_cfg["device"]),
        "effective_device": choice.effective,
        "imgsz": int(options.imgsz),
        "conf": float(options.conf),
        "iou": float(options.iou),
        "max_det": int(options.max_det),
        "num_images": len(chosen),
        "requested_images": requested_images,
        "available_images": available_images,
        "warmup": warmup,
        "repeats": repeats,
        "batch": BENCHMARK_BATCH,
        "half": bool(options.half),
        "mean_ms": overall["mean_ms"],
        "p50_ms": overall["p50_ms"],
        "p95_ms": overall["p95_ms"],
        "fps": overall["fps"],
        "timing_scope": TIMING_SCOPE,
        "hardware": _hardware(choice.effective),
        "per_repeat": per_repeat,
        "num_measurements": len(raw),
        "data_yaml": str(data_yaml_path),
        "split": split,
        "warnings": warnings,
        "created_at": now_iso(),
    }
    write_json_atomic(benchmark_dir / "summary.json", summary)
    return summary


def _options(config: Mapping[str, Any], device: str) -> InferenceOptions:
    benchmark_cfg = dict(config["benchmark"])
    return InferenceOptions(
        imgsz=int(benchmark_cfg["imgsz"]),
        conf=float(benchmark_cfg["conf"]),
        iou=float(benchmark_cfg["iou"]),
        max_det=int(benchmark_cfg["max_det"]),
        device=str(device),
        half=bool(benchmark_cfg["half"]),
    )
