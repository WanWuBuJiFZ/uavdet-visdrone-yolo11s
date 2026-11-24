"""Qt-free detection engine used by the CLI ``predict`` command and the GUI.

Image convention (single rule, enforced at the boundary): a frame is an
OpenCV-style **BGR uint8** ``numpy`` array of shape ``(H, W, 3)``. It is handed to
``model.predict`` **unchanged** — Ultralytics treats a numpy ndarray source as
BGR, so converting with ``cv2.cvtColor`` here (or in the caller) would flip the
channels twice and produce wrong colours. ``annotate`` draws on a *copy* and
returns BGR as well.

Every prediction passes ``imgsz``/``conf``/``iou``/``max_det``/``device``
explicitly plus the configured precision, so no Ultralytics default can apply
silently. A requested device that does not exist falls back to CPU through
:func:`uavdet.devices.resolve_device`; the fallback is recorded in
``detector.warnings`` and in the ``run_meta.json`` written by
:func:`save_predictions` — never silently.

A model must expose the 10 fixed VisDrone classes; anything else raises
:class:`~uavdet.types.WeightError` and leaves the detector unloaded (a failed
load never leaves a half-initialised model behind).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .artifacts import now_iso, sha256_file, write_json_atomic, write_text_atomic
from .devices import DeviceChoice, resolve_device
from .types import (
    CLASS_COLORS_BGR,
    CLASS_NAMES,
    NUM_CLASSES,
    Detection,
    InferenceOptions,
    PredictionResult,
    UAVDetError,
    WeightError,
)

__all__ = [
    "Detector",
    "save_predictions",
    "default_model_factory",
    "label_rect",
    "label_font_scale",
]

#: Bounding-box colour of a class id that the palette does not know (never drawn
#: for a verified 10-class model).
FALLBACK_COLOR = (255, 255, 255)


def default_model_factory(weights: Path) -> Any:
    """Load the real Ultralytics model (imported lazily on purpose)."""
    from ultralytics import YOLO  # noqa: PLC0415 - optional/heavy dependency

    return YOLO(str(weights))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _supports_ultralytics_argument(name: str) -> bool | None:
    """Whether the installed Ultralytics cfg knows ``name``.

    ``None`` means "ultralytics is not importable", which only happens on a
    plotting-only machine or with a fake model factory in tests.
    """
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT  # noqa: PLC0415
    except Exception:  # noqa: BLE001 - no ultralytics: caller decides
        return None
    return name in DEFAULT_CFG_DICT


def _precision_kwargs(half: bool, warnings: list[str]) -> dict[str, Any]:
    """Carry the requested precision explicitly across Ultralytics versions.

    ``half`` was the historical argument; current versions replaced it with
    ``quantize`` (``None``/32 = FP32, 16 = FP16) and reject the unknown ``half``
    key. The requested precision is never dropped silently: the chosen form is
    recorded, and a ``half=True`` request that cannot be honoured is warned about.
    """
    if _supports_ultralytics_argument("half"):
        return {"half": bool(half)}
    if _supports_ultralytics_argument("quantize"):
        if half:
            warnings.append(
                "half precision was requested; the installed ultralytics uses the "
                "'quantize' argument, so quantize=16 is passed instead"
            )
        return {"quantize": 16 if half else None}
    return {"half": bool(half)}


def _to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros((0,), dtype=np.float64)
    if isinstance(value, np.ndarray):
        return value
    detach = getattr(value, "detach", None)
    if detach is not None:
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _model_names(model: Any) -> dict[int, str]:
    raw = getattr(model, "names", None)
    names: dict[int, str] = {}
    if isinstance(raw, Mapping) and raw:
        for key, value in raw.items():
            try:
                names[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
    elif isinstance(raw, (list, tuple)) and raw:
        names = {index: str(value) for index, value in enumerate(raw)}
    return names


def _validate_frame(image: Any) -> np.ndarray:
    """Reject anything that is not a BGR uint8 ``(H, W, 3)`` array."""
    if not isinstance(image, np.ndarray):
        raise UAVDetError(
            f"the detection engine expects a BGR uint8 numpy array (H, W, 3), got "
            f"{type(image).__name__}; decode/convert before calling predict()"
        )
    if image.dtype != np.uint8:
        raise UAVDetError(
            f"the detection engine expects uint8 pixels, got {image.dtype}; convert with "
            f"astype(np.uint8) after rescaling yourself"
        )
    if image.ndim != 3 or image.shape[2] != 3:
        raise UAVDetError(
            f"the detection engine expects an (H, W, 3) BGR array, got shape {image.shape}"
        )
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise UAVDetError(f"empty image of shape {image.shape}")
    return image


# ---------------------------------------------------------------------------
# label geometry (kept separate so the clamping is testable without drawing)
# ---------------------------------------------------------------------------
LABEL_FONT = 0.4
LABEL_MIN_FONT = 0.25
LABEL_MAX_FONT = 1.0
LABEL_PADDING = 4
#: Frame short side that maps to the maximum font scale (a 2000x1500 export gets a
#: legible label; a 64x48 preview keeps a compact one).
LABEL_FONT_REFERENCE = 900.0


def label_font_scale(text: str, width: int, height: int | None = None) -> float:
    """Font scale for a label: readable on the frame, then shrunk to fit ``width``.

    The base scale follows the frame size (a 2000x1500 export must stay readable,
    a 64x48 preview must not be dominated by text); it is then reduced -- never
    below ``LABEL_MIN_FONT`` -- until the string fits the image width.
    """
    import cv2  # noqa: PLC0415 - optional/heavy dependency

    scale = LABEL_FONT
    if height is not None and int(height) > 0:
        reference = min(int(width), int(height)) / LABEL_FONT_REFERENCE
        scale = float(min(LABEL_MAX_FONT, max(LABEL_FONT, reference)))
    if width <= 2 * LABEL_PADDING:
        return LABEL_MIN_FONT
    (text_width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    if text_width + 2 * LABEL_PADDING <= width:
        return scale
    fitted = scale * (width - 2 * LABEL_PADDING) / float(text_width + 2 * LABEL_PADDING)
    return float(max(LABEL_MIN_FONT, min(scale, fitted)))


def label_rect(
    x1: float,
    y1: float,
    y2: float,
    text_width: int,
    text_height: int,
    baseline: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Background rectangle ``(x, y, w, h)`` fully inside the image.

    The label is placed above the box; when there is no room above it moves below
    the box, and if neither fits it is clamped into the image. Horizontally it is
    shifted left instead of being clipped at the right edge, so a box touching
    ``x=0`` (very common in VisDrone) still shows the whole class name.
    """
    width = int(width)
    height = int(height)
    box_width = min(int(text_width) + 2 * LABEL_PADDING, width)
    box_height = min(int(text_height) + int(baseline) + 2 * LABEL_PADDING, height)

    x0 = int(round(float(x1)))
    if x0 + box_width > width:
        x0 = width - box_width
    x0 = max(0, min(x0, max(0, width - box_width)))

    y0 = int(round(float(y1))) - box_height
    if y0 < 0:
        below = int(round(float(y2)))
        if below + box_height <= height:
            y0 = below
        else:
            y0 = int(round(float(y1)))
    y0 = max(0, min(y0, max(0, height - box_height)))
    return x0, y0, box_width, box_height


# ---------------------------------------------------------------------------
# detector
# ---------------------------------------------------------------------------
class Detector:
    """One model, one device, explicit inference parameters."""

    def __init__(
        self,
        weights: str | Path,
        device: str = "cpu",
        *,
        model_factory: Callable[[Path], Any] | None = None,
        name_map: Mapping[int, str] | None = None,
    ) -> None:
        self.weights = Path(weights)
        self.requested_device = str(device)
        self.effective_device = "cpu"
        self.model_factory = model_factory
        self.name_map = {int(key): str(value) for key, value in (name_map or {}).items()}
        self.warnings: list[str] = []
        self.model: Any = None
        self.device_choice: DeviceChoice | None = None
        self._names: dict[int, str] = {}
        self._loaded = False
        self._load_stats: dict[str, Any] = {}

    # -- state -------------------------------------------------------------
    @property
    def loaded(self) -> bool:
        return bool(self._loaded and self.model is not None)

    @property
    def names(self) -> dict[int, str]:
        return dict(self._names)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"Detector(weights={self.weights.name!r}, device={self.requested_device!r}, "
            f"effective={self.effective_device!r}, loaded={self.loaded})"
        )

    # -- lifecycle ---------------------------------------------------------
    def load(self) -> dict[str, Any]:
        """Load the weights; returns a JSON-safe statistics dictionary."""
        if self.loaded:
            return dict(self._load_stats)
        if not self.weights.is_file():
            raise WeightError(
                f"weights not found: {self.weights}; a 10-class VisDrone checkpoint is required"
            )

        choice = resolve_device(self.requested_device)
        self.device_choice = choice
        self.effective_device = choice.effective
        for message in choice.warnings:
            if message not in self.warnings:
                self.warnings.append(message)

        factory = self.model_factory or default_model_factory
        try:
            model = factory(self.weights)
        except Exception as exc:  # noqa: BLE001 - surfaced as WeightError
            self._reset()
            raise WeightError(
                f"cannot load weights {self.weights}: {type(exc).__name__}: {exc}"
            ) from exc

        names = _model_names(model)
        if len(names) != NUM_CLASSES:
            self._reset()
            raise WeightError(
                f"{self.weights.name} describes {len(names)} class(es), but this project "
                f"requires the {NUM_CLASSES} VisDrone classes {list(CLASS_NAMES)}; refusing to "
                f"run an 80-class COCO model as if it were the project model"
            )
        unknown = sorted(set(names) - set(range(NUM_CLASSES)))
        if unknown:
            self._reset()
            raise WeightError(
                f"{self.weights.name} uses class ids {unknown}, which are outside the fixed "
                f"0-{NUM_CLASSES - 1} VisDrone order"
            )
        if self.name_map:
            outside = sorted(set(self.name_map) - set(range(NUM_CLASSES)))
            if outside:
                raise WeightError(f"name_map contains class ids outside 0-{NUM_CLASSES - 1}: {outside}")
            names = {class_id: self.name_map.get(class_id, name) for class_id, name in names.items()}
        if tuple(names[index] for index in range(NUM_CLASSES)) != CLASS_NAMES and not self.name_map:
            self.warnings.append(
                f"{self.weights.name} uses the class names "
                f"{[names[index] for index in range(NUM_CLASSES)]} instead of the VisDrone names; "
                f"the class order (0-9) is still the one the project uses"
            )

        self.model = model
        self._names = names
        self._loaded = True
        self._load_stats = {
            "weights": str(self.weights),
            "weights_sha256": sha256_file(self.weights),
            "num_classes": len(names),
            "names": {str(key): value for key, value in sorted(names.items())},
            "device": {
                "requested": choice.requested,
                "effective": choice.effective,
                "kind": choice.kind,
                "warnings": list(choice.warnings),
            },
            "warnings": list(self.warnings),
        }
        return dict(self._load_stats)

    def _reset(self) -> None:
        self.model = None
        self._names = {}
        self._loaded = False
        self._load_stats = {}

    def close(self) -> None:
        """Release the model (idempotent; a later ``load``/``predict`` reloads)."""
        self._reset()

    def __enter__(self) -> "Detector":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    # -- inference ---------------------------------------------------------
    def inference_options(self, options: InferenceOptions | None = None) -> InferenceOptions:
        """``options`` or the device-correct default of this detector."""
        if options is not None:
            return options
        device = self.effective_device if self.loaded else resolve_device(self.requested_device).effective
        return InferenceOptions(device=str(device))

    def predict(
        self,
        image_bgr: np.ndarray,
        options: InferenceOptions | None = None,
        *,
        image_id: str = "array",
        source: str = "",
    ) -> PredictionResult:
        """Run one image through the model and return original-pixel boxes."""
        image = _validate_frame(image_bgr)
        if not self.loaded:
            self.load()
        resolved = self.inference_options(options)
        kwargs = self._predict_kwargs(resolved)
        started = time.perf_counter()
        # The BGR array is passed through unchanged: Ultralytics reads ndarray
        # sources as BGR, so any cvtColor here would be a double conversion.
        raw_results = self.model.predict(source=image, **kwargs)
        wall_ms = (time.perf_counter() - started) * 1000.0
        results = list(raw_results) if isinstance(raw_results, (list, tuple)) else [raw_results]
        result = results[0] if results else None
        if result is not None and not hasattr(result, "boxes"):
            raise UAVDetError(
                f"model.predict returned {type(result).__name__} instead of an Ultralytics "
                f"Results object with a 'boxes' attribute; the detection result cannot be read"
            )
        height, width = int(image.shape[0]), int(image.shape[1])
        orig_shape = getattr(result, "orig_shape", None)
        if orig_shape is not None and len(tuple(orig_shape)) >= 2:
            height, width = int(orig_shape[0]), int(orig_shape[1])

        detections = self._parse_detections(result, limit=int(resolved.max_det))
        timing = self._timing(result, wall_ms)
        return PredictionResult(
            image_id=str(image_id),
            source=str(source),
            width=width,
            height=height,
            detections=detections,
            timing=timing,
        )

    def _predict_kwargs(self, options: InferenceOptions) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "imgsz": int(options.imgsz),
            "conf": float(options.conf),
            "iou": float(options.iou),
            "max_det": int(options.max_det),
            "device": str(self.effective_device or options.device),
            "verbose": False,
        }
        kwargs.update(_precision_kwargs(bool(options.half), self.warnings))
        return kwargs

    def _parse_detections(self, result: Any, *, limit: int) -> list[Detection]:
        if result is None:
            return []
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        xyxy = _to_numpy(getattr(boxes, "xyxy", None)).astype(np.float64).reshape(-1, 4)
        confidences = _to_numpy(getattr(boxes, "conf", None)).astype(np.float64).reshape(-1)
        classes = _to_numpy(getattr(boxes, "cls", None)).astype(np.float64).reshape(-1)
        count = min(len(xyxy), len(confidences), len(classes))
        parsed: list[Detection] = []
        for index in range(count):
            coords = [float(value) for value in xyxy[index]]
            if not all(np.isfinite(value) for value in coords):
                continue
            class_id = int(round(float(classes[index])))
            name = self._names.get(class_id, f"class_{class_id}")
            try:
                parsed.append(
                    Detection(
                        class_id=class_id,
                        class_name=name,
                        confidence=float(confidences[index]),
                        xyxy=(coords[0], coords[1], coords[2], coords[3]),
                    )
                )
            except ValueError as exc:
                self.warnings.append(f"dropped an invalid box from the model output: {exc}")
        # Highest confidence first, then the configured max_det cap (the native
        # NMS already applies it; this keeps a fake/other backend honest too).
        parsed.sort(key=lambda detection: detection.confidence, reverse=True)
        if limit > 0 and len(parsed) > limit:
            parsed = parsed[:limit]
        return parsed

    def _timing(self, result: Any, wall_ms: float) -> dict[str, float]:
        timing: dict[str, float] = {}
        speed = getattr(result, "speed", None) if result is not None else None
        if isinstance(speed, Mapping) and speed:
            total = 0.0
            for key in ("preprocess", "inference", "postprocess"):
                value = speed.get(key)
                if value is None:
                    continue
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(number):
                    timing[f"{key}_ms"] = number
                    total += number
            if timing:
                timing["total_ms"] = total
        timing["wall_ms"] = float(wall_ms)
        return timing

    # -- drawing -----------------------------------------------------------
    def annotate(
        self,
        image_bgr: np.ndarray,
        detections: Sequence[Detection],
        options: InferenceOptions | None = None,
    ) -> np.ndarray:
        """Draw boxes on a BGR copy and return it (input stays untouched)."""
        import cv2  # noqa: PLC0415 - optional/heavy dependency

        image = _validate_frame(image_bgr)
        canvas = np.array(image, copy=True)
        height, width = int(canvas.shape[0]), int(canvas.shape[1])
        min_conf = float(options.conf) if options is not None else 0.0
        for detection in detections:
            if detection.confidence < min_conf:
                continue
            colour = (CLASS_COLORS_BGR[detection.class_id]
                      if 0 <= detection.class_id < len(CLASS_COLORS_BGR) else FALLBACK_COLOR)
            x1 = int(max(0, min(width - 1, round(detection.x1))))
            y1 = int(max(0, min(height - 1, round(detection.y1))))
            x2 = int(max(0, min(width - 1, round(detection.x2))))
            y2 = int(max(0, min(height - 1, round(detection.y2))))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
            label = f"{detection.class_name} {detection.confidence:.2f}"
            scale = label_font_scale(label, width, height)
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1
            )
            label_x, label_y, label_w, label_h = label_rect(
                x1, y1, y2, text_width, text_height, baseline, width, height
            )
            cv2.rectangle(
                canvas,
                (label_x, label_y),
                (label_x + label_w, label_y + label_h),
                colour,
                -1,
            )
            # The text origin is the baseline: keep it inside the rectangle.
            text_y = min(label_y + label_h - 2,
                         label_y + text_height + max(1, baseline // 2) + 1)
            cv2.putText(
                canvas,
                label,
                (label_x + LABEL_PADDING // 2, max(text_height, text_y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        return canvas


# ---------------------------------------------------------------------------
# batch export
# ---------------------------------------------------------------------------
def save_predictions(
    detector: Detector,
    images: Sequence[str | Path],
    out_dir: str | Path,
    options: InferenceOptions,
) -> list[PredictionResult]:
    """Run and save an annotated image, a CSV, both JSON files and the run meta."""
    import cv2  # noqa: PLC0415 - optional/heavy dependency

    target = Path(out_dir)
    annotated_dir = target / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    results: list[PredictionResult] = []
    rows: list[list[Any]] = []
    used_ids: dict[str, int] = {}

    for source in images:
        path = Path(source)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise UAVDetError(f"cannot decode image: {path}")
        image_id = path.stem
        if image_id in used_ids:
            used_ids[image_id] += 1
            # Two files with the same stem must not share a stable id or a file name.
            image_id = f"{image_id}_{used_ids[image_id]}"
        else:
            used_ids[image_id] = 0
        result = detector.predict(
            image, options, image_id=image_id, source=str(path)
        )
        annotated = detector.annotate(image, result.detections, options)
        annotated_path = annotated_dir / f"{image_id}.jpg"
        if not cv2.imwrite(str(annotated_path), annotated):
            raise UAVDetError(f"cannot write {annotated_path}")
        results.append(result)
        if result.detections:
            for detection in result.detections:
                rows.append([
                    result.image_id, result.source, result.count,
                    detection.class_id, detection.class_name,
                    f"{detection.confidence:.6f}",
                    f"{detection.x1:.3f}", f"{detection.y1:.3f}",
                    f"{detection.x2:.3f}", f"{detection.y2:.3f}",
                ])
        else:
            # An image with zero detections is still an output row (never dropped).
            rows.append([result.image_id, result.source, 0, "", "", "", "", "", "", ""])

    header = ["image_id", "source", "num_detections", "class_id", "class_name",
              "confidence", "x1", "y1", "x2", "y2"]
    csv_lines = [",".join(header)]
    for row in rows:
        csv_lines.append(",".join(_csv_cell(value) for value in row))
    write_text_atomic(target / "detections.csv", "\n".join(csv_lines) + "\n")

    load_stats = dict(detector._load_stats) if detector.loaded else {}
    choice = detector.device_choice
    device_block = {
        "requested": choice.requested if choice is not None else detector.requested_device,
        "effective": choice.effective if choice is not None else detector.effective_device,
        "kind": choice.kind if choice is not None else "cpu",
        "warnings": list(detector.warnings),
    }
    options_block = {
        "imgsz": int(options.imgsz),
        "conf": float(options.conf),
        "iou": float(options.iou),
        "max_det": int(options.max_det),
        "half": bool(options.half),
    }
    write_json_atomic(
        target / "predictions.json",
        {
            "schema": 1,
            "created_at": now_iso(),
            "weights": str(detector.weights),
            "weights_sha256": load_stats.get("weights_sha256"),
            "class_names": [detector.names.get(index, CLASS_NAMES[index]) for index in range(NUM_CLASSES)],
            "options": options_block,
            "device": device_block,
            "num_images": len(results),
            "num_detections": sum(result.count for result in results),
            "results": [result.to_dict() for result in results],
        },
    )
    write_json_atomic(
        target / "run_meta.json",
        {
            "schema": 1,
            "created_at": now_iso(),
            "weights": str(detector.weights),
            "weights_sha256": load_stats.get("weights_sha256"),
            "num_classes": load_stats.get("num_classes"),
            "options": options_block,
            "device": device_block,
            "warnings": list(detector.warnings),
            "num_images": len(results),
            "num_detections": sum(result.count for result in results),
            "timing_ms": _aggregate_timing(results),
            "annotated_dir": "annotated",
            "software": {"opencv": getattr(cv2, "__version__", "unknown")},
        },
    )
    return results


def _csv_cell(value: Any) -> str:
    text = str(value)
    if any(character in text for character in (",", '"', "\n")):
        return '"' + text.replace('"', '""') + '"'
    return text


def _aggregate_timing(results: Sequence[PredictionResult]) -> dict[str, float]:
    totals = [float(result.timing.get("wall_ms", 0.0)) for result in results]
    if not totals:
        return {}
    return {
        "num_images": len(totals),
        "sum_wall_ms": float(sum(totals)),
        "mean_wall_ms": float(sum(totals) / len(totals)),
        "max_wall_ms": float(max(totals)),
    }
