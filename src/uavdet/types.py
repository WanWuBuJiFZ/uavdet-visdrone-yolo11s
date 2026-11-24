"""Qt-free data types shared by the CLI, the figures and the desktop app.

Everything in this module must stay importable without Qt, torch, ultralytics
or OpenCV so that ``uavdet --help``, ``check`` and ``plot`` work on a bare
CPU/cloud machine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "CLASS_NAMES",
    "CLASS_NAMES_CN",
    "CLASS_COLORS_BGR",
    "CLASS_COLORS_HEX",
    "NUM_CLASSES",
    "STAGE_NAMES",
    "STAGE_STATES",
    "STAGE_STATUS_PENDING",
    "STAGE_STATUS_RUNNING",
    "STAGE_STATUS_COMPLETE",
    "STAGE_STATUS_FAILED",
    "STAGE_STATUS_INTERRUPTED",
    "STAGE_STATUS_SKIPPED",
    "UAVDetError",
    "ConfigError",
    "DataError",
    "RunError",
    "WeightError",
    "EvaluationError",
    "AdapterError",
    "Detection",
    "PredictionResult",
    "InferenceOptions",
    "RunContext",
]

# ---------------------------------------------------------------------------
# Fixed VisDrone2019-DET class order (0-9). Never reorder: labels, metrics,
# figures and the GUI all rely on this sequence.
# ---------------------------------------------------------------------------
CLASS_NAMES: tuple[str, ...] = (
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
)
NUM_CLASSES: int = len(CLASS_NAMES)

CLASS_NAMES_CN: tuple[str, ...] = (
    "行人",
    "人群",
    "自行车",
    "小汽车",
    "面包车",
    "卡车",
    "三轮车",
    "带篷三轮车",
    "公交车",
    "摩托车",
)

# Annotation colours for OpenCV drawing (BGR order on purpose, cv2 is BGR).
CLASS_COLORS_BGR: tuple[tuple[int, int, int], ...] = (
    (43, 87, 255),
    (0, 195, 255),
    (83, 200, 0),
    (243, 150, 33),
    (176, 39, 156),
    (0, 152, 255),
    (212, 188, 0),
    (99, 30, 233),
    (80, 175, 76),
    (183, 58, 103),
)

# Colourblind-safe class palette for report figures (Okabe-Ito extended).
CLASS_COLORS_HEX: tuple[str, ...] = (
    "#0072B2",
    "#009E73",
    "#E69F00",
    "#D55E00",
    "#56B4E9",
    "#CC79A7",
    "#F0E442",
    "#8C6D31",
    "#4B4B4B",
    "#7E57C2",
)

# ---------------------------------------------------------------------------
# Experiment stages / states
# ---------------------------------------------------------------------------
STAGE_NAMES: tuple[str, ...] = ("audit", "train", "evaluation", "benchmark", "figures")

STAGE_STATUS_PENDING = "pending"
STAGE_STATUS_RUNNING = "running"
STAGE_STATUS_COMPLETE = "complete"
STAGE_STATUS_FAILED = "failed"
STAGE_STATUS_INTERRUPTED = "interrupted"
STAGE_STATUS_SKIPPED = "skipped"

STAGE_STATES: tuple[str, ...] = (
    STAGE_STATUS_PENDING,
    STAGE_STATUS_RUNNING,
    STAGE_STATUS_COMPLETE,
    STAGE_STATUS_FAILED,
    STAGE_STATUS_INTERRUPTED,
    STAGE_STATUS_SKIPPED,
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class UAVDetError(Exception):
    """Base class for all project errors (message is user facing)."""

    exit_code = 1


class ConfigError(UAVDetError):
    """Invalid configuration, unknown key or illegal run id."""

    exit_code = 2


class DataError(UAVDetError):
    """Dataset audit failed or a dataset view cannot be prepared."""

    exit_code = 3


class RunError(UAVDetError):
    """Run directory / stage state problem (e.g. re-running a completed stage)."""

    exit_code = 4


class WeightError(UAVDetError):
    """Missing, unverified or wrong-class-count weights."""

    exit_code = 5


class EvaluationError(UAVDetError):
    """Native validation or metric export failed."""

    exit_code = 6


class AdapterError(EvaluationError):
    """The installed Ultralytics version does not expose a required field."""


# ---------------------------------------------------------------------------
# Detection results
# ---------------------------------------------------------------------------
def _finite(value: float, what: str) -> float:
    """Return ``value`` as float, raising ValueError for NaN/inf."""
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{what} is not finite: {value!r}")
    return value


@dataclass(frozen=True)
class Detection:
    """One detected box in *original image pixel* coordinates.

    Values are kept at full precision here; rounding only happens when a table
    or CSV is rendered. ``xyxy`` is ``(x1, y1, x2, y2)``.
    """

    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.class_id, int) or isinstance(self.class_id, bool):
            raise ValueError(f"class_id must be int, got {self.class_id!r}")
        if not (0 <= self.class_id < 255):
            raise ValueError(f"class_id out of range: {self.class_id}")
        coords = tuple(_finite(v, "xyxy") for v in self.xyxy)
        if len(coords) != 4:
            raise ValueError(f"xyxy needs 4 numbers, got {len(coords)}")
        x1, y1, x2, y2 = coords
        if x2 < x1 or y2 < y1:
            raise ValueError(f"degenerate box: {coords}")
        object.__setattr__(self, "xyxy", coords)
        object.__setattr__(self, "confidence", _finite(self.confidence, "confidence"))

    # -- derived geometry (all in original-image pixels) -------------------
    @property
    def x1(self) -> float:
        return self.xyxy[0]

    @property
    def y1(self) -> float:
        return self.xyxy[1]

    @property
    def x2(self) -> float:
        return self.xyxy[2]

    @property
    def y2(self) -> float:
        return self.xyxy[3]

    @property
    def width(self) -> float:
        return self.xyxy[2] - self.xyxy[0]

    @property
    def height(self) -> float:
        return self.xyxy[3] - self.xyxy[1]

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict (full precision, no rounding)."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "xyxy": [self.xyxy[0], self.xyxy[1], self.xyxy[2], self.xyxy[3]],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Detection":
        xyxy = data["xyxy"]
        return cls(
            class_id=int(data["class_id"]),
            class_name=str(data["class_name"]),
            confidence=float(data["confidence"]),
            xyxy=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
        )


@dataclass
class PredictionResult:
    """All detections for one image.

    An image with zero detections is still a valid result: ``detections`` is
    then an empty list and the image id is preserved, never dropped.
    """

    image_id: str
    source: str
    width: int
    height: int
    detections: list[Detection] = field(default_factory=list)
    timing: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.width = int(self.width)
        self.height = int(self.height)
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"invalid image size {self.width}x{self.height}")
        if not isinstance(self.detections, list):
            self.detections = list(self.detections)

    @property
    def count(self) -> int:
        return len(self.detections)

    def class_counts(self) -> dict[str, int]:
        """Per-class counts keyed by class name, fixed 0-9 order, zeros kept."""
        counts = {name: 0 for name in CLASS_NAMES}
        for det in self.detections:
            counts[det.class_name] = counts.get(det.class_name, 0) + 1
        return counts

    def sorted_detections(self) -> list[Detection]:
        """Highest confidence first (stable for equal confidences)."""
        return sorted(self.detections, key=lambda d: d.confidence, reverse=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "source": self.source,
            "width": self.width,
            "height": self.height,
            "num_detections": self.count,
            "detections": [d.to_dict() for d in self.detections],
            "timing": {k: float(v) for k, v in self.timing.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PredictionResult":
        return cls(
            image_id=str(data["image_id"]),
            source=str(data.get("source", "")),
            width=int(data["width"]),
            height=int(data["height"]),
            detections=[Detection.from_dict(d) for d in data.get("detections", [])],
            timing={str(k): float(v) for k, v in (data.get("timing") or {}).items()},
        )


@dataclass(frozen=True)
class InferenceOptions:
    """Explicit inference parameters used by *every* call path.

    The desktop app, the CLI ``predict`` command and the benchmark all build
    one of these from configuration, so no code path can silently fall back to
    Ultralytics defaults.
    """

    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.70
    max_det: int = 300
    device: str = "cpu"
    half: bool = False
    verbose: bool = False

    def __post_init__(self) -> None:
        if int(self.imgsz) <= 0 or int(self.imgsz) % 32 != 0:
            raise ValueError(f"imgsz must be a positive multiple of 32, got {self.imgsz}")
        if not (0.0 <= float(self.conf) <= 1.0):
            raise ValueError(f"conf must be within [0, 1], got {self.conf}")
        if not (0.0 < float(self.iou) <= 1.0):
            raise ValueError(f"iou must be within (0, 1], got {self.iou}")
        if int(self.max_det) < 1:
            raise ValueError(f"max_det must be >= 1, got {self.max_det}")

    def to_ultralytics_kwargs(self) -> dict[str, Any]:
        """Keyword arguments handed to ``model.predict``/``model.val``."""
        return {
            "imgsz": int(self.imgsz),
            "conf": float(self.conf),
            "iou": float(self.iou),
            "max_det": int(self.max_det),
            "device": str(self.device),
            "half": bool(self.half),
        }

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        section: str = "predict",
        **overrides: Any,
    ) -> "InferenceOptions":
        """Build from ``config[section]``; ``overrides`` win."""
        block = dict(config.get(section) or {})
        values = {
            "imgsz": block.get("imgsz", 640),
            "conf": block.get("conf", 0.25),
            "iou": block.get("iou", 0.70),
            "max_det": block.get("max_det", 300),
            "device": block.get("device", "cpu"),
            "half": block.get("half", False),
        }
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)


@dataclass
class RunContext:
    """Handle for one experiment directory ``outputs/<run_id>/``.

    Only path bookkeeping lives here; manifest/stage transitions are in
    :mod:`uavdet.artifacts` so that this class stays import-cheap.
    """

    run_id: str
    run_dir: Path
    config: dict[str, Any]
    scope: str = "formal"
    manifest: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        self.scope = str(self.scope or self.manifest.get("scope", "formal"))

    # -- standard sub-directories ------------------------------------------
    @property
    def audit_dir(self) -> Path:
        return self.run_dir / "audit"

    @property
    def train_dir(self) -> Path:
        return self.run_dir / "train"

    @property
    def evaluation_dir(self) -> Path:
        return self.run_dir / "evaluation"

    @property
    def benchmark_dir(self) -> Path:
        return self.run_dir / "benchmark"

    @property
    def logs_dir(self) -> Path:
        return self.run_dir / "logs"

    def figures_dir(self, figure_tag: str = "v1") -> Path:
        return self.run_dir / "figures" / str(figure_tag)

    # -- files ---------------------------------------------------------------
    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    @property
    def environment_path(self) -> Path:
        return self.run_dir / "environment.json"

    @property
    def resolved_config_path(self) -> Path:
        return self.run_dir / "config_resolved.yaml"

    # -- helpers -------------------------------------------------------------
    def ensure_dirs(self) -> None:
        for path in (
            self.run_dir,
            self.audit_dir,
            self.train_dir,
            self.evaluation_dir,
            self.benchmark_dir,
            self.logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def stage_status(self, stage: str) -> str:
        stage_block = (self.manifest.get("stages") or {}).get(stage) or {}
        return str(stage_block.get("status", STAGE_STATUS_PENDING))

    def stage_block(self, stage: str) -> dict[str, Any]:
        return dict((self.manifest.get("stages") or {}).get(stage) or {})

    def stage_complete(self, stage: str) -> bool:
        return self.stage_status(stage) == STAGE_STATUS_COMPLETE

    def rel(self, path: str | Path) -> str:
        """Path relative to the run dir when possible (posix separators)."""
        candidate = Path(path)
        try:
            return candidate.resolve().relative_to(self.run_dir.resolve()).as_posix()
        except (ValueError, OSError):
            return candidate.as_posix()

    # Stage transitions delegate to artifacts (avoids an import cycle).
    def begin_stage(self, stage: str, allow_repeat: bool = False) -> dict[str, Any]:
        from . import artifacts

        return artifacts.begin_stage(self, stage, allow_repeat=allow_repeat)

    def complete_stage(self, stage: str, **summary: Any) -> dict[str, Any]:
        from . import artifacts

        return artifacts.complete_stage(self, stage, **summary)

    def fail_stage(self, stage: str, error: str) -> dict[str, Any]:
        from . import artifacts

        return artifacts.fail_stage(self, stage, error)

    def interrupt_stage(self, stage: str, error: str) -> dict[str, Any]:
        from . import artifacts

        return artifacts.interrupt_stage(self, stage, error)


def load_vocab(names: Sequence[str]) -> tuple[str, ...]:
    """Validate and freeze a class-name sequence (used for config sanity)."""
    names = tuple(str(n) for n in names)
    if len(names) != NUM_CLASSES:
        raise ConfigError(
            f"data.class_names must contain exactly {NUM_CLASSES} names, got {len(names)}"
        )
    return names


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU of two ``(x1, y1, x2, y2)`` boxes (pixels, any consistent unit)."""
    ax1, ay1, ax2, ay2 = (float(v) for v in a[:4])
    bx1, by1, bx2, by2 = (float(v) for v in b[:4])
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def iter_chunks(items: Iterable[Any], size: int) -> Iterable[list[Any]]:
    """Yield lists of at most ``size`` items."""
    chunk: list[Any] = []
    for item in items:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk
