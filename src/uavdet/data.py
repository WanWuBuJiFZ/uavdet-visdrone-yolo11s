"""Dataset audit and the writable dataset view.

Two responsibilities:

1. :func:`audit_dataset` — read-only inspection of the shipped VisDrone copy:
   image/label pairing, decodability, class/coordinate validity, empty labels,
   per-class counts, objects per image, relative object-size distribution,
   content duplicates inside and across splits.
2. :func:`prepare_dataset_view` — copy the audited (and, for ``scope=smoke``,
   deterministically selected) files into ``workspace/data_views/<digest>`` and
   generate a data.yaml that points there. Ultralytics may write label caches,
   so the original ``dataset_visdrone/`` directory is never handed to the
   training library.

The original input directory is never written to: the audit only opens files
for reading and the view is a separate copy.
"""

from __future__ import annotations

import csv
import hashlib
import math
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from .artifacts import (
    now_iso,
    read_json,
    sha256_file,
    write_json_atomic,
    write_text_atomic,
)
from .types import (
    CLASS_NAMES,
    NUM_CLASSES,
    DataError,
    RunContext,
)

__all__ = [
    "IMAGE_SUFFIXES",
    "LABEL_TOLERANCE",
    "audit_dataset",
    "save_audit",
    "load_audit",
    "dataset_stats",
    "prepare_dataset_view",
    "load_view_manifest",
    "view_split_entries",
    "split_dir_paths",
    "read_image_size",
    "label_row_to_xyxy",
    "normalized_box_to_pixels",
    "dataset_manifest_digest",
    "LabelLine",
    "LABEL_REPAIR_POLICIES",
]

IMAGE_SUFFIXES: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
LABEL_TOLERANCE = 1e-6
LABEL_REPAIR_POLICIES = ("strict", "drop_degenerate_boxes")
# Shared log-spaced bins for the relative-area (w_norm * h_norm) distribution of
# F02. Both splits use these edges; the figure only reads these numbers.
AREA_BIN_COUNT = 120
AREA_MIN = 1e-7
AREA_MAX = 1.0


# ---------------------------------------------------------------------------
# split resolution
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SplitPaths:
    split: str
    images_dir: Path
    labels_dir: Path


def _read_data_yaml(data_root: Path) -> dict[str, Any]:
    yaml_path = data_root / "data.yaml"
    if not yaml_path.exists():
        return {}
    try:
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise DataError(f"cannot parse {yaml_path}: {exc}") from exc
    return raw if isinstance(raw, dict) else {}


def split_dir_paths(data_root: str | Path, split: str) -> SplitPaths:
    """Resolve ``<split>/images`` and ``<split>/labels`` for a split name.

    Uses ``data.yaml`` (``path`` + ``train``/``val`` entries) when present and
    falls back to the ``VisDrone2019-DET-<split>/images`` convention.
    """
    root = Path(data_root)
    meta = _read_data_yaml(root)
    base = root
    if meta.get("path"):
        base = (root / str(meta["path"])).resolve() if not Path(str(meta["path"])).is_absolute() else Path(str(meta["path"]))
    entry = meta.get(split)
    if isinstance(entry, str):
        images_dir = Path(entry)
        if not images_dir.is_absolute():
            images_dir = base / images_dir
    else:
        images_dir = base / f"VisDrone2019-DET-{split}" / "images"
    images_dir = images_dir.resolve()
    labels_dir = (images_dir.parent / "labels").resolve()
    return SplitPaths(split=split, images_dir=images_dir, labels_dir=labels_dir)


# ---------------------------------------------------------------------------
# label parsing
# ---------------------------------------------------------------------------
@dataclass
class LabelLine:
    """One non-empty label line with its parse result.

    ``problem`` is ``None`` for a usable line. ``degenerate`` marks a line whose
    only defect is a zero-width/height box: such a line is unusable (0 px area)
    and is the *only* defect the explicit ``drop_degenerate_boxes`` repair policy
    is allowed to remove.
    """

    line_no: int
    raw: str
    cls: int | None = None
    box: tuple[int, float, float, float, float] | None = None
    problem: str | None = None
    degenerate: bool = False


def _parse_label_file(path: Path) -> list[LabelLine]:
    """Parse one YOLO label file into per-line results.

    Raises :class:`DataError` when the file cannot be read; malformed lines are
    returned with a ``problem`` so the audit can report every position.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise DataError(f"cannot read label file {path}: {exc}") from exc
    results: list[LabelLine] = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        item = LabelLine(line_no=line_no, raw=line)
        results.append(item)
        parts = line.split()
        if len(parts) != 5:
            item.problem = f"line {line_no}: expected 5 values, got {len(parts)}"
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, w, h = (float(value) for value in parts[1:])
        except ValueError:
            item.problem = f"line {line_no}: non-numeric value(s) {line!r}"
            continue
        if not all(math.isfinite(v) for v in (cx, cy, w, h)):
            item.problem = f"line {line_no}: NaN/inf coordinate(s)"
            continue
        if w <= 0.0 or h <= 0.0:
            item.problem = f"line {line_no}: non-positive box size w={w} h={h}"
            item.degenerate = True
            continue
        if cls < 0 or cls >= NUM_CLASSES:
            item.problem = f"line {line_no}: class id {cls} outside 0-{NUM_CLASSES - 1}"
            continue
        x1, y1 = cx - w / 2.0, cy - h / 2.0
        x2, y2 = cx + w / 2.0, cy + h / 2.0
        if (
            cx < -LABEL_TOLERANCE or cx > 1 + LABEL_TOLERANCE
            or cy < -LABEL_TOLERANCE or cy > 1 + LABEL_TOLERANCE
            or x1 < -LABEL_TOLERANCE or y1 < -LABEL_TOLERANCE
            or x2 > 1 + LABEL_TOLERANCE or y2 > 1 + LABEL_TOLERANCE
        ):
            item.problem = (
                f"line {line_no}: box outside image bounds "
                f"xyxy=({x1:.4f},{y1:.4f},{x2:.4f},{y2:.4f})"
            )
            continue
        item.cls = cls
        item.box = (cls, cx, cy, w, h)
    return results


def _duplicate_annotation_lines(rows: Sequence[tuple[int, float, float, float, float]]) -> int:
    """Count lines that repeat the same class and (rounded) box."""
    seen: Counter[tuple[int, int, int, int, int]] = Counter()
    for cls, cx, cy, w, h in rows:
        key = (cls, round(cx, 4), round(cy, 4), round(w, 4), round(h, 4))
        seen[key] += 1
    return sum(count - 1 for count in seen.values() if count > 1)


def label_row_to_xyxy(cx: float, cy: float, w: float, h: float,
                      image_width: int, image_height: int) -> tuple[float, float, float, float]:
    """Convert one YOLO normalised row to original-image pixel ``xyxy``.

    Reference case (test A06): a 100x80 image with ``0 .5 .5 .2 .25`` yields
    ``(40.0, 30.0, 60.0, 50.0)`` — the single source of truth for the
    normalised-to-pixel convention used by GT export and case plots.
    """
    image_width = int(image_width)
    image_height = int(image_height)
    if image_width <= 0 or image_height <= 0:
        raise DataError(f"invalid image size {image_width}x{image_height}")
    box_width = float(w) * image_width
    box_height = float(h) * image_height
    centre_x = float(cx) * image_width
    centre_y = float(cy) * image_height
    return (
        centre_x - box_width / 2.0,
        centre_y - box_height / 2.0,
        centre_x + box_width / 2.0,
        centre_y + box_height / 2.0,
    )


def normalized_box_to_pixels(xyxy: Sequence[float], image_width: int,
                             image_height: int) -> tuple[float, float, float, float]:
    """Scale a normalised ``xyxy`` box to original-image pixels."""
    x1, y1, x2, y2 = (float(value) for value in xyxy[:4])
    return (x1 * float(image_width), y1 * float(image_height),
            x2 * float(image_width), y2 * float(image_height))


def read_image_size(path: Path, verify_decode: bool = True) -> tuple[int, int]:
    """Decode an image and return ``(width, height)``.

    ``Image.verify()`` only checks headers, so with ``verify_decode`` the pixels
    are fully loaded too (truncated JPEGs pass ``verify`` but fail here).
    """
    from PIL import Image

    with Image.open(path) as image:
        if verify_decode:
            image.load()
        width, height = image.size
    if width <= 0 or height <= 0:
        raise DataError(f"image reports an invalid size {width}x{height}: {path}")
    return int(width), int(height)


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------
def _collect_images(images_dir: Path) -> dict[str, Path]:
    if not images_dir.is_dir():
        raise DataError(f"images directory not found: {images_dir}")
    found: dict[str, Path] = {}
    for path in sorted(images_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            stem = path.stem
            if stem in found:
                raise DataError(
                    f"duplicate image stem {stem!r} in {images_dir}: "
                    f"{found[stem].name} and {path.name}"
                )
            found[stem] = path
    return found


def _area_edges() -> list[float]:
    log_min, log_max = math.log10(AREA_MIN), math.log10(AREA_MAX)
    step = (log_max - log_min) / AREA_BIN_COUNT
    return [round(10 ** (log_min + step * i), 12) for i in range(AREA_BIN_COUNT + 1)]


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    index = min(len(sorted_values) - 1, max(0, int(round(q * (len(sorted_values) - 1)))))
    return float(sorted_values[index])


def audit_dataset(
    config: Mapping[str, Any],
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Audit the dataset configured in ``config['data']`` (read-only).

    The returned dict is JSON-safe and is written by :func:`save_audit` into
    ``audit/data_manifest.json`` / ``audit/dataset_stats.json`` /
    ``audit/issues.csv``. ``status`` is ``"ok"`` only when no ``error`` level
    issue was found *and* the image counts match ``data.expected_*``.
    """
    data_cfg = config["data"]
    data_root = Path(data_cfg["root"])
    if not data_root.is_dir():
        raise DataError(f"data root not found: {data_root}")

    splits = [str(data_cfg["train_split"]), str(data_cfg["val_split"])]
    expected = {
        str(data_cfg["train_split"]): int(data_cfg["expected_train_images"]),
        str(data_cfg["val_split"]): int(data_cfg["expected_val_images"]),
    }
    class_names = list(data_cfg["class_names"])
    if len(class_names) != NUM_CLASSES:
        raise DataError(f"expected {NUM_CLASSES} class names, got {len(class_names)}")

    edges = _area_edges()
    label_repair = str(data_cfg.get("label_repair", "strict"))
    if label_repair not in LABEL_REPAIR_POLICIES:
        raise DataError(
            f"data.label_repair must be one of {list(LABEL_REPAIR_POLICIES)}, got {label_repair!r}"
        )
    repairs: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    manifest_entries: list[dict[str, Any]] = []
    split_stats: dict[str, dict[str, Any]] = {}
    split_paths: dict[str, dict[str, str]] = {}
    failures: list[str] = []

    image_sha_index: dict[str, list[tuple[str, str]]] = defaultdict(list)
    label_sha_index: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for split in splits:
        paths = split_dir_paths(data_root, split)
        split_paths[split] = {
            "images_dir": str(paths.images_dir),
            "labels_dir": str(paths.labels_dir),
        }
        if progress:
            progress(f"audit: scanning {split} images in {paths.images_dir}")
        images = _collect_images(paths.images_dir)
        label_files = {
            p.stem: p
            for p in sorted(paths.labels_dir.glob("*.txt"))
            if p.is_file()
        } if paths.labels_dir.is_dir() else {}

        per_class = [0] * NUM_CLASSES
        per_image_counts: list[dict[str, Any]] = []
        areas: list[float] = []
        widths: list[int] = []
        heights: list[int] = []
        empty_label_files = 0
        missing_label_files = 0
        bad_label_files = 0
        repaired_label_files = 0
        dropped_boxes = 0

        if (len(images) != expected[split]):
            failures.append(
                f"{split}: found {len(images)} images, expected {expected[split]} "
                f"(update data.expected_* only when the dataset really changed)"
            )
            issues.append({
                "split": split, "image_id": "-", "severity": "error",
                "kind": "image_count_mismatch",
                "detail": f"found {len(images)} images, expected {expected[split]}",
            })

        for index, stem in enumerate(sorted(images), start=1):
            image_path = images[stem]
            label_path = label_files.get(stem)
            entry: dict[str, Any] = {
                "split": split,
                "image_id": stem,
                "image_rel": image_path.relative_to(data_root).as_posix(),
                "label_rel": label_path.relative_to(data_root).as_posix() if label_path else None,
                "image_sha256": sha256_file(image_path),
                "label_sha256": sha256_file(label_path) if label_path else None,
                "width": None,
                "height": None,
                "num_objects": 0,
                "empty_label": False,
            }
            if progress and index % 500 == 0:
                progress(f"audit: {split} {index}/{len(images)}")

            try:
                width, height = read_image_size(image_path, bool(data_cfg.get("verify_image_decode", True)))
            except (DataError, OSError, ValueError) as exc:
                failures.append(f"{split}/{image_path.name}: not a decodable image ({exc})")
                issues.append({
                    "split": split, "image_id": stem, "severity": "error",
                    "kind": "corrupt_image", "detail": str(exc),
                })
                manifest_entries.append(entry)
                per_image_counts.append({"image_id": stem, "num_objects": None})
                continue
            entry["width"], entry["height"] = width, height
            widths.append(width)
            heights.append(height)

            rows: list[tuple[int, float, float, float, float]] = []
            if label_path is None:
                missing_label_files += 1
                empty_label_files += 1     # a missing label file means "zero objects"
                entry["empty_label"] = True
                issues.append({
                    "split": split, "image_id": stem, "severity": "warning",
                    "kind": "missing_label_file",
                    "detail": "no label file: treated as an image with zero objects",
                })
            else:
                try:
                    lines = _parse_label_file(label_path)
                except DataError as exc:
                    lines = [LabelLine(line_no=0, raw="", problem=str(exc))]
                file_dropped = 0
                file_bad = False
                for item in lines:
                    if item.problem is None and item.box is not None:
                        rows.append(item.box)
                        continue
                    if item.degenerate and label_repair == "drop_degenerate_boxes":
                        # Explicit, recorded repair: a 0-pixel box carries no object.
                        file_dropped += 1
                        repairs.append({
                            "split": split,
                            "image_id": stem,
                            "label_rel": label_path.relative_to(data_root).as_posix(),
                            "line_no": item.line_no,
                            "raw": item.raw,
                            "reason": item.problem,
                        })
                        issues.append({
                            "split": split, "image_id": stem, "severity": "warning",
                            "kind": "dropped_degenerate_box",
                            "detail": f"{item.problem} -> line dropped by label_repair policy",
                        })
                        continue
                    file_bad = True
                    failures.append(f"{split}/{label_path.name}: {item.problem}")
                    issues.append({
                        "split": split, "image_id": stem, "severity": "error",
                        "kind": "invalid_label", "detail": str(item.problem),
                    })
                if file_bad:
                    bad_label_files += 1
                if file_dropped:
                    repaired_label_files += 1
                    dropped_boxes += file_dropped
                if label_path.stat().st_size == 0 or not rows:
                    empty_label_files += 1
                    entry["empty_label"] = True

            duplicates = _duplicate_annotation_lines(rows) if rows else 0
            if duplicates:
                issues.append({
                    "split": split, "image_id": stem, "severity": "warning",
                    "kind": "duplicate_annotation",
                    "detail": f"{duplicates} repeated class+box line(s)",
                })

            for cls, _cx, _cy, w, h in rows:
                per_class[cls] += 1
                areas.append(w * h)
            entry["num_objects"] = len(rows)
            per_image_counts.append({"image_id": stem, "num_objects": len(rows)})
            manifest_entries.append(entry)
            if entry["image_sha256"]:
                image_sha_index[entry["image_sha256"]].append((split, stem))
            if entry["label_sha256"]:
                label_sha_index[entry["label_sha256"]].append((split, stem))

        # labels without an image
        for stem, label_path in sorted(label_files.items()):
            if stem not in images:
                failures.append(f"{split}/{label_path.name}: label without image")
                issues.append({
                    "split": split, "image_id": stem, "severity": "error",
                    "kind": "orphan_label",
                    "detail": f"label file {label_path.name} has no matching image",
                })

        counts_sorted = sorted(
            (item["num_objects"] for item in per_image_counts if item["num_objects"] is not None)
        )
        area_sorted = sorted(areas)
        hist_counts = [0] * AREA_BIN_COUNT
        log_edges = [math.log10(edge) for edge in edges]
        for area in areas:
            position = _bisect_log10(area, log_edges)
            hist_counts[position] += 1

        split_stats[split] = {
            "num_images": len(images),
            "num_instances": int(sum(per_class)),
            "num_label_files": len(label_files),
            "empty_label_images": empty_label_files,
            "missing_label_files": missing_label_files,
            "invalid_label_files": bad_label_files,
            "repaired_label_files": repaired_label_files,
            "dropped_boxes": dropped_boxes,
            "per_class": {name: per_class[i] for i, name in enumerate(CLASS_NAMES)},
            "per_class_list": per_class,
            "per_image_objects": [item["num_objects"] for item in per_image_counts],
            "per_image_ids": [item["image_id"] for item in per_image_counts],
            "objects_per_image": {
                "min": counts_sorted[0] if counts_sorted else 0,
                "max": counts_sorted[-1] if counts_sorted else 0,
                "mean": round(sum(counts_sorted) / len(counts_sorted), 4) if counts_sorted else 0.0,
                "median": _quantile(counts_sorted, 0.5) if counts_sorted else 0.0,
                "p95": _quantile(counts_sorted, 0.95) if counts_sorted else 0.0,
                "zeros": sum(1 for value in counts_sorted if value == 0),
            },
            "area_histogram": {
                "edges": edges,
                "counts": hist_counts,
                "definition": "relative area = (w_px / image_width) * (h_px / image_height), "
                              "log10-spaced bins at audit time",
                "binning": "log10",
                "bin_count": AREA_BIN_COUNT,
            },
            "area_quantiles": {
                "min": _quantile(area_sorted, 0.0) if area_sorted else 0.0,
                "p1": _quantile(area_sorted, 0.01) if area_sorted else 0.0,
                "p5": _quantile(area_sorted, 0.05) if area_sorted else 0.0,
                "p25": _quantile(area_sorted, 0.25) if area_sorted else 0.0,
                "p50": _quantile(area_sorted, 0.50) if area_sorted else 0.0,
                "p75": _quantile(area_sorted, 0.75) if area_sorted else 0.0,
                "p95": _quantile(area_sorted, 0.95) if area_sorted else 0.0,
                "p99": _quantile(area_sorted, 0.99) if area_sorted else 0.0,
                "max": _quantile(area_sorted, 1.0) if area_sorted else 0.0,
                "mean": round(sum(area_sorted) / len(area_sorted), 12) if area_sorted else 0.0,
            },
            "image_sizes": {
                "width": {"min": min(widths) if widths else None, "max": max(widths) if widths else None},
                "height": {"min": min(heights) if heights else None, "max": max(heights) if heights else None},
                "distinct_sizes": sorted({(w, h) for w, h in zip(widths, heights)}),
            },
        }
        if progress:
            progress(f"audit: {split} done — {len(images)} images, {sum(per_class)} objects")

    # --- content duplicates -------------------------------------------------
    duplicates: list[dict[str, Any]] = []
    for digest, owners in sorted(image_sha_index.items()):
        if len(owners) < 2:
            continue
        split_set = {split for split, _ in owners}
        kind = "cross_split_duplicate_image" if len(split_set) > 1 else "duplicate_image"
        duplicates.append({
            "kind": kind, "sha256": digest,
            "images": [f"{split}/{stem}" for split, stem in owners],
        })
        if kind == "cross_split_duplicate_image":
            failures.append(
                "identical image content appears in more than one split: "
                + ", ".join(f"{split}/{stem}" for split, stem in owners)
            )
        issues.append({
            "split": ",".join(sorted(split_set)), "image_id": owners[0][1],
            "severity": "error" if kind == "cross_split_duplicate_image" else "warning",
            "kind": kind,
            "detail": ", ".join(f"{split}/{stem}" for split, stem in owners),
        })
    for digest, owners in label_sha_index.items():
        if len(owners) < 2:
            continue
        first_split, first_stem = owners[0]
        for split, stem in owners[1:]:
            if (split, stem) != (first_split, first_stem):
                issues.append({
                    "split": split, "image_id": stem, "severity": "warning",
                    "kind": "duplicate_label_content",
                    "detail": f"label content identical to {first_split}/{first_stem}",
                })

    status = "ok" if not failures else "failed"
    audit = {
        "schema": 1,
        "created_at": now_iso(),
        "status": status,
        "scope": str(config.get("scope", "formal")),
        "data_root": str(data_root),
        "data_yaml": str(data_root / "data.yaml") if (data_root / "data.yaml").exists() else None,
        "class_names": class_names,
        "expected_images": expected,
        "split_paths": split_paths,
        "splits": split_stats,
        "totals": {
            "num_images": sum(s["num_images"] for s in split_stats.values()),
            "num_instances": sum(s["num_instances"] for s in split_stats.values()),
            "num_empty_label_images": sum(s["empty_label_images"] for s in split_stats.values()),
        },
        "issues": issues,
        "failures": failures,
        "duplicates": duplicates,
        "label_repair": {
            "policy": label_repair,
            "repaired_label_files": sum(
                block.get("repaired_label_files", 0) for block in split_stats.values()
            ),
            "dropped_boxes": sum(
                block.get("dropped_boxes", 0) for block in split_stats.values()
            ),
        },
        "repairs": repairs,
        "images": manifest_entries,
        "checks": {
            "image_label_pairing": True,
            "decode_all_images": bool(data_cfg.get("verify_image_decode", True)),
            "class_order": list(CLASS_NAMES),
            "label_tolerance": LABEL_TOLERANCE,
        },
    }
    audit["manifest_digest"] = dataset_manifest_digest(audit)
    return audit


def _bisect_log10(area: float, log_edges: Sequence[float]) -> int:
    """Index of the log10 bin for ``area`` (last bin is inclusive on the right)."""
    value = math.log10(area) if area > 0 else -300.0
    low, high = 0, len(log_edges) - 1
    # log_edges has bin_count + 1 entries.
    while low < high:
        middle = (low + high) // 2
        if value < log_edges[middle]:
            high = middle
        else:
            low = middle + 1
    return max(0, min(len(log_edges) - 2, low - 1))


def dataset_manifest_digest(audit: Mapping[str, Any]) -> str:
    """Stable digest over split/image/label content hashes only."""
    rows = []
    for entry in sorted(
        audit.get("images", []),
        key=lambda item: (item["split"], item["image_id"]),
    ):
        rows.append(
            "|".join([
                entry["split"],
                entry["image_id"],
                str(entry.get("image_sha256") or ""),
                str(entry.get("label_sha256") or ""),
            ])
        )
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------
ISSUE_FIELDS = ("split", "image_id", "severity", "kind", "detail")


def save_audit(audit: Mapping[str, Any], run: RunContext) -> dict[str, Path]:
    """Persist the audit into ``audit/`` (manifest, stats, issues)."""
    audit_dir = run.audit_dir
    audit_dir.mkdir(parents=True, exist_ok=True)

    stats = dataset_stats(audit)
    manifest = {
        key: audit[key]
        for key in ("schema", "created_at", "status", "scope", "data_root", "data_yaml",
                    "class_names", "expected_images", "split_paths", "totals",
                    "manifest_digest", "checks", "duplicates", "label_repair", "repairs")
        if key in audit
    }
    manifest["num_images_listed"] = len(audit.get("images", []))
    manifest["images"] = audit.get("images", [])

    write_json_atomic(audit_dir / "data_manifest.json", manifest)
    write_json_atomic(audit_dir / "dataset_stats.json", stats)

    issue_path = audit_dir / "issues.csv"
    if audit.get("issues"):
        lines = [",".join(ISSUE_FIELDS)]
        for issue in audit["issues"]:
            values = [str(issue.get(field, "")).replace('"', "'") for field in ISSUE_FIELDS]
            lines.append(",".join(f'"{value}"' for value in values))
        write_text_atomic(issue_path, "\n".join(lines) + "\n")
    else:
        write_text_atomic(issue_path, ",".join(ISSUE_FIELDS) + "\n")
    repair_path = audit_dir / "label_repairs.csv"
    repair_fields = ("split", "image_id", "label_rel", "line_no", "raw", "reason")
    lines = [",".join(repair_fields)]
    for repair in audit.get("repairs") or []:
        values = [str(repair.get(field, "")).replace('"', "'") for field in repair_fields]
        lines.append(",".join(f'"{value}"' for value in values))
    write_text_atomic(repair_path, "\n".join(lines) + "\n")
    return {
        "data_manifest": audit_dir / "data_manifest.json",
        "dataset_stats": audit_dir / "dataset_stats.json",
        "issues": issue_path,
        "label_repairs": repair_path,
    }


def load_audit(run: RunContext) -> dict[str, Any]:
    """Rebuild the full audit dict from the saved artefacts.

    The returned mapping has the same top-level keys as :func:`audit_dataset`
    (``splits``, ``images``, ``repairs``, ...) so callers such as
    ``prepare_dataset_view`` work identically with a fresh or a reloaded audit.
    ``manifest``/``stats`` sub-dicts are kept for direct inspection.
    """
    manifest_path = run.audit_dir / "data_manifest.json"
    stats_path = run.audit_dir / "dataset_stats.json"
    if not manifest_path.exists() or not stats_path.exists():
        raise DataError(
            f"run '{run.run_id}' has no saved audit; run the 'check' command first"
        )
    manifest = read_json(manifest_path)
    stats = read_json(stats_path)
    issues: list[dict[str, Any]] = []
    issues_path = run.audit_dir / "issues.csv"
    if issues_path.exists():
        with open(issues_path, "r", encoding="utf-8", newline="") as handle:
            issues = [
                {key: (row.get(key) or "") for key in ISSUE_FIELDS}
                for row in csv.DictReader(handle)
            ]
    failures = [
        f"{issue['split']}/{issue['image_id']}: {issue['detail']}"
        for issue in issues
        if issue.get("severity") == "error"
    ]
    return {
        "schema": stats.get("schema", 1),
        "created_at": manifest.get("created_at"),
        "status": manifest.get("status", "failed"),
        "scope": manifest.get("scope"),
        "data_root": manifest.get("data_root"),
        "data_yaml": manifest.get("data_yaml"),
        "class_names": manifest.get("class_names"),
        "expected_images": manifest.get("expected_images"),
        "split_paths": manifest.get("split_paths"),
        "checks": manifest.get("checks"),
        "duplicates": manifest.get("duplicates") or [],
        "label_repair": manifest.get("label_repair") or stats.get("label_repair"),
        "repairs": manifest.get("repairs") or [],
        "totals": manifest.get("totals"),
        "splits": stats.get("splits") or {},
        "images": manifest.get("images") or [],
        "issues": issues,
        "failures": failures,
        "manifest_digest": manifest.get("manifest_digest"),
        "manifest": manifest,
        "stats": stats,
    }


def dataset_stats(audit: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the data-characteristics block used by figures F01-F03."""
    return {
        "schema": 1,
        "created_at": audit.get("created_at"),
        "status": audit.get("status"),
        "scope": audit.get("scope"),
        "data_root": audit.get("data_root"),
        "class_names": audit.get("class_names"),
        "manifest_digest": audit.get("manifest_digest"),
        "totals": audit.get("totals"),
        "splits": {
            split: {
                key: block[key]
                for key in (
                    "num_images", "num_instances", "num_label_files",
                    "empty_label_images", "missing_label_files", "invalid_label_files",
                    "repaired_label_files", "dropped_boxes",
                    "per_class", "per_class_list", "per_image_objects", "per_image_ids",
                    "objects_per_image", "area_histogram", "area_quantiles", "image_sizes",
                )
                if key in block
            }
            for split, block in (audit.get("splits") or {}).items()
        },
        "label_repair": audit.get("label_repair"),
        "issue_counts": _issue_counts(audit.get("issues") or []),
    }


def _issue_counts(issues: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for issue in issues:
        counts[f"{issue.get('severity', 'info')}:{issue.get('kind', 'unknown')}"] += 1
    return dict(sorted(counts.items()))


# ---------------------------------------------------------------------------
# dataset view
# ---------------------------------------------------------------------------
def _selection(
    split: str,
    entries: Sequence[Mapping[str, Any]],
    limit: int | None,
    seed: int,
) -> list[Mapping[str, Any]]:
    """Deterministic subset selection (sorted ids, seeded sample, re-sorted)."""
    ordered = sorted(entries, key=lambda item: item["image_id"])
    if limit is None or limit >= len(ordered):
        return list(ordered)
    ids = [item["image_id"] for item in ordered]
    rng = random.Random(f"{seed}:{split}")
    picked = set(rng.sample(ids, limit))
    return [item for item in ordered if item["image_id"] in picked]


def view_digest(audit: Mapping[str, Any], selection: Mapping[str, Sequence[Mapping[str, Any]]],
                scope: str) -> str:
    rows = [f"scope={scope}"]
    for split in sorted(selection):
        for entry in selection[split]:
            rows.append(
                f"{split}|{entry['image_id']}|{entry.get('image_sha256') or ''}|"
                f"{entry.get('label_sha256') or ''}"
            )
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def prepare_dataset_view(
    audit: Mapping[str, Any],
    run: RunContext,
    config: Mapping[str, Any] | None = None,
) -> Path:
    """Copy the audited files into a writable view and return its ``data.yaml``.

    ``scope=smoke`` copies only the configured smoke subset; identical content
    digests reuse the existing view after verifying its manifest.
    """
    config = config or run.config
    if audit.get("status") != "ok":
        raise DataError(
            "dataset audit did not pass; refusing to build a training view "
            f"({len(audit.get('failures') or [])} failure(s), see audit/issues.csv)"
        )
    data_cfg = config["data"]
    scope = str(config.get("scope", run.scope))
    seed = int(config["train"]["seed"])
    data_root = Path(audit["data_root"])
    workspace_root = Path(config["project"]["workspace_root"])

    train_split = str(data_cfg["train_split"])
    val_split = str(data_cfg["val_split"])
    limits: dict[str, int | None] = {train_split: None, val_split: None}
    if scope == "smoke":
        limits[train_split] = int(data_cfg["smoke_train_images"])
        limits[val_split] = int(data_cfg["smoke_val_images"])
        # Never silently train on a smoke subset when the operator asked for formal.
        if int(data_cfg["expected_train_images"]) != audit["splits"][train_split]["num_images"]:
            raise DataError("smoke view requires the full audited dataset as input")

    repairs_by_image: dict[tuple[str, str], set[int]] = defaultdict(set)
    for repair in audit.get("repairs") or []:
        repairs_by_image[(str(repair["split"]), str(repair["image_id"]))].add(
            int(repair["line_no"])
        )

    by_split: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for entry in audit["images"]:
        by_split[entry["split"]].append(entry)
    selection = {
        split: _selection(split, by_split[split], limits[split], seed)
        for split in (train_split, val_split)
    }
    digest = view_digest(audit, selection, scope)
    view_root = workspace_root / "data_views" / f"{digest[:16]}_{scope}"
    view_manifest_path = view_root / "view_manifest.json"
    data_yaml_path = view_root / "data.yaml"

    total_bytes = 0
    for split, entries in selection.items():
        for entry in entries:
            total_bytes += (data_root / entry["image_rel"]).stat().st_size
            if entry["label_rel"]:
                total_bytes += (data_root / entry["label_rel"]).stat().st_size

    if view_manifest_path.exists() and data_yaml_path.exists():
        previous = read_json(view_manifest_path)
        if previous.get("digest") == digest and previous.get("scope") == scope:
            run.manifest["dataset_view"] = dict(
                digest=digest, scope=scope, root=str(view_root),
                data_yaml=str(data_yaml_path), reused=True,
            )
            write_json_atomic(run.run_dir / "manifest.json", run.manifest)
            return data_yaml_path

    workspace_root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(workspace_root).free
    required = int(total_bytes * 1.15) + (64 << 20)
    if free < required:
        raise DataError(
            f"not enough free space for the dataset view: need ~{required / 2**20:.0f} MiB, "
            f"have {free / 2**20:.0f} MiB on {workspace_root}"
        )

    view_root.mkdir(parents=True, exist_ok=True)
    copied_bytes = 0
    split_records: dict[str, dict[str, Any]] = {}
    for split, entries in selection.items():
        (view_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (view_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        records = []
        bytes_for_split = 0
        for entry in entries:
            source_image = data_root / entry["image_rel"]
            target_image = view_root / "images" / split / source_image.name
            shutil.copy2(source_image, target_image)
            bytes_for_split += target_image.stat().st_size
            label_name = None
            dropped_lines = repairs_by_image.get((split, entry["image_id"]), set())
            if entry["label_rel"]:
                source_label = data_root / entry["label_rel"]
                target_label = view_root / "labels" / split / source_label.name
                if dropped_lines:
                    # Apply the recorded repair only in the derived view copy.
                    original = source_label.read_text(encoding="utf-8").splitlines()
                    kept = [
                        line for index, line in enumerate(original, start=1)
                        if index not in dropped_lines
                    ]
                    target_label.write_text(
                        "\n".join(kept) + ("\n" if kept else ""), encoding="utf-8"
                    )
                else:
                    shutil.copy2(source_label, target_label)
                bytes_for_split += target_label.stat().st_size
                label_name = source_label.name
            else:
                # Create an explicit empty label so the trainer sees zero objects.
                target_label = view_root / "labels" / split / f"{entry['image_id']}.txt"
                target_label.write_text("", encoding="utf-8")
                label_name = target_label.name
            records.append({
                "image_id": entry["image_id"],
                "image_rel": f"images/{split}/{source_image.name}",
                "label_rel": f"labels/{split}/{label_name}",
                "image_sha256": entry.get("image_sha256"),
                "label_sha256": entry.get("label_sha256"),
                "num_objects": entry.get("num_objects"),
                "width": entry.get("width"),
                "height": entry.get("height"),
                "dropped_label_lines": sorted(dropped_lines),
            })
        copied_bytes += bytes_for_split
        split_records[split] = {"count": len(records), "bytes": bytes_for_split, "entries": records}

    view_manifest = {
        "schema": 1,
        "digest": digest,
        "scope": scope,
        "created_at": now_iso(),
        "source_root": str(data_root),
        "source_manifest_digest": audit.get("manifest_digest"),
        "seed": seed,
        "bytes_copied": copied_bytes,
        "class_names": list(data_cfg["class_names"]),
        "label_repair": dict(audit.get("label_repair") or {"policy": "strict"}),
        "splits": split_records,
    }
    write_json_atomic(view_manifest_path, view_manifest)
    write_text_atomic(
        view_root / ".readonly-source-marker",
        "This directory is a generated copy. The source dataset is never modified.\n",
    )

    data_yaml = {
        "path": str(view_root.resolve()),
        "train": f"images/{train_split}",
        "val": f"images/{val_split}",
        "nc": NUM_CLASSES,
        "names": {index: name for index, name in enumerate(data_cfg["class_names"])},
    }
    write_text_atomic(
        data_yaml_path,
        yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True, default_flow_style=False),
    )
    run.manifest["dataset_view"] = dict(
        digest=digest, scope=scope, root=str(view_root), data_yaml=str(data_yaml_path),
        splits={split: rec["count"] for split, rec in split_records.items()},
        bytes_copied=copied_bytes, reused=False,
    )
    write_json_atomic(run.run_dir / "manifest.json", run.manifest)
    return data_yaml_path


def load_view_manifest(data_yaml_path: str | Path) -> dict[str, Any]:
    path = Path(data_yaml_path)
    manifest_path = path.parent / "view_manifest.json"
    if not manifest_path.exists():
        raise DataError(f"no view_manifest.json next to {path}")
    return read_json(manifest_path)


def view_split_entries(data_yaml_path: str | Path, split: str) -> list[dict[str, Any]]:
    manifest = load_view_manifest(data_yaml_path)
    block = (manifest.get("splits") or {}).get(split) or {}
    return list(block.get("entries") or [])
