"""Strict YAML configuration loading with path resolution and overrides.

Design rules (see 设计规范 §3):

* ``project.root`` is resolved relative to *the config file*, every other
  relative path is resolved against the resolved ``project.root``. The current
  working directory never influences a path.
* Precedence: command line override > config file > code default.
* Unknown or misspelled keys are hard errors, never silently ignored.
* The fully resolved config is what gets written into a run directory, so a
  later machine can rebuild exactly the same paths.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from .types import CLASS_NAMES, ConfigError, NUM_CLASSES

__all__ = [
    "load_config",
    "load_config_data",
    "validate_run_id",
    "config_digest",
    "protocol_digest",
    "training_protocol",
    "write_resolved_config",
    "parse_override",
    "parse_overrides",
    "apply_overrides",
    "default_config_path",
    "DEFAULTS",
    "CONFIG_SECTIONS",
]

CONFIG_SECTIONS: tuple[str, ...] = (
    "project",
    "data",
    "train",
    "eval",
    "predict",
    "benchmark",
    "figures",
)

# Metadata block written into the resolved config (read-only, reserved).
RESERVED_TOP_LEVEL: tuple[str, ...] = ("meta",)

RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

_DEVICE_PATTERN = re.compile(r"^(cpu|mps|(cuda:?)?\d+(,\d+)*|cuda)$", re.IGNORECASE)


class _Spec:
    """One schema entry: python type, default and optional extra validator."""

    __slots__ = ("kind", "default", "check", "choices")

    def __init__(self, kind: str, default: Any, check=None, choices: Sequence[Any] | None = None):
        self.kind = kind
        self.default = default
        self.check = check
        self.choices = tuple(choices) if choices else None


def _between(low: float, high: float | None = None, *, allow_low: bool = True):
    def check(value: float) -> str | None:
        if not allow_low and value <= low:
            return f"must be > {low}, got {value}"
        if allow_low and value < low:
            return f"must be >= {low}, got {value}"
        if high is not None and value > high:
            return f"must be <= {high}, got {value}"
        return None

    return check


def _positive(value):
    return None if value >= 1 else f"must be >= 1, got {value}"


def _non_negative(value):
    return None if value >= 0 else f"must be >= 0, got {value}"


def _img_size(value):
    if value <= 0:
        return f"must be positive, got {value}"
    if value % 32 != 0:
        return f"must be a multiple of 32 (YOLO stride), got {value}"
    return None


def _device(value: str):
    if not value or not _DEVICE_PATTERN.match(str(value)):
        return f"not a device string (expected cpu|mps|0|0,1|cuda:0), got {value!r}"
    return None


_SCHEMA: dict[str, dict[str, _Spec]] = {
    "project": {
        "name": _Spec("str", "uavdet-visdrone-yolo11s"),
        "root": _Spec("str", ".."),
        "output_root": _Spec("str", "outputs"),
        "workspace_root": _Spec("str", "workspace"),
        "app_data_root": _Spec("str", "app_data"),
    },
    "data": {
        "root": _Spec("str", "dataset_visdrone"),
        "train_split": _Spec("str", "train"),
        "val_split": _Spec("str", "val"),
        "expected_train_images": _Spec("int", 6471, _positive),
        "expected_val_images": _Spec("int", 548, _positive),
        "verify_image_decode": _Spec("bool", True),
        "view_mode": _Spec("str", "copy", choices=("copy",)),
        "label_repair": _Spec("str", "strict",
                              choices=("strict", "drop_degenerate_boxes")),
        "class_names": _Spec("names", list(CLASS_NAMES)),
        "smoke_train_images": _Spec("int", 20, _positive),
        "smoke_val_images": _Spec("int", 10, _positive),
    },
    "train": {
        "init_weights": _Spec("str", "weights/yolo11s.pt"),
        "allow_trained_init": _Spec("bool", False),
        "epochs": _Spec("int", 300, _positive),
        "imgsz": _Spec("int", 640, _img_size),
        "batch": _Spec("int", 16, _positive),
        "workers": _Spec("int", 4, _non_negative),
        "device": _Spec("str", "cpu", _device),
        "optimizer": _Spec("str", "SGD", choices=("SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp", "auto")),
        "lr0": _Spec("float", 0.01, _between(0.0, None, allow_low=False)),
        "lrf": _Spec("float", 0.01, _between(0.0, 1.0)),
        "momentum": _Spec("float", 0.937, _between(0.0, 1.0)),
        "weight_decay": _Spec("float", 0.0005, _between(0.0)),
        "warmup_epochs": _Spec("float", 3.0, _between(0.0)),
        "cos_lr": _Spec("bool", False),
        "seed": _Spec("int", 0, _non_negative),
        "deterministic": _Spec("bool", True),
        "amp": _Spec("bool", True),
        "mosaic": _Spec("float", 1.0, _between(0.0, 1.0)),
        "close_mosaic": _Spec("int", 10, _non_negative),
        "hsv_h": _Spec("float", 0.015, _between(0.0, 1.0)),
        "hsv_s": _Spec("float", 0.7, _between(0.0, 1.0)),
        "hsv_v": _Spec("float", 0.4, _between(0.0, 1.0)),
        "translate": _Spec("float", 0.1, _between(0.0, 1.0)),
        "scale": _Spec("float", 0.5, _between(0.0, 1.0)),
        "fliplr": _Spec("float", 0.5, _between(0.0, 1.0)),
        "flipud": _Spec("float", 0.0, _between(0.0, 1.0)),
        "mixup": _Spec("float", 0.0, _between(0.0, 1.0)),
        "copy_paste": _Spec("float", 0.0, _between(0.0, 1.0)),
        "multi_scale": _Spec("bool", False),
        "patience": _Spec("int", 0, _non_negative),
        "plots": _Spec("bool", True),
        "save_period": _Spec("int", -1),
    },
    "eval": {
        "split": _Spec("str", "val", choices=("val", "test")),
        "imgsz": _Spec("int", 640, _img_size),
        "batch": _Spec("int", 1, _positive),
        "conf": _Spec("float", 0.001, _between(0.0, 1.0)),
        "iou": _Spec("float", 0.70, _between(0.0, 1.0, allow_low=False)),
        "max_det": _Spec("int", 300, _positive),
        "device": _Spec("str", "cpu", _device),
        "plots": _Spec("bool", True),
        "case_conf": _Spec("float", 0.25, _between(0.0, 1.0)),
        "case_iou": _Spec("float", 0.50, _between(0.0, 1.0, allow_low=False)),
        "max_cases": _Spec("int", 6, _positive),
        # Explicit image ids the operator wants in the case panels (plan 02 §2:
        # "支持用户显式指定 image_id"). Empty = purely automatic selection.
        "case_images": _Spec("str_list", []),
    },
    "predict": {
        "imgsz": _Spec("int", 640, _img_size),
        "conf": _Spec("float", 0.25, _between(0.0, 1.0)),
        "iou": _Spec("float", 0.70, _between(0.0, 1.0, allow_low=False)),
        "max_det": _Spec("int", 300, _positive),
        "device": _Spec("str", "cpu", _device),
        "half": _Spec("bool", False),
    },
    "benchmark": {
        "num_images": _Spec("int", 100, _positive),
        "warmup": _Spec("int", 5, _non_negative),
        "repeats": _Spec("int", 3, _positive),
        "batch": _Spec("int", 1, _positive),
        "half": _Spec("bool", False),
        "seed": _Spec("int", 0, _non_negative),
        "imgsz": _Spec("int", 640, _img_size),
        "conf": _Spec("float", 0.25, _between(0.0, 1.0)),
        "iou": _Spec("float", 0.70, _between(0.0, 1.0, allow_low=False)),
        "max_det": _Spec("int", 300, _positive),
        "device": _Spec("str", "cpu", _device),
    },
    "figures": {
        "dpi": _Spec("int", 300, _positive),
        "formats": _Spec("formats", ["png", "pdf", "svg"]),
        "figure_width": _Spec("float", 7.0, _between(2.0)),
        "font_family": _Spec("str", "auto"),
        "tag": _Spec("str", "v1"),
    },
}

#: read-only convenience: code default config (never mutated).
DEFAULTS: dict[str, Any] = {
    section: {key: copy.deepcopy(spec.default) for key, spec in keys.items()}
    for section, keys in _SCHEMA.items()
}
DEFAULTS["scope"] = "formal"

_SCOPE_VALUES = ("formal", "smoke")
_FORMATS_ALLOWED = ("png", "pdf", "svg")

# Keys excluded from the *training protocol* fingerprint: they may legitimately
# change when a run is resumed on a different machine.
_PROTOCOL_EXCLUDED_TRAIN = ("device", "workers", "init_weights", "plots", "save_period")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def validate_run_id(run_id: str) -> str:
    """Accept ``[A-Za-z0-9_-]{1,64}`` only; reject separators and ``..``."""
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.match(run_id):
        raise ConfigError(
            f"invalid run_id {run_id!r}: only letters, digits, '_' and '-' are allowed "
            "(1-64 characters, no path separators, no '..')"
        )
    return run_id


def _resolve_path(base: Path, value: str | os.PathLike[str]) -> Path:
    candidate = Path(os.path.expanduser(str(value)))
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def _normalise_value(spec: _Spec, key: str, value: Any, where: str) -> Any:
    kind = spec.kind
    if kind == "str":
        if not isinstance(value, str):
            raise ConfigError(f"{where}.{key} must be a string, got {type(value).__name__}")
        return value
    if kind == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}.{key} must be an integer, got {value!r}")
        return int(value)
    if kind == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where}.{key} must be a number, got {value!r}")
        return float(value)
    if kind == "bool":
        if not isinstance(value, bool):
            raise ConfigError(f"{where}.{key} must be true/false, got {value!r}")
        return bool(value)
    if kind == "names":
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{where}.{key} must be a list of class names")
        names = [str(v) for v in value]
        if len(names) != NUM_CLASSES:
            raise ConfigError(f"{where}.{key} must have exactly {NUM_CLASSES} entries, got {len(names)}")
        if tuple(names) != CLASS_NAMES:
            raise ConfigError(
                f"{where}.{key} must keep the VisDrone class order {list(CLASS_NAMES)}; "
                f"got {names}"
            )
        return names
    if kind == "str_list":
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{where}.{key} must be a list of strings")
        return [str(v) for v in value]
    if kind == "formats":
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{where}.{key} must be a list of formats")
        formats = [str(v).lower() for v in value]
        unknown = [f for f in formats if f not in _FORMATS_ALLOWED]
        if unknown:
            raise ConfigError(f"{where}.{key} supports {list(_FORMATS_ALLOWED)}, got {unknown}")
        if not formats:
            raise ConfigError(f"{where}.{key} must not be empty")
        return formats
    raise ConfigError(f"internal: unsupported schema kind {kind!r}")


def _validate_section_block(section: str, block: Any) -> dict[str, Any]:
    if not isinstance(block, Mapping):
        raise ConfigError(f"config section '{section}' must be a mapping")
    spec_map = _SCHEMA[section]
    unknown = sorted(set(block) - set(spec_map))
    if unknown:
        raise ConfigError(
            f"unknown key(s) in section '{section}': {unknown}. "
            f"Allowed keys: {sorted(spec_map)}"
        )
    result: dict[str, Any] = {}
    for key, spec in spec_map.items():
        if key not in block:
            continue
        value = _normalise_value(spec, key, block[key], section)
        if spec.choices is not None and value not in spec.choices:
            raise ConfigError(f"{section}.{key} must be one of {list(spec.choices)}, got {value!r}")
        if spec.check is not None:
            problem = spec.check(value)
            if problem:
                raise ConfigError(f"{section}.{key} {problem}")
        result[key] = value
    return result


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover - message path
        raise ConfigError(f"cannot parse YAML {path}: {exc}") from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"config file {path} must contain a YAML mapping")
    return dict(raw)


def _merge_extends(path: Path, seen: tuple[Path, ...] = ()) -> tuple[dict[str, Any], list[str]]:
    """Load a config honouring ``extends:`` (recursively) and return chain."""
    path = path.resolve()
    if path in seen:
        chain = " -> ".join(str(p) for p in (*seen, path))
        raise ConfigError(f"circular 'extends' chain: {chain}")
    raw = _read_yaml(path)
    extends = raw.pop("extends", None)
    if extends is None:
        return raw, [str(path)]
    if not isinstance(extends, str) or not extends.strip():
        raise ConfigError(f"{path}: 'extends' must be a non-empty string path")
    base_path = (path.parent / extends).resolve()
    base_raw, chain = _merge_extends(base_path, (*seen, path))
    merged = _deep_merge(base_raw, raw)
    return merged, [*chain, str(path)]


def _validate_scope(scope: Any) -> str:
    if scope not in _SCOPE_VALUES:
        raise ConfigError(f"scope must be one of {list(_SCOPE_VALUES)}, got {scope!r}")
    return str(scope)


def _finalise(raw: Mapping[str, Any], config_path: Path, chain: list[str]) -> dict[str, Any]:
    unknown_top = sorted(set(raw) - set(_SCHEMA) - {"scope"} - set(RESERVED_TOP_LEVEL))
    if unknown_top:
        raise ConfigError(
            f"unknown top-level key(s): {unknown_top}. Allowed: "
            f"{sorted(set(_SCHEMA) | {'scope'})}"
        )
    if RESERVED_TOP_LEVEL and any(k in raw for k in RESERVED_TOP_LEVEL):
        raise ConfigError("the 'meta' block is reserved and cannot be set in a config file")

    config: dict[str, Any] = {}
    scope = _validate_scope(raw.get("scope", DEFAULTS["scope"]))
    config["scope"] = scope
    for section in CONFIG_SECTIONS:
        block = dict(DEFAULTS[section])
        block.update(_validate_section_block(section, raw.get(section) or {}))
        config[section] = block

    # --- path resolution ---------------------------------------------------
    base_dir = config_path.parent
    project_root = _resolve_path(base_dir, config["project"]["root"])
    config["project"]["root"] = str(project_root)
    for key in ("output_root", "workspace_root", "app_data_root"):
        config["project"][key] = str(_resolve_path(project_root, config["project"][key]))
    config["data"]["root"] = str(_resolve_path(project_root, config["data"]["root"]))
    config["train"]["init_weights"] = str(_resolve_path(project_root, config["train"]["init_weights"]))

    # --- smoke/formal sanity ----------------------------------------------
    if scope == "smoke":
        if config["train"]["epochs"] > 20 or config["train"]["batch"] > 8:
            raise ConfigError(
                "scope=smoke must stay a short run (epochs<=20, batch<=8); "
                "change scope to formal for the real protocol"
            )
    config["meta"] = {
        "config_path": str(config_path.resolve()),
        "config_chain": chain,
        "loaded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "schema": 1,
    }
    _cross_validate(config)
    return config


def _cross_validate(config: Mapping[str, Any]) -> None:
    """Checks that involve more than one section."""
    if config["train"]["close_mosaic"] > config["train"]["epochs"]:
        raise ConfigError(
            "train.close_mosaic cannot exceed train.epochs "
            f"({config['train']['close_mosaic']} > {config['train']['epochs']})"
        )
    if config["train"]["batch"] < 1:
        raise ConfigError("train.batch must be >= 1")


def load_config_data(path: str | Path) -> dict[str, Any]:
    """Read (with ``extends``) and validate a config without overrides."""
    config_path = Path(path)
    raw, chain = _merge_extends(config_path)
    return _finalise(raw, config_path, chain)


def apply_overrides(config: dict[str, Any], overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    """Apply dotted-key overrides on top of a validated config (in place copy)."""
    if not overrides:
        return config
    config = copy.deepcopy(config)
    for dotted, value in overrides.items():
        if value is None:
            continue
        text = str(dotted)
        if text == "scope":
            config["scope"] = _validate_scope(value)
            continue
        parts = text.split(".")
        if len(parts) != 2:
            raise ConfigError(f"override key must be 'section.key' (or 'scope'), got {dotted!r}")
        section, key = parts
        if section not in _SCHEMA or key not in _SCHEMA[section]:
            raise ConfigError(f"unknown override {dotted!r}")
        spec = _SCHEMA[section][key]
        config[section][key] = _normalise_value(spec, key, value, section)
        if spec.choices is not None and config[section][key] not in spec.choices:
            raise ConfigError(f"{section}.{key} must be one of {list(spec.choices)}")
        if spec.check is not None:
            problem = spec.check(config[section][key])
            if problem:
                raise ConfigError(f"{section}.{key} {problem}")

    # Re-resolve the paths that overrides may have touched.
    base_dir = Path(config["meta"]["config_path"]).parent
    project_root = _resolve_path(base_dir, config["project"]["root"])
    config["project"]["root"] = str(project_root)
    for key in ("output_root", "workspace_root", "app_data_root"):
        config["project"][key] = str(_resolve_path(project_root, config["project"][key]))
    config["data"]["root"] = str(_resolve_path(project_root, config["data"]["root"]))
    config["train"]["init_weights"] = str(_resolve_path(project_root, config["train"]["init_weights"]))
    _cross_validate(config)
    return config


def load_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Load, validate, resolve and override a YAML config.

    Parameters
    ----------
    path:
        YAML file. Relative values inside it resolve against this file.
    overrides:
        Mapping of ``"section.key" -> value`` (already typed) that wins over the
        file. ``None`` values are ignored so argparse defaults never clobber.
    """
    config = load_config_data(path)
    return apply_overrides(config, overrides)


# ---------------------------------------------------------------------------
# digests / protocol fingerprint
# ---------------------------------------------------------------------------
def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _without_meta(config: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in config.items() if k != "meta"}


def config_digest(config: Mapping[str, Any]) -> str:
    """SHA256 over the resolved config ignoring the load timestamp."""
    return hashlib.sha256(_canonical_json(_without_meta(config)).encode("utf-8")).hexdigest()


def training_protocol(config: Mapping[str, Any]) -> dict[str, Any]:
    """Protocol subset that must not change when resuming the same run."""
    train = {
        key: value
        for key, value in config["train"].items()
        if key not in _PROTOCOL_EXCLUDED_TRAIN
    }
    return {
        "scope": config["scope"],
        "train": train,
        "data": {
            "expected_train_images": config["data"]["expected_train_images"],
            "expected_val_images": config["data"]["expected_val_images"],
            "class_names": list(config["data"]["class_names"]),
            "train_split": config["data"]["train_split"],
            "val_split": config["data"]["val_split"],
        },
    }


def protocol_digest(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(training_protocol(config)).encode("utf-8")).hexdigest()


def write_resolved_config(config: Mapping[str, Any], path: str | Path) -> Path:
    """Write the resolved config atomically as YAML."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(
        yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )
    os.replace(tmp, target)
    return target


# ---------------------------------------------------------------------------
# CLI style overrides: "train.epochs=5", "data.conf=0.3"
# ---------------------------------------------------------------------------
def parse_override(text: str) -> tuple[str, Any]:
    """Parse ``section.key=value`` (value parsed as YAML scalar/container)."""
    if "=" not in text:
        raise ConfigError(f"override must look like section.key=value, got {text!r}")
    key, _, raw_value = text.partition("=")
    key = key.strip()
    if "." not in key:
        raise ConfigError(f"override key must be 'section.key', got {key!r}")
    try:
        value = yaml.safe_load(raw_value)
    except yaml.YAMLError as exc:
        raise ConfigError(f"cannot parse override value in {text!r}: {exc}") from exc
    return key, value


def parse_overrides(items: Iterable[str] | None) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in items or ():
        key, value = parse_override(item)
        overrides[key] = value
    return overrides


def default_config_path(start: str | Path | None = None) -> Path:
    """Locate ``configs/project.yaml`` from the cwd or the package location."""
    here = Path(start) if start is not None else Path.cwd()
    for base in (here, here.parent, Path(__file__).resolve()):
        candidate = base / "configs" / "project.yaml"
        if candidate.exists():
            return candidate
    # Walk up from the current directory a few levels (repo layout).
    current = here.resolve()
    for _ in range(6):
        candidate = current / "configs" / "project.yaml"
        if candidate.exists():
            return candidate
        if current.parent == current:
            break
        current = current.parent
    raise ConfigError(
        "cannot locate configs/project.yaml; pass --config <path> explicitly"
    )
