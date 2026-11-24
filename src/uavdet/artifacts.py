"""Experiment directories, stage bookkeeping and atomic artefact I/O.

Layout (see 设计规范 §5)::

    outputs/<run_id>/
      manifest.json
      config_resolved.yaml
      environment.json
      audit/            data_manifest.json, dataset_stats.json, issues.csv
      train/            args.yaml, results.csv, weights/{best,last}.pt, train.log
      evaluation/       metrics.json, per_class.csv, curves.npz, ...
      benchmark/        image_list.json, timings.csv, summary.json
      figures/<tag>/    *.png/*.pdf/*.svg, source_data/, figure_manifest.json
      logs/

A stage is only ever ``complete`` after its transition ran; a finished run with
``skipped`` stages is not the same as a complete one.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from .types import (
    STAGE_NAMES,
    STAGE_STATUS_COMPLETE,
    STAGE_STATUS_FAILED,
    STAGE_STATUS_INTERRUPTED,
    STAGE_STATUS_PENDING,
    STAGE_STATUS_RUNNING,
    STAGE_STATUS_SKIPPED,
    RunContext,
    RunError,
)

__all__ = [
    "open_run",
    "read_manifest",
    "list_runs",
    "begin_stage",
    "complete_stage",
    "fail_stage",
    "interrupt_stage",
    "skip_stage",
    "record_event",
    "save_resolved_config",
    "record_environment",
    "run_summary",
    "stage_products_ok",
    "missing_stage_products",
    "write_json_atomic",
    "write_text_atomic",
    "append_jsonl",
    "read_json",
    "read_jsonl",
    "sha256_file",
    "sha256_bytes",
    "sha256_text",
    "sha256_json",
    "file_sha_records",
    "now_iso",
    "utc_stamp",
]

MANIFEST_SCHEMA = 1
_MAX_EVENTS = 500


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------
def now_iso() -> str:
    """Local ISO-8601 timestamp with timezone (human readable logs)."""
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def utc_stamp() -> str:
    """Filename safe UTC stamp."""
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(payload: Any) -> str:
    return sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha_records(paths: Iterable[str | Path], base: Path | None = None) -> dict[str, str]:
    """Map ``relative-or-absolute path -> sha256`` for the files that exist."""
    records: dict[str, str] = {}
    for raw in paths:
        path = Path(raw)
        if not path.is_file():
            continue
        key = str(path)
        if base is not None:
            try:
                key = path.resolve().relative_to(base.resolve()).as_posix()
            except ValueError:
                key = str(path)
        records[key] = sha256_file(path)
    return records


def write_json_atomic(path: str | Path, payload: Any) -> Path:
    """Write JSON via a temp file + ``os.replace`` (readers never see half a file).

    ``allow_nan=False`` means NaN/inf raise instead of producing invalid JSON.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    tmp.write_text(text + "\n", encoding="utf-8")
    os.replace(tmp, target)
    return target


def write_text_atomic(path: str | Path, text: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, target)
    return target


def read_json(path: str | Path, default: Any = None) -> Any:
    target = Path(path)
    if not target.exists():
        if default is not None:
            return default
        raise FileNotFoundError(str(target))
    with open(target, "r", encoding="utf-8") as handle:
        return json.load(handle)


def append_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    """Append JSON lines (one object per line) to ``path``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n")
    return target


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(target, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# run lifecycle
# ---------------------------------------------------------------------------
def _initial_manifest(config: Mapping[str, Any], run_id: str, scope: str,
                      parent_run_id: str | None) -> dict[str, Any]:
    from .config import config_digest, protocol_digest

    return {
        "schema": MANIFEST_SCHEMA,
        "run_id": run_id,
        "scope": scope,
        "parent_run_id": parent_run_id,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "config_digest": config_digest(config),
        "protocol_digest": protocol_digest(config),
        "project_name": config["project"]["name"],
        "data_root": config["data"].get("root"),
        "stages": {
            stage: {
                "status": STAGE_STATUS_PENDING,
                "attempts": 0,
                "started_at": None,
                "finished_at": None,
                "duration_s": None,
                "summary": {},
                "error": None,
            }
            for stage in STAGE_NAMES
        },
        "events": [],
    }


def _save_manifest(run: RunContext) -> None:
    run.manifest["updated_at"] = now_iso()
    write_json_atomic(run.manifest_path, run.manifest)


def record_event(run: RunContext, kind: str, message: str, **data: Any) -> dict[str, Any]:
    """Append an audit event to the manifest and to ``logs/events.jsonl``."""
    event = {"at": now_iso(), "kind": str(kind), "message": str(message)}
    if data:
        event["data"] = {k: _jsonable(v) for k, v in data.items()}
    events = run.manifest.setdefault("events", [])
    events.append(event)
    if len(events) > _MAX_EVENTS:
        del events[: len(events) - _MAX_EVENTS]
    run.logs_dir.mkdir(parents=True, exist_ok=True)
    append_jsonl(run.logs_dir / "events.jsonl", [event])
    _save_manifest(run)
    return event


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def open_run(
    config: Mapping[str, Any],
    run_id: str,
    parent_run_id: str | None = None,
) -> RunContext:
    """Open (or create) ``outputs/<run_id>`` and return its context.

    Re-opening an existing run:

    * refuses when the *training protocol* fingerprint changed (a new run is
      required for batch/lr/data changes),
    * records a resume event when only paths/device/workers moved,
    * never deletes existing artefacts.
    """
    from .config import config_digest, protocol_digest, validate_run_id

    validate_run_id(run_id)
    output_root = Path(config["project"]["output_root"])
    run_dir = output_root / run_id
    scope = str(config.get("scope", "formal"))

    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get("protocol_digest") != protocol_digest(config):
            raise RunError(
                f"run '{run_id}' exists with a different training protocol "
                f"(batch/epochs/lr/data/class order changed). Create a new --run-id "
                f"instead of overwriting a finished experiment."
            )
        run = RunContext(run_id=run_id, run_dir=run_dir, config=dict(config),
                         scope=str(manifest.get("scope", scope)), manifest=manifest)
        changed = manifest.get("config_digest") != config_digest(config)
        if changed:
            # Paths/device/workers moved; keep the old resolved config for audit.
            old_digest = str(manifest.get("config_digest", ""))[:8]
            backup = run_dir / f"config_resolved.{old_digest or 'old'}.yaml"
            if run.resolved_config_path.exists() and not backup.exists():
                shutil.copy2(run.resolved_config_path, backup)
            save_resolved_config(run, config)
            record_event(
                run,
                "config_rebase",
                "resolved config changed (path/device/workers move); protocol unchanged",
                old_digest=manifest.get("config_digest"),
                new_digest=config_digest(config),
            )
        run.manifest["config_digest"] = config_digest(config)
        run.ensure_dirs()
        _save_manifest(run)
        return run

    if run_dir.exists() and any(run_dir.iterdir()):
        # A directory without a manifest is not a run we created: never adopt it.
        raise RunError(
            f"{run_dir} already exists but has no manifest.json; refusing to adopt an "
            f"unknown directory. Use another --run-id or move it away yourself."
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = _initial_manifest(config, run_id, scope, parent_run_id)
    run = RunContext(run_id=run_id, run_dir=run_dir, config=dict(config),
                     scope=scope, manifest=manifest)
    run.ensure_dirs()
    save_resolved_config(run, config)
    record_event(run, "created", f"run '{run_id}' created with scope={scope}",
                 parent_run_id=parent_run_id)
    _save_manifest(run)
    return run


def read_manifest(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / "manifest.json"
    if not path.exists():
        raise RunError(f"no manifest.json in {run_dir}")
    return read_json(path)


def list_runs(output_root: str | Path) -> list[dict[str, Any]]:
    """Summarise every run directory below ``output_root`` (newest first)."""
    root = Path(output_root)
    if not root.exists():
        return []
    summaries: list[dict[str, Any]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        manifest_path = child / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = read_json(manifest_path)
        except (json.JSONDecodeError, OSError):
            continue
        summaries.append(
            {
                "run_id": manifest.get("run_id", child.name),
                "run_dir": str(child),
                "scope": manifest.get("scope", "?"),
                "created_at": manifest.get("created_at"),
                "updated_at": manifest.get("updated_at"),
                "stages": {
                    name: (block or {}).get("status", STAGE_STATUS_PENDING)
                    for name, block in (manifest.get("stages") or {}).items()
                },
            }
        )
    summaries.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return summaries


def save_resolved_config(run: RunContext, config: Mapping[str, Any] | None = None) -> Path:
    from .config import write_resolved_config

    return write_resolved_config(config or run.config, run.resolved_config_path)


# ---------------------------------------------------------------------------
# stage transitions
# ---------------------------------------------------------------------------
def _require_stage_name(stage: str) -> str:
    if stage not in STAGE_NAMES:
        raise RunError(f"unknown stage {stage!r}; expected one of {list(STAGE_NAMES)}")
    return stage


def begin_stage(run: RunContext, stage: str, allow_repeat: bool = False) -> dict[str, Any]:
    """Mark ``stage`` as running.

    Refuses to restart a ``complete`` stage unless ``allow_repeat`` is set (the
    operator explicitly asked to re-run it, e.g. a new figure tag). A stale
    ``running`` state from a crashed process is converted to ``interrupted``
    first so the history stays honest.
    """
    _require_stage_name(stage)
    block = run.manifest.setdefault("stages", {}).setdefault(stage, {})
    status = block.get("status", STAGE_STATUS_PENDING)

    if status == STAGE_STATUS_COMPLETE and not allow_repeat:
        raise RunError(
            f"stage '{stage}' of run '{run.run_id}' is already complete; "
            f"refusing to overwrite it (use a new --run-id, or the explicit "
            f"re-run flag if the command supports it)"
        )
    if status == STAGE_STATUS_RUNNING:
        block["status"] = STAGE_STATUS_INTERRUPTED
        block["error"] = "previous process ended while the stage was running"
        record_event(run, "stage_interrupted", f"stage '{stage}' found stale 'running' state")

    block["status"] = STAGE_STATUS_RUNNING
    block["attempts"] = int(block.get("attempts") or 0) + 1
    block["started_at"] = now_iso()
    block["finished_at"] = None
    block["duration_s"] = None
    block["error"] = None
    block["summary"] = {}
    _save_manifest(run)
    record_event(run, "stage_start", f"stage '{stage}' started", attempt=block["attempts"])
    return dict(block)


def complete_stage(run: RunContext, stage: str, **summary: Any) -> dict[str, Any]:
    """Mark ``stage`` complete with a JSON-safe summary."""
    _require_stage_name(stage)
    block = run.manifest.setdefault("stages", {}).setdefault(stage, {})
    if block.get("status") not in (STAGE_STATUS_RUNNING, STAGE_STATUS_PENDING):
        raise RunError(
            f"cannot complete stage '{stage}' from status {block.get('status')!r}"
        )
    started = block.get("started_at")
    block["status"] = STAGE_STATUS_COMPLETE
    block["finished_at"] = now_iso()
    block["duration_s"] = _elapsed(started)
    block["error"] = None
    block["summary"] = {k: _jsonable(v) for k, v in summary.items()}
    _save_manifest(run)
    record_event(run, "stage_complete", f"stage '{stage}' complete",
                 duration_s=block["duration_s"])
    return dict(block)


def fail_stage(run: RunContext, stage: str, error: str) -> dict[str, Any]:
    """Mark ``stage`` failed (a failed stage must never read as complete)."""
    _require_stage_name(stage)
    block = run.manifest.setdefault("stages", {}).setdefault(stage, {})
    started = block.get("started_at")
    block["status"] = STAGE_STATUS_FAILED
    block["finished_at"] = now_iso()
    block["duration_s"] = _elapsed(started)
    block["error"] = str(error)
    _save_manifest(run)
    record_event(run, "stage_failed", f"stage '{stage}' failed: {error}")
    return dict(block)


def interrupt_stage(run: RunContext, stage: str, reason: str) -> dict[str, Any]:
    """Mark ``stage`` interrupted (SIGINT / user stop / machine reboot)."""
    _require_stage_name(stage)
    block = run.manifest.setdefault("stages", {}).setdefault(stage, {})
    started = block.get("started_at")
    block["status"] = STAGE_STATUS_INTERRUPTED
    block["finished_at"] = now_iso()
    block["duration_s"] = _elapsed(started)
    block["error"] = str(reason)
    _save_manifest(run)
    record_event(run, "stage_interrupted", f"stage '{stage}' interrupted: {reason}")
    return dict(block)


def skip_stage(run: RunContext, stage: str, reason: str) -> dict[str, Any]:
    """Mark ``stage`` skipped (explicitly not part of this run)."""
    _require_stage_name(stage)
    block = run.manifest.setdefault("stages", {}).setdefault(stage, {})
    block["status"] = STAGE_STATUS_SKIPPED
    block["finished_at"] = now_iso()
    block["error"] = None
    block["summary"] = {"reason": str(reason)}
    _save_manifest(run)
    record_event(run, "stage_skipped", f"stage '{stage}' skipped: {reason}")
    return dict(block)


def _elapsed(started_at: str | None) -> float | None:
    if not started_at:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S%z",):
        try:
            start = time.mktime(time.strptime(started_at, fmt))
            return round(max(0.0, time.time() - start), 3)
        except (ValueError, OverflowError):
            continue
    return None


# ---------------------------------------------------------------------------
# environment snapshot
# ---------------------------------------------------------------------------
def _module_version(name: str) -> str | None:
    try:
        module = __import__(name)
    except Exception:
        return None
    return str(getattr(module, "__version__", "unknown"))


def record_environment(run: RunContext, device: str | None = None) -> dict[str, Any]:
    """Collect real Python/package/device facts and persist them in the run."""
    info: dict[str, Any] = {
        "recorded_at": now_iso(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "packages": {},
        "requested_device": device,
        "torch": None,
        "gpus": [],
    }
    for name in ("torch", "torchvision", "ultralytics", "cv2", "numpy", "pandas",
                 "matplotlib", "PIL", "yaml", "PyQt6"):
        version = _module_version(name)
        if version is not None:
            info["packages"][name] = version
    try:
        import torch

        info["torch"] = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(torch.version, "cuda", None),
            "mps_available": bool(getattr(torch.backends, "mps", None)
                                  and torch.backends.mps.is_available()),
            "threads": torch.get_num_threads(),
        }
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                info["gpus"].append(
                    {
                        "index": index,
                        "name": props.name,
                        "total_memory_gb": round(props.total_memory / 1024 ** 3, 2),
                    }
                )
    except Exception as exc:  # torch missing is allowed for plot-only machines
        info["torch"] = {"error": f"{type(exc).__name__}: {exc}"}
    write_json_atomic(run.environment_path, info)
    return info


# ---------------------------------------------------------------------------
# summaries / pipeline helpers
# ---------------------------------------------------------------------------
_STAGE_PRODUCTS: dict[str, tuple[str, ...]] = {
    "audit": ("audit/data_manifest.json", "audit/dataset_stats.json"),
    "train": ("train/results.csv", "train/weights/best.pt"),
    "evaluation": ("evaluation/metrics.json", "evaluation/per_class.csv"),
    "benchmark": ("benchmark/summary.json",),
}


def missing_stage_products(run: RunContext, stage: str,
                           figure_tag: str | None = None) -> list[str]:
    """Which key products of ``stage`` are absent (empty list when all exist)."""
    if stage == "figures":
        tag = figure_tag or "v1"
        rel = f"figures/{tag}/figure_manifest.json"
        return [] if (run.run_dir / rel).exists() else [rel]
    return [rel for rel in _STAGE_PRODUCTS.get(stage, ())
            if not (run.run_dir / rel).exists()]


def stage_products_ok(run: RunContext, stage: str, figure_tag: str | None = None) -> bool:
    """True when the stage is complete *and* its key products exist on disk."""
    if run.stage_status(stage) != STAGE_STATUS_COMPLETE:
        return False
    return not missing_stage_products(run, stage, figure_tag)


def run_summary(run: RunContext) -> dict[str, Any]:
    """Compact, JSON-safe description of a run for the GUI / reports."""
    stages = {
        name: {
            "status": run.stage_status(name),
            "attempts": int(run.stage_block(name).get("attempts") or 0),
            "finished_at": run.stage_block(name).get("finished_at"),
            "error": run.stage_block(name).get("error"),
            "summary": run.stage_block(name).get("summary") or {},
        }
        for name in STAGE_NAMES
    }
    metrics_path = run.evaluation_dir / "metrics.json"
    metrics = read_json(metrics_path, default={}) if metrics_path.exists() else {}
    figure_dirs = sorted(
        p.name for p in (run.run_dir / "figures").glob("*") if p.is_dir()
    ) if (run.run_dir / "figures").exists() else []
    return {
        "run_id": run.run_id,
        "run_dir": str(run.run_dir),
        "scope": run.scope,
        "created_at": run.manifest.get("created_at"),
        "updated_at": run.manifest.get("updated_at"),
        "stages": stages,
        "metrics": {
            key: metrics.get(key)
            for key in ("map50", "map50_95", "precision", "recall", "num_images",
                        "num_instances", "split", "status")
            if metrics.get(key) is not None
        },
        "figure_tags": figure_dirs,
        "is_formal_complete": bool(
            run.scope == "formal"
            and all(run.stage_status(s) == STAGE_STATUS_COMPLETE
                    for s in ("audit", "train", "evaluation", "benchmark", "figures"))
        ),
    }


def tools_summary() -> dict[str, Any]:
    """Small helper listing available optional tooling (used by ``--help`` extras)."""
    return {
        "torch": _module_version("torch"),
        "ultralytics": _module_version("ultralytics"),
        "cv2": _module_version("cv2"),
        "PyQt6": _module_version("PyQt6"),
    }
