"""Device resolution with an explicit, recorded CPU fallback.

A requested device is never silently replaced: if ``device=0`` was asked for and
no CUDA GPU exists, this module returns the effective device *and* a warning that
callers write into the run log / environment record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["DeviceChoice", "resolve_device", "torch_available"]


@dataclass
class DeviceChoice:
    requested: str
    effective: str
    warnings: list[str] = field(default_factory=list)
    kind: str = "cpu"  # cpu | cuda | mps


def torch_available() -> bool:
    """True when ``torch`` can be imported (needed for any real inference)."""
    try:
        import torch  # noqa: F401  (availability probe)
    except Exception:
        return False
    return True


def resolve_device(requested: str | int | None) -> DeviceChoice:
    """Map a config device string to what this machine can actually run.

    ``"cpu"`` is always available. ``"mps"``/``"cuda"``/``"0"`` fall back to CPU
    with a warning when the backend is missing, so the caller (not this helper)
    decides whether that is acceptable for a formal run.
    """
    requested_text = "cpu" if requested is None else str(requested)
    choice = DeviceChoice(requested=requested_text, effective="cpu")

    try:
        import torch
    except Exception as exc:
        choice.warnings.append(f"torch is not importable ({exc}); using CPU without a model")
        return choice

    text = requested_text.strip().lower()
    if text in ("cpu", ""):
        choice.kind = "cpu"
        return choice

    wants_mps = text == "mps"
    wants_cuda = text == "cuda" or text.startswith("cuda:") or text[0].isdigit()

    if wants_mps:
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            choice.effective = "mps"
            choice.kind = "mps"
            return choice
        choice.warnings.append(
            "device 'mps' requested but the MPS backend is unavailable; falling back to CPU"
        )
        return choice

    if wants_cuda:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            choice.effective = requested_text if text.startswith("cuda") else str(requested_text)
            choice.kind = "cuda"
            return choice
        choice.warnings.append(
            "device '%s' requested but no CUDA GPU is available; falling back to CPU"
            % requested_text
        )
        return choice

    choice.warnings.append(f"unrecognised device {requested_text!r}; using CPU")
    return choice


def device_summary(choice: DeviceChoice) -> dict[str, Any]:
    return {
        "requested": choice.requested,
        "effective": choice.effective,
        "kind": choice.kind,
        "warnings": list(choice.warnings),
    }
