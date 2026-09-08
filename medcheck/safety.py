from __future__ import annotations

from .constants import HIGH_RISK_PATTERNS


def requires_forced_abstention(text: str) -> bool:
    normalized = " ".join((text or "").lower().split())
    return any(pattern.lower() in normalized for pattern in HIGH_RISK_PATTERNS)


def filter_evidence_ids(
    evidence_ids: list[str],
    valid_ids: set[str],
) -> list[str]:
    return [item for item in evidence_ids if item in valid_ids]
