"""The analyze layer — correlation, roadmap, and before/after SERP diff.

Turns extracted per-page factors into a Cora-style correlation table plus a
prioritised roadmap, and diffs two SERP snapshots around an algorithm update.
"""
from __future__ import annotations

from ranklens.analyze.compare import build_compare
from ranklens.analyze.correlate import correlate, critical_value
from ranklens.analyze.recommend import recommend

__all__ = ["correlate", "critical_value", "recommend", "build_compare"]
