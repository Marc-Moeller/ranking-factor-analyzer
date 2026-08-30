"""RankLens report layer: AI narratives + a polished self-contained HTML report.

Public surface:
- :func:`narrate_analyze` / :func:`narrate_compare` — async AI markdown narratives.
- :func:`render_analyze` / :func:`render_compare` — report models -> HTML string.
- :func:`save_report` — write a rendered HTML string to disk (UTF-8).
- :func:`md_to_html` — the tiny Markdown -> HTML helper used by the templates.
"""
from __future__ import annotations

from .ai import narrate_analyze, narrate_compare
from .html import render_analyze, render_compare, save_report
from .markdown import md_to_html

__all__ = [
    "narrate_analyze",
    "narrate_compare",
    "render_analyze",
    "render_compare",
    "save_report",
    "md_to_html",
]
