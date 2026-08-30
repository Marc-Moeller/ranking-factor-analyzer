"""Factor-vs-rank correlation — the statistical core (Cora's Overview sheet).

For every factor present in the ranking set we correlate its value against the
ranking position across the fetched pages, using **Spearman** (rank) and
**Pearson** (linear), then take the *Best of Both* (the one with the larger
absolute value, sign preserved). Because a *lower* rank number is *better*, a
negative correlation means "more of the factor → better rank" → the factor's
``direction`` is ``more_is_better``.

A factor is *significant* when ``|best_of_both| >= critical_value(n)``.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import pearsonr, spearmanr

from ranklens.factors_registry import BY_ID
from ranklens.models import FactorCorrelation, PageFactors

# Need at least this many usable (rank, value) pairs to bother correlating.
MIN_POINTS = 5


def critical_value(n: int) -> float:
    """95% two-tailed Pearson critical r for sample size ``n``.

    Approximated as ``1.96 / sqrt(n - 1)`` (clamped to a floor of 0.05).
    Sanity points: n=20 -> ~0.45, n=100 -> ~0.197.
    """
    if n is None or n <= 2:
        return 1.0
    crit = 1.96 / math.sqrt(n - 1)
    return max(crit, 0.05)


def _safe_corr(func, values: np.ndarray, ranks: np.ndarray) -> float | None:
    """Run a scipy correlation, mapping NaN / errors to ``None``.

    Constant input (zero variance on either axis) yields NaN in scipy and a
    runtime warning; we swallow both and return ``None``.
    """
    try:
        with np.errstate(all="ignore"):
            r = func(values, ranks).statistic
    except Exception:
        return None
    if r is None or (isinstance(r, float) and math.isnan(r)):
        return None
    return float(r)


def _best_of_both(spearman: float | None, pearson: float | None) -> float | None:
    """Whichever of the two has the larger absolute value (sign preserved)."""
    candidates = [c for c in (spearman, pearson) if c is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda c: abs(c))


def correlate(
    pages: list[PageFactors],
    target_factors: dict[str, float] | None = None,
) -> list[FactorCorrelation]:
    """Correlate each factor's value against rank across ``pages``.

    For every factor id that appears in at least one page we build the
    (rank, value) series — pages missing the factor default to ``0.0`` *only*
    because at least one page in the set has it. Factors with fewer than
    ``MIN_POINTS`` usable points, or zero variance in either axis, are skipped.
    Results are sorted by ``|best_of_both|`` descending (None last).
    """
    target_factors = target_factors or {}

    # Discover every factor id present anywhere in the ranking set.
    factor_ids: list[str] = []
    seen: set[str] = set()
    for page in pages:
        for fid in page.factors:
            if fid not in seen:
                seen.add(fid)
                factor_ids.append(fid)

    out: list[FactorCorrelation] = []

    for fid in factor_ids:
        # Build aligned (rank, value) arrays ONLY from pages that actually
        # measured this factor. A page we couldn't fetch is *unknown* for HTML
        # factors, not zero — zero-filling it would destroy the correlation.
        ranks_list: list[float] = []
        values_list: list[float] = []
        for page in pages:
            if fid not in page.factors:
                continue
            ranks_list.append(float(page.rank))
            values_list.append(float(page.factors[fid]))

        ranks = np.asarray(ranks_list, dtype=float)
        values = np.asarray(values_list, dtype=float)

        n = len(values)
        nonzero = int(np.count_nonzero(values))

        meta = BY_ID.get(fid)
        name = meta.name if meta else fid
        group = meta.group if meta else ""

        # Skip factors that can't yield a meaningful correlation.
        too_few = n < MIN_POINTS or nonzero < MIN_POINTS
        zero_var = float(np.ptp(values)) == 0.0 or float(np.ptp(ranks)) == 0.0

        if too_few or zero_var:
            # Still emit a record (with stats=None) so the caller sees the factor,
            # but it can never be significant.
            page1_avg, top_max, usage = _summary_stats(values, ranks, n)
            out.append(
                FactorCorrelation(
                    factor_id=fid,
                    name=name,
                    group=group,
                    spearman=None,
                    pearson=None,
                    best_of_both=None,
                    significant=False,
                    page1_avg=page1_avg,
                    top_max=top_max,
                    usage=usage,
                    target_value=target_factors.get(fid),
                    direction=(meta.direction if meta else "more_is_better"),
                )
            )
            continue

        spearman = _safe_corr(spearmanr, values, ranks)
        pearson = _safe_corr(pearsonr, values, ranks)
        bob = _best_of_both(spearman, pearson)

        crit = critical_value(n)
        significant = bob is not None and abs(bob) >= crit

        # Direction from the measured sign: negative corr (more value -> lower/
        # better rank) => more_is_better; positive => less_is_better.
        if bob is None:
            direction = meta.direction if meta else "more_is_better"
        elif bob < 0:
            direction = "more_is_better"
        else:
            direction = "less_is_better"

        page1_avg, top_max, usage = _summary_stats(values, ranks, n)

        out.append(
            FactorCorrelation(
                factor_id=fid,
                name=name,
                group=group,
                spearman=spearman,
                pearson=pearson,
                best_of_both=bob,
                significant=significant,
                page1_avg=page1_avg,
                top_max=top_max,
                usage=usage,
                target_value=target_factors.get(fid),
                direction=direction,
            )
        )

    # Sort by headline strength, None (no correlation) last.
    out.sort(key=lambda c: abs(c.best_of_both) if c.best_of_both is not None else -1.0, reverse=True)
    return out


def _summary_stats(
    values: np.ndarray, ranks: np.ndarray, n: int
) -> tuple[float, float, float]:
    """page1_avg (mean over the top min(10, n) ranked pages), top_max, usage."""
    band = min(10, n)
    # Indices of the `band` best (lowest-numbered) ranks.
    order = np.argsort(ranks, kind="stable")[:band]
    page1_avg = float(np.mean(values[order])) if band else 0.0
    top_max = float(np.max(values)) if n else 0.0
    usage = float(np.count_nonzero(values) / n) if n else 0.0
    return page1_avg, top_max, usage
