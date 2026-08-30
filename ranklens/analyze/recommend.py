"""Roadmap builder — the 10 highest-impact things the tracked page can do.

Mirrors Cora's Roadmap sheet, but tuned for a *small, noisy* SERP sample
(10-25 pages). For each *more-is-better* factor where the tracked page sits
below the Page-1 winners we compute the deficit ("goal"), score the action by
**impact**, and return the top ``ROADMAP_LIMIT`` actions across every group —
on-page, entities, backlinks, and brand alike.

Why this is not a pure significance gate (see the brand-name false-positive incident,
2026-06-08): with only ~20 fetched pages the 95% Pearson critical value is
~0.45, and the enrichment factors that matter most (entities, referring
domains, brand volume) are measured on the fewest/noisiest points — so they
*never* clear the bar and the roadmap collapses to one trivial on-page count
while the page is missing every entity on the SERP. So:

* Significance is a **score booster**, not an entry gate.
* The factor's KNOWN direction (registry) wins over a flipped small-sample
  sign — a confounded ``+0.42`` on Branded Search Volume no longer deletes a
  real brand-demand gap.
* Known-important levers (entities, authority/backlinks, brand, Cora Top-200)
  are eligible on a real deficit even when the per-SERP correlation is weak.
* The result is impact-ranked, capped at 10, and diversified so no single
  group crowds out the rest.
"""
from __future__ import annotations

from ranklens.factors_registry import BY_ID
from ranklens.models import FactorCorrelation, Recommendation, TargetSummary

# How many actions the roadmap shows, and how many any single factor-group may
# occupy (so "add entities" doesn't take all 10 slots when the page has none).
ROADMAP_LIMIT = 10
GROUP_CAP = 3
# A factor with no known importance prior still earns a slot if its measured
# correlation clears this relaxed bar (well below the strict 95% line) — keeps
# random low-prior on-page counts out without a full significance test.
RELAXED_MIN_CORR = 0.15
# Groups we trust as real ranking levers even on a weak per-SERP correlation.
IMPORTANT_GROUPS = {"Entity", "Authority", "Brand"}


def _goal_for(corr: FactorCorrelation, is_bool: bool, is_int: bool) -> float:
    """The target value. Boolean ("Uses/Has") factors goal=1; else page1_avg.

    Integer-valued factors (tag/word/link counts, …) round to a WHOLE number —
    a goal of "1.10 H1 tags" is nonsense and makes a page with 1 H1 look short
    by 0.1, producing a bogus "Add 0 more" action and docking its score.
    """
    if is_bool:
        return 1.0
    avg = corr.page1_avg if corr.page1_avg is not None else 0.0
    if is_int:
        return float(round(avg))
    # Fractional bands (KB, percent): whole above 10, one decimal below.
    if avg >= 10:
        return float(round(avg))
    return float(round(avg, 1))


def _base_weight(group: str, top200: bool) -> float:
    """Importance prior so the big levers aren't buried under trivial counts.

    Backlinks/brand and entities are the factors users *can't* see in Cora's
    on-page sheet and the ones this report exists to surface, so they lead.
    """
    if group in ("Authority", "Brand"):  # off-page demand + link equity
        return 1.6
    if group == "Entity":
        return 1.5
    if top200:  # Cora's strongest-tier on-page factors
        return 1.2
    return 1.0


def _impact(
    *,
    group: str,
    top200: bool,
    difficulty: str,
    correlation: float | None,
    deficit: float,
    goal: float,
    significant: bool,
) -> float:
    """Score one candidate action. Higher = do it sooner.

    impact = importance × difficulty × (evidence) × (size of the gap) × sig
    Every term is bounded so a single dimension can't dominate; the gap term
    is *relative* (deficit / goal) so "you have 0 of something everyone has"
    outranks "you're 7% short on a 1,800-tag page".
    """
    base = _base_weight(group, top200)
    diff_factor = 1.15 if (difficulty or "").lower() == "easy" else 0.9
    corr_factor = 0.4 + min(abs(correlation or 0.0), 1.0)      # 0.4 … 1.4
    deficit_ratio = deficit / max(goal, 1.0)
    deficit_factor = 1.0 + min(deficit_ratio, 1.0)             # 1.0 … 2.0
    sig_factor = 1.15 if significant else 1.0
    return base * diff_factor * corr_factor * deficit_factor * sig_factor


def recommend(
    correlations: list[FactorCorrelation],
    target_factors: dict | None,
) -> tuple[list[Recommendation], TargetSummary | None]:
    """Build the impact-ranked roadmap (+ optional target summary).

    Emits a :class:`Recommendation` for every eligible more-is-better factor
    where the tracked page sits below the Page-1 winners, scores each by
    impact, then returns at most ``ROADMAP_LIMIT`` actions with a per-group
    cap so the list stays diverse. When no target is supplied, every eligible
    factor yields generic "the winners have N" guidance (current=None).

    Returns ``(roadmap, target_summary)``. ``roadmap`` is ordered by impact
    (``priority_score``) descending.
    """
    has_target = target_factors is not None
    tf = target_factors or {}

    candidates: list[Recommendation] = []
    sig_total = 0      # significant more-is-better factors (the opt-score base)
    sig_met = 0        # …of those, how many the page already meets

    for corr in correlations:
        meta = BY_ID.get(corr.factor_id)
        is_bool = bool(meta.is_bool) if meta else False
        unit = meta.unit if meta else "Count"
        # Everything except KB ("Other") and percentages is a whole-number count.
        is_int = unit not in ("Other", "Percent")
        group = meta.group if meta else corr.group
        top200 = bool(meta.top200) if meta else False
        difficulty = meta.difficulty if meta else "Easy"
        category = meta.category if meta else "On Page"
        phase = meta.phase if meta else "Technical"
        measured = corr.best_of_both

        # --- direction gate ---
        if meta is not None:
            # A KNOWN less-is-better / nonlinear factor is never an "add more"
            # action, regardless of how it happened to correlate this run.
            if meta.direction != "more_is_better":
                continue
        else:
            # Unknown factor: fall back to the measured sign.
            if corr.direction != "more_is_better":
                continue

        # Respect a CLEAR opposing signal from the data: a meaningful positive
        # correlation means "the winners have less", so don't tell the user to
        # add more — *unless* this is a trusted-prior lever (entities, backlinks,
        # brand), where a small confounded SERP routinely flips the sign and the
        # registry direction is the better guide. This is what keeps a spurious
        # "Has Email" (registry direction is just the default) off the roadmap
        # while preserving a real Branded-Search-Volume gap.
        opposing = (
            measured is not None
            and measured > 0
            and (corr.significant or measured >= RELAXED_MIN_CORR)
        )
        if opposing and group not in IMPORTANT_GROUPS:
            continue

        goal = _goal_for(corr, is_bool, is_int)

        # Current value of the tracked page (None when no target supplied).
        current = corr.target_value if corr.target_value is not None else tf.get(corr.factor_id)
        if has_target and current is None:
            current = 0.0  # target provided but page lacks this factor -> 0

        # Deficit: how far below goal the page sits.
        if current is not None:
            deficit = max(goal - current, 0.0)
            met = current >= goal
        else:
            deficit = goal
            met = False

        # Track the significance-based optimization score over the factors that
        # demonstrably matter in THIS SERP (kept strict — it's a scorecard, not
        # the roadmap). Direction already vetted above.
        if corr.significant:
            sig_total += 1
            if has_target and met:
                sig_met += 1

        # --- roadmap eligibility (broad on purpose; the cap does the pruning) ---
        if has_target and deficit <= 0:
            continue  # the page already meets/leads this factor
        if not has_target and goal <= 0:
            continue
        # Never surface a no-op like "Add 0 more": a sub-unit deficit rounds away.
        if not is_bool and round(deficit) < 1:
            continue
        # Evidence gate: a known lever (entity/authority/brand/Top-200) earns a
        # slot on its deficit alone; a low-prior factor must show *some*
        # measured correlation so we don't recommend statistical noise.
        important = top200 or group in IMPORTANT_GROUPS
        has_signal = abs(corr.best_of_both or 0.0) >= RELAXED_MIN_CORR or corr.significant
        if not (important or has_signal):
            continue

        # Action text.
        name = meta.name if meta else corr.factor_id
        if is_bool:
            if name.lower().startswith(("uses", "has")):
                action_text = "Add this."
            else:
                action_text = f"Implement {name}."
        else:
            action_text = f"Add {round(deficit)} more. ({unit})"

        priority = _impact(
            group=group,
            top200=top200,
            difficulty=difficulty,
            correlation=corr.best_of_both,
            deficit=deficit,
            goal=goal,
            significant=corr.significant,
        )

        candidates.append(
            Recommendation(
                factor_id=corr.factor_id,
                name=name,
                phase=phase,
                group=group,
                difficulty=difficulty,
                category=category,
                top200=top200,
                current=current,
                goal=goal,
                deficit=deficit,
                action_text=action_text,
                correlation=corr.best_of_both,
                priority_score=priority,
            )
        )

    # Rank by impact, then take the top N with a per-group cap so the roadmap
    # stays diverse (entities + backlinks + brand + on-page all get airtime),
    # backfilling past the cap only if we'd otherwise fall short of the limit.
    candidates.sort(key=lambda r: r.priority_score, reverse=True)
    roadmap: list[Recommendation] = []
    per_group: dict[str, int] = {}
    for rec in candidates:
        if len(roadmap) >= ROADMAP_LIMIT:
            break
        if per_group.get(rec.group, 0) >= GROUP_CAP:
            continue
        roadmap.append(rec)
        per_group[rec.group] = per_group.get(rec.group, 0) + 1
    if len(roadmap) < ROADMAP_LIMIT:
        chosen = set(id(r) for r in roadmap)
        for rec in candidates:
            if len(roadmap) >= ROADMAP_LIMIT:
                break
            if id(rec) not in chosen:
                roadmap.append(rec)
    roadmap.sort(key=lambda r: r.priority_score, reverse=True)

    # Target summary (only when a target was supplied).
    summary: TargetSummary | None = None
    if has_target:
        opt_score = 100.0 * sig_met / max(sig_total, 1)
        quick_wins = sum(
            1 for r in roadmap if r.difficulty == "Easy" and (r.deficit or 0.0) > 0
        )
        url = tf.get("__url__") or ""
        summary = TargetSummary(
            url=str(url),
            found_in_serp=False,
            serp_rank=None,
            optimization_score=opt_score,
            factors_met=sig_met,
            factors_total=sig_total,
            quick_wins=quick_wins,
        )

    return roadmap, summary
