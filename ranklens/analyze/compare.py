"""Before/after SERP diff for an algorithm update.

Builds rank maps for both snapshots, classifies every domain's move (entered /
dropped / up / down / same), and splits the picture into a *real-sites* lens
(winners/losers) and a *mega-platform* macro lens (YouTube, Reddit, Wikipedia,
marketplaces…) — those move for reasons unrelated to a small site's on-page work,
so they're aggregated separately rather than mixed into the actionable list.
"""
from __future__ import annotations

from ranklens.models import (
    CompareReport,
    CompareRequest,
    DomainMove,
    MoveStatus,
    Serp,
)

# Giant platforms whose SERP moves reflect Google's site-level treatment, not
# anything a normal site competes with on-page. Reported via the macro lens only.
MEGA_PLATFORMS: set[str] = {
    "youtube.com",
    "reddit.com",
    "facebook.com",
    "instagram.com",
    "quora.com",
    "twitter.com",
    "x.com",
    "tiktok.com",
    "amazon.com",
    "ebay.com",
    "walmart.com",
    "wikipedia.org",
    "wikihow.com",
    "fandom.com",
    "trustpilot.com",
    "scribd.com",
    "discord.com",
    "pinterest.com",
    "linkedin.com",
    "medium.com",
}

# Sort priority for the moves list.
_STATUS_ORDER: dict[MoveStatus, int] = {
    MoveStatus.entered: 0,
    MoveStatus.up: 1,
    MoveStatus.same: 2,
    MoveStatus.down: 3,
    MoveStatus.dropped: 4,
}


def registrable(domain: str) -> str:
    """Crude registrable domain: the last two dot-labels of ``domain``."""
    if not domain:
        return ""
    host = domain.strip().lower().lstrip(".")
    if host.startswith("www."):
        host = host[4:]
    labels = [p for p in host.split(".") if p]
    if len(labels) <= 2:
        return ".".join(labels)
    return ".".join(labels[-2:])


def _rank_map(serp: Serp) -> dict[str, int]:
    """{domain: best rank} for a SERP (first/best occurrence wins)."""
    out: dict[str, int] = {}
    for item in serp.items:
        dom = (item.domain or "").strip().lower()
        if not dom:
            continue
        if dom not in out or item.rank < out[dom]:
            out[dom] = item.rank
    return out


def _authority_for(
    domain: str, authority: dict[str, dict] | None
) -> dict | None:
    """Look up authority by exact domain, then by registrable domain."""
    if not authority:
        return None
    if domain in authority:
        return authority[domain]
    reg = registrable(domain)
    return authority.get(reg)


def build_compare(
    before: Serp,
    after: Serp,
    request: CompareRequest,
    authority: dict[str, dict] | None = None,
) -> CompareReport:
    """Diff ``before`` vs ``after`` SERPs into a :class:`CompareReport`."""
    before_map = _rank_map(before)
    after_map = _rank_map(after)
    domains = set(before_map) | set(after_map)

    moves: list[DomainMove] = []
    entered: list[DomainMove] = []

    for domain in domains:
        b = before_map.get(domain)
        a = after_map.get(domain)

        if b is None and a is not None:
            status = MoveStatus.entered
        elif b is not None and a is None:
            status = MoveStatus.dropped
        elif a < b:
            status = MoveStatus.up
        elif a > b:
            status = MoveStatus.down
        else:
            status = MoveStatus.same

        delta = (b - a) if (b is not None and a is not None) else None
        is_winner = status in (MoveStatus.entered, MoveStatus.up)
        is_mega = registrable(domain) in MEGA_PLATFORMS

        move = DomainMove(
            domain=domain,
            before_rank=b,
            after_rank=a,
            status=status,
            delta=delta,
            is_winner=is_winner,
            is_mega=is_mega,
        )

        auth = _authority_for(domain, authority)
        if auth:
            move.authority_score = auth.get("authority_score")
            move.traffic_visits = auth.get("traffic_visits")
            move.traffic_trend_pct = auth.get("traffic_trend_pct")

        moves.append(move)
        if status is MoveStatus.entered:
            entered.append(move)

    # Real-site lenses (exclude mega platforms).
    winners = [
        m
        for m in moves
        if m.is_winner and not m.is_mega
    ]
    losers = [
        m
        for m in moves
        if m.status in (MoveStatus.dropped, MoveStatus.down) and not m.is_mega
    ]

    # #1 flip.
    n1_before = before.items[0].domain if before.items else None
    n1_after = after.items[0].domain if after.items else None
    n1_flip = bool(n1_before and n1_after and n1_before != n1_after)

    # Churn: share of the after top-N whose domain wasn't in the before set.
    after_n = len(after.items)
    churn_pct = (len(entered) / after_n * 100.0) if after_n else 0.0

    # Macro lens: aggregate over mega platforms only.
    mega_moves = [m for m in moves if m.is_mega]
    mega_entered_up = sum(
        1 for m in mega_moves if m.status in (MoveStatus.entered, MoveStatus.up)
    )
    mega_dropped_down = sum(
        1 for m in mega_moves if m.status in (MoveStatus.dropped, MoveStatus.down)
    )
    macro = {
        "mega_entered_up": mega_entered_up,
        "mega_dropped_down": mega_dropped_down,
        "mega_domains": sorted({m.domain for m in mega_moves}),
        "note": (
            f"{len(mega_moves)} mega-platform domain(s) in play: "
            f"{mega_entered_up} gained, {mega_dropped_down} lost ground. "
            "Tracked separately from real-site winners/losers."
        ),
    }

    # Sort the full moves list: status priority, then delta (bigger gain first).
    def _move_key(m: DomainMove) -> tuple[int, float]:
        # delta None -> 0; descending delta means negate.
        return (_STATUS_ORDER[m.status], -(m.delta if m.delta is not None else 0))

    moves.sort(key=_move_key)

    return CompareReport(
        request=request,
        before=before,
        after=after,
        before_date=before.snapshot_date,
        after_date=after.snapshot_date,
        moves=moves,
        winners=winners,
        losers=losers,
        n1_flip=n1_flip,
        n1_before=n1_before,
        n1_after=n1_after,
        churn_pct=churn_pct,
        macro=macro,
        ai_narrative=None,
    )
