"""Writing brief — turn the finished analysis into writing recommendations.

Answers the question an SEO (or an LLM asked to rewrite the page) actually has:
*"What should this page SAY that it currently doesn't?"* Everything is derived
deterministically from the entity table (consensus-graded gaps only) and the
correlation roadmap — no LLM call, so the brief is free, instant, and can never
recommend something the data doesn't support.

Sections:

* **entities_to_add** — high-consensus entities (the ``must_add`` set) the
  target never mentions.
* **topics_to_expand** — medium-consensus entities (2+ pages, below the
  must-add bar) the target is missing; grouped as "worth covering".
* **facts_to_state** — graded ``(entity, attribute)`` claims the target is
  silent on, with the competitor evidence ("top pages state $61–$94").
* **values_to_review** — the target asserts a value that disagrees with a real
  (2+ page) consensus.
* **relationships_to_make** — graded entity→entity connections the target
  doesn't make.
* **your_edge** — entities and facts unique to the target, which a rewrite
  should preserve and strengthen.
* **information_gain** — open sub-intents no page in the SERP covers well.
* **style_targets** — content-shape goals from the roadmap (word count,
  headings, keyword placement), phrased as writing instructions.

Two renderings ship with the data: ``markdown`` (export-ready, human) and
``llm_prompt`` (a self-contained instruction block to paste into any LLM along
with the page draft). :func:`build_writing_brief` never raises — on any failure
it returns ``None`` and the report simply renders without the brief.
"""
from __future__ import annotations

from ranklens.models import AnalyzeReport, BriefItem, WritingBrief

# Caps per section — a brief nobody reads helps nobody.
MAX_ENTITIES = 12
MAX_TOPICS = 12
MAX_FACTS = 15
MAX_VALUES = 8
MAX_EDGES = 8
MAX_STYLE = 6
MAX_YOUR_EDGE = 8
MAX_INFORMATION_GAIN = 8

# Roadmap phases/groups that translate into writing (not technical) work.
_WRITING_GROUPS = {"Content", "Title & Headings", "Entity"}

# Roadmap factors that are NOT writing instructions even when their group is —
# derived metrics (EAV completeness rises by following the fact sections above),
# byte counts, and duplicates of Word Count.
_STYLE_EXCLUDE = {"EAV_COMPLETENESS", "PAGE_SIZE_KB", "UNABRIDGED_WORD_COUNT"}

_MISSING = ("missing_entity", "missing_attribute", "missing_value")


def build_writing_brief(report: AnalyzeReport) -> WritingBrief | None:
    """Build the writing brief for a finished analyze report. Never raises."""
    try:
        return _build(report)
    except Exception:  # noqa: BLE001 — the brief is derived context, never fatal
        return None


def _fmt_pages(n: int, total: int) -> str:
    return f"{n} of {total} top pages"


def _build(report: AnalyzeReport) -> WritingBrief | None:
    et = report.entity_table
    if et is None:
        return None

    keyword = report.request.keyword
    target_url = (report.request.target_url or "").strip()
    has_target = bool(et.has_target and target_url)
    n_brands = max(0, len([r for r in et.ranks if r != 0]))

    brief = WritingBrief(keyword=keyword, target_url=target_url)

    # ---- your edge (target-only entities and uncommon asserted facts) ----- #
    if has_target and et.target_read_ok:
        for row in et.coverage_rows:
            if len(brief.your_edge) >= MAX_YOUR_EDGE:
                break
            if row.role == "subject" or not row.target_present or row.consensus > 0:
                continue
            etype = f" ({row.entity_type})" if row.entity_type else ""
            brief.your_edge.append(BriefItem(
                text=f"Keep and amplify “{row.entity}”{etype} — only your page covers it.",
                evidence="target-only entity coverage",
                entity=row.entity,
            ))

        for row in et.eav_rows:
            if len(brief.your_edge) >= MAX_YOUR_EDGE:
                break
            if (
                row.role == "subject" or not row.graded or not row.target_present
                or not row.target_value or row.gap != ""
                or row.consensus * 2 >= n_brands
            ):
                continue
            etype = f" ({row.entity_type})" if row.entity_type else ""
            brief.your_edge.append(BriefItem(
                text=(
                    f"Keep and amplify your {row.attribute} detail for "
                    f"“{row.entity}”{etype}: “{row.target_value}”."
                ),
                evidence=_fmt_pages(row.consensus, n_brands) + " state this value",
                entity=row.entity,
                attribute=row.attribute,
            ))

    # ---- information gain (sub-intents no page covers well) --------------- #
    if report.semantic is not None:
        for gap in report.semantic.open_gaps[:MAX_INFORMATION_GAIN]:
            brief.information_gain.append(BriefItem(
                text=f"Build substantive coverage of “{gap}”.",
                evidence="no top page covers this well — first-mover opportunity",
            ))

    # ---- entities to add (must-add = high consensus, entirely absent) ------ #
    cov_by_entity = {row.entity: row for row in et.coverage_rows}
    for name in et.must_add_entities[:MAX_ENTITIES]:
        row = cov_by_entity.get(name)
        cons = row.consensus if row else 0
        etype = f" ({row.entity_type})" if row and row.entity_type else ""
        brief.entities_to_add.append(BriefItem(
            text=f"Mention and discuss “{name}”{etype} — your page never does.",
            evidence=_fmt_pages(cons, n_brands) + " cover it",
            entity=name,
        ))

    # ---- topics to expand (2+ page consensus, absent, below must-add) ------ #
    must = set(et.must_add_entities)
    if has_target:
        for row in et.coverage_rows:
            if len(brief.topics_to_expand) >= MAX_TOPICS:
                break
            if row.entity in must or row.target_present:
                continue
            if row.gap != "missing_entity" or row.consensus < 2:
                continue
            etype = f" ({row.entity_type})" if row.entity_type else ""
            brief.topics_to_expand.append(BriefItem(
                text=f"Consider covering “{row.entity}”{etype}.",
                evidence=_fmt_pages(row.consensus, n_brands) + " mention it",
                entity=row.entity,
            ))

    # ---- facts to state (graded EAV gaps) ---------------------------------- #
    if has_target:
        for row in et.eav_rows:
            if len(brief.facts_to_state) >= MAX_FACTS:
                break
            if not row.graded or row.gap not in _MISSING:
                continue
            if row.role == "subject":
                subject = "your business/service"
                text = f"State your own {row.attribute} on the page."
            else:
                subject = f"“{row.entity}”"
                text = f"State the {row.attribute} of {subject}."
            ev = _fmt_pages(row.consensus, n_brands) + " state it"
            # "each page states its own" is a display sentinel, not an example value.
            if row.consensus_value and not row.consensus_value.startswith("each page"):
                ev += f" — e.g. {row.consensus_value}"
            brief.facts_to_state.append(BriefItem(
                text=text, evidence=ev, entity=row.entity, attribute=row.attribute,
            ))

        # ---- values to review (off a real consensus) ----------------------- #
        for row in et.eav_rows:
            if len(brief.values_to_review) >= MAX_VALUES:
                break
            if not row.graded or row.gap != "off_consensus":
                continue
            brief.values_to_review.append(BriefItem(
                text=(
                    f"Re-check your {row.attribute} for “{row.entity}”: you say "
                    f"“{row.target_value}”, the SERP consensus is “{row.consensus_value}”."
                ),
                evidence=_fmt_pages(row.consensus, n_brands) + " assert this attribute",
                entity=row.entity, attribute=row.attribute,
            ))

        # ---- relationships to make (graded edge gaps) ----------------------- #
        for row in et.edge_rows:
            if len(brief.relationships_to_make) >= MAX_EDGES:
                break
            if not row.graded or row.gap not in _MISSING:
                continue
            if row.role == "subject":
                # First-party edge: the consensus target is some OTHER business's
                # owner/location — never quote it as what THIS page should say.
                text = (
                    f"Make an explicit “{row.attribute}” statement about your own "
                    f"business in the copy (use your real details)."
                )
            else:
                text = (
                    f"Make the connection “{row.entity} → {row.attribute} → "
                    f"{row.consensus_value}” explicit in your copy."
                )
            brief.relationships_to_make.append(BriefItem(
                text=text,
                evidence=_fmt_pages(row.consensus, n_brands) + " make it",
                entity=row.entity, attribute=row.attribute,
            ))

    # ---- style targets from the roadmap (content-shape goals) -------------- #
    if has_target:
        # When the target's fact extraction failed, its entity factors are all
        # zero — entity-count "targets" derived from them would be noise.
        target_read_ok = getattr(et, "target_read_ok", True)
        recs = sorted(report.recommendations, key=lambda r: r.priority_score, reverse=True)
        for r in recs:
            if len(brief.style_targets) >= MAX_STYLE:
                break
            if r.group not in _WRITING_GROUPS and r.phase not in _WRITING_GROUPS:
                continue
            if r.factor_id in _STYLE_EXCLUDE:
                continue
            if not target_read_ok and (r.group == "Entity" or r.phase == "Entities"):
                continue
            cur = _num(r.current)
            goal = _num(r.goal)
            detail = f" (now {cur}, aim for {goal})" if cur is not None and goal is not None else ""
            brief.style_targets.append(BriefItem(
                text=f"{r.name}: {r.action_text}".strip().rstrip(".") + detail + ".",
                evidence=f"{r.phase} · {r.difficulty}",
            ))

    total = (
        len(brief.entities_to_add) + len(brief.topics_to_expand)
        + len(brief.facts_to_state) + len(brief.values_to_review)
        + len(brief.relationships_to_make) + len(brief.style_targets)
        + len(brief.your_edge) + len(brief.information_gain)
    )
    if total == 0:
        return None

    if has_target:
        brief.summary = (
            f"{total} writing recommendations for {target_url} on “{keyword}”, "
            f"including gaps to close, unique points to preserve, and information-gain "
            f"opportunities across the top {n_brands} ranking pages."
        )
    else:
        brief.summary = (
            f"A {total}-point content blueprint for “{keyword}”, combining what the "
            f"top {n_brands} ranking pages cover with information-gain opportunities."
        )

    brief.markdown = _render_markdown(brief, has_target)
    brief.llm_prompt = _render_llm_prompt(brief, has_target)
    return brief


def _num(v: float | None) -> str | None:
    if v is None:
        return None
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.1f}"


def _section(lines: list[str], title: str, items: list[BriefItem]) -> None:
    if not items:
        return
    lines.append(f"## {title}")
    lines.append("")
    for it in items:
        ev = f" _({it.evidence})_" if it.evidence else ""
        lines.append(f"- {it.text}{ev}")
    lines.append("")


def _render_markdown(brief: WritingBrief, has_target: bool) -> str:
    lines: list[str] = [
        f"# Writing brief — “{brief.keyword}”",
        "",
        brief.summary,
        "",
    ]
    _section(lines, "Entities to add (the top pages cover these, you don't)", brief.entities_to_add)
    _section(lines, "Topics worth covering", brief.topics_to_expand)
    _section(lines, "Facts to state", brief.facts_to_state)
    _section(lines, "Values to double-check", brief.values_to_review)
    _section(lines, "Connections to make explicit", brief.relationships_to_make)
    _section(lines, "What only you say — keep it", brief.your_edge)
    _section(lines, "Information gain: what nobody covers", brief.information_gain)
    _section(lines, "Content shape targets", brief.style_targets)
    lines.append("---")
    lines.append(
        "_Grounded in cross-page consensus, target-unique claims, and semantic open gaps. "
        "Verify facts about your own business before publishing — never invent them._"
    )
    return "\n".join(lines)


def _render_llm_prompt(brief: WritingBrief, has_target: bool) -> str:
    """A self-contained instruction block the user pastes into any LLM."""
    out: list[str] = []
    out.append(
        "You are an expert SEO content editor. "
        + (
            f"Improve the page {brief.target_url} for the search query "
            f"\"{brief.keyword}\". I will paste the current page copy below."
            if has_target
            else f"Draft a page that competes for the search query \"{brief.keyword}\"."
        )
    )
    out.append("")
    out.append(
        "Rewrite/extend the copy so it satisfies EVERY instruction that applies, "
        "while keeping the existing voice and structure where possible:"
    )
    out.append("")

    def _block(title: str, items: list[BriefItem]) -> None:
        if not items:
            return
        out.append(f"{title}:")
        for it in items:
            ev = f" [{it.evidence}]" if it.evidence else ""
            out.append(f"- {it.text}{ev}")
        out.append("")

    _block("ADD these entities (work each into a substantive sentence or section, not a keyword list)", brief.entities_to_add)
    _block("COVER these topics if relevant to the business", brief.topics_to_expand)
    _block("STATE these concrete facts (use the REAL values for this business — ask me if unknown)", brief.facts_to_state)
    _block("DOUBLE-CHECK these values (the page may be outdated or off-consensus)", brief.values_to_review)
    _block("MAKE these relationships explicit", brief.relationships_to_make)
    _block("PRESERVE these unique points (do not delete them in the rewrite)", brief.your_edge)
    _block("ADD content no competitor covers", brief.information_gain)
    _block("MATCH these content-shape targets", brief.style_targets)

    out.append("Rules:")
    out.append("- NEVER invent facts about the business (prices, addresses, credentials). Ask for any real value you need.")
    out.append("- Integrate naturally — no keyword stuffing, no bolted-on FAQ spam.")
    out.append("- Keep claims consistent with the rest of the page.")
    out.append("- Return the improved copy plus a short list of what you changed and why.")
    out.append("")
    out.append("PAGE COPY:")
    out.append("<paste your current page text here>")
    return "\n".join(out)
