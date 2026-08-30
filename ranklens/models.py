"""The contract. Every layer (engine, API, future frontend) speaks these models.

Design notes:
- `PageFactors.factors` is a flat ``{factor_id: value}`` dict so the extraction
  layer can evolve the factor set without churning the schema. Factor *metadata*
  (name, group, phase, direction, top200) lives in `ranklens.factors_registry`.
- Everything is JSON-serializable (`model_dump(mode="json")`) so a `Run` can be
  persisted to SQLite now and Postgres later with no code change.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# Credential / provider fields that may arrive on AnalyzeRequest as BYOK
# overrides. Never persist these on a Run — blank them before store.save_run.
BYOK_REQUEST_FIELDS: tuple[str, ...] = (
    "llm_api_url",
    "llm_api_key",
    "llm_model",
    "dataforseo_login",
    "dataforseo_password",
    "crux_api_key",
)


def blank_byok_fields(payload: dict[str, Any] | None) -> None:
    """Set every BYOK credential field on a request dict to None in place."""
    if not isinstance(payload, dict):
        return
    for name in BYOK_REQUEST_FIELDS:
        if name in payload:
            payload[name] = None


def blank_request_byok(request: "AnalyzeRequest") -> None:
    """Drop BYOK secrets from an in-memory AnalyzeRequest after settings copy."""
    for name in BYOK_REQUEST_FIELDS:
        setattr(request, name, None)


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #
class AnalyzeRequest(BaseModel):
    """Live Cora-style analysis: one keyword, an optional target URL to grade."""
    keyword: str
    target_url: Optional[str] = None
    country: str = "us"               # Google gl code
    language: str = "en"
    max_pages: Optional[int] = None   # None -> settings default
    include_authority: bool = False   # pull domain authority/traffic from the configured authority API
    include_backlinks: bool = False   # pull page/domain backlink power + quality
    include_brand: bool = False       # pull branded search volume as a brand-demand factor
    include_entities: bool = True     # LLM entity/EAV discovery + the top-10 topical EAV comparison table (on by default)
    include_topical: bool = True      # sitemap + Google site: topical-authority read on the target domain (on by default; needs target_url)
    include_funnel: bool = True       # ranking-funnel panels (semantic/intent/quality/engagement/CrUX); --no-funnel to skip
    serp_source: str = "auto"         # auto | serpmaster | dataforseo
    notes: Optional[str] = None
    # Per-run BYOK overrides. None = use process Settings / env. These MUST be
    # blanked before a Run is persisted (see ranklens.store.save_run).
    llm_api_url: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_model: Optional[str] = None
    dataforseo_login: Optional[str] = None
    dataforseo_password: Optional[str] = None
    crux_api_key: Optional[str] = None


class CompareRequest(BaseModel):
    """Before/after an algorithm update for one keyword."""
    keyword: str
    update_date: str                  # ISO date, rollout start (the 'before' cutoff)
    update_name: Optional[str] = None # e.g. "May 2026 Core Update"
    country: str = "us"
    language: str = "en"
    depth: int = 20
    include_authority: bool = True


# --------------------------------------------------------------------------- #
# SERP
# --------------------------------------------------------------------------- #
class SerpItem(BaseModel):
    rank: int
    url: str
    domain: str                       # registrable host, www-stripped
    title: str = ""
    snippet: str = ""
    displayed_url: str = ""


class Serp(BaseModel):
    keyword: str
    country: str
    language: str = "en"
    source: str                       # "serp-api" | "dataforseo-historical" | "dataforseo-live"
    captured_at: datetime = Field(default_factory=_now)
    snapshot_date: Optional[str] = None   # for historical: the dated snapshot used
    items: list[SerpItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Page factors
# --------------------------------------------------------------------------- #
class PageFactors(BaseModel):
    url: str
    rank: int
    domain: str
    fetched_ok: bool = False
    status_code: Optional[int] = None
    load_ms: Optional[float] = None
    error: Optional[str] = None
    # factor_id -> numeric value (bool stored as 0/1). See factors_registry.
    factors: dict[str, float] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Analysis output
# --------------------------------------------------------------------------- #
class FactorCorrelation(BaseModel):
    factor_id: str
    name: str
    group: str
    spearman: Optional[float] = None
    pearson: Optional[float] = None
    best_of_both: Optional[float] = None   # signed; the headline strength
    significant: bool = False              # |best_of_both| > critical value
    page1_avg: Optional[float] = None      # mean over top-N (the "good" target band)
    top_max: Optional[float] = None        # max observed
    usage: Optional[float] = None          # fraction of pages with a non-zero value
    target_value: Optional[float] = None   # the tracked URL's value (if any)
    goal: Optional[float] = None           # recommended target
    deficit: Optional[float] = None        # goal - target_value (signed)
    direction: str = "more_is_better"      # derived from correlation sign


class Recommendation(BaseModel):
    factor_id: str
    name: str
    phase: str                             # "Title & Headings", "Content", ...
    group: str
    difficulty: str = "Easy"               # Easy | Difficult
    category: str = "On Page"
    top200: bool = False
    current: Optional[float] = None
    goal: Optional[float] = None
    deficit: Optional[float] = None
    action_text: str = ""                  # "Add 12 more.", "Add JSON-LD markup.", ...
    correlation: Optional[float] = None
    priority_score: float = 0.0            # ranks the roadmap


class TargetSummary(BaseModel):
    """How the tracked URL stacks up against the SERP."""
    url: str
    found_in_serp: bool = False
    serp_rank: Optional[int] = None
    optimization_score: float = 0.0        # 0-100, share of significant factors at/above goal
    factors_met: int = 0
    factors_total: int = 0
    quick_wins: int = 0                    # Easy + significant + deficit>0


# --------------------------------------------------------------------------- #
# Off-page (backlink) layer — provider-sourced context, not on-page factors
# --------------------------------------------------------------------------- #
class BacklinkStats(BaseModel):
    """Backlink counters for one target (a page, a homepage, or a domain)."""
    scope: str = "url"                     # url | homepage | domain
    target: str = ""                       # the exact target queried
    authority_score: Optional[float] = None
    referring_domains: Optional[int] = None
    total_backlinks: Optional[int] = None
    follow: Optional[int] = None
    nofollow: Optional[int] = None
    dofollow_ratio: Optional[float] = None  # follow / (follow + nofollow)


class Backlink(BaseModel):
    """One inbound link to a ranking page (from /v1/backlinks)."""
    source_url: str
    source_domain: str = ""
    anchor: str = ""
    dofollow: bool = True
    source_authority: Optional[float] = None   # source page ascore
    domain_authority: Optional[float] = None   # source domain ascore
    first_seen: Optional[str] = None
    topical_relevance: Optional[float] = None  # 0-1, LLM-judged vs our topic
    to_domain: str = ""                         # ranking site this link points at
    to_rank: Optional[int] = None               # that site's SERP rank


class LinkQuality(BaseModel):
    """A composite quality+power read on the target's backlink profile."""
    score: Optional[float] = None              # 0-100 composite
    sample_size: int = 0
    avg_source_authority: Optional[float] = None
    mean_topical_relevance: Optional[float] = None   # 0-1
    dofollow_ratio: Optional[float] = None
    summary: str = ""


class OffPagePanel(BaseModel):
    """Per-target off-page context for the report (backlink power + quality)."""
    target_page_stats: Optional[BacklinkStats] = None
    target_homepage_stats: Optional[BacklinkStats] = None
    target_domain_stats: Optional[BacklinkStats] = None
    page1_avg_page_authority: Optional[float] = None
    page1_avg_ref_domains: Optional[float] = None
    target_backlinks: list[Backlink] = Field(default_factory=list)
    # Inbound links across the top-N ranking sites (incl. the target if it ranks),
    # each tagged with ``to_domain`` / ``to_rank`` so the report can filter by site.
    competitor_backlinks: list[Backlink] = Field(default_factory=list)
    link_quality: Optional[LinkQuality] = None
    cost_usd: float = 0.0


class BrandKeyword(BaseModel):
    """One branded search phrase + its monthly search volume."""
    phrase: str
    volume: float = 0.0


class BrandCompetitor(BaseModel):
    """A domain's brand-demand profile, for the competitor comparison table."""
    domain: str
    brand_term: str = ""                        # humanized brand label
    total_volume: float = 0.0                   # summed branded search volume
    keyword_count: int = 0                      # number of distinct branded variations
    rank: Optional[int] = None                  # best SERP position of this domain (None = not in SERP)
    top_keywords: list["BrandKeyword"] = Field(default_factory=list)  # biggest variations
    is_target: bool = False                     # True for the tracked target's own domain


class BrandPanel(BaseModel):
    """Brand-demand context: branded search volume of the ranking domains.

    The brand signal is the summed monthly search volume of the target domain's
    BRANDED ranked keywords (phrases containing the brand-name tokens), sourced
    from ``/v1/ranked-keywords``.
    """
    brand_term: str = ""                       # humanized brand term for the target
    target_brand_volume: Optional[float] = None  # summed branded search volume
    page1_avg_brand_volume: Optional[float] = None  # mean over top-N domains
    domain_brand_volume: dict[str, float] = Field(default_factory=dict)  # domain -> volume
    target_brand_keywords: list["BrandKeyword"] = Field(default_factory=list)  # every target brand variation
    competitors: list["BrandCompetitor"] = Field(default_factory=list)  # leaderboard incl. target + top-5 rivals
    brand_rank: Optional[int] = None           # target's 1-based position in the brand leaderboard
    summary: str = ""
    cost_usd: float = 0.0


# --------------------------------------------------------------------------- #
# Entity / EAV layer — LLM-discovered topical entities + the top-10 EAV table
# --------------------------------------------------------------------------- #
class EntityMention(BaseModel):
    """One real-world entity a page is about (LLM- or schema-discovered)."""
    name: str                                  # surface form as the page uses it
    type: str = ""                             # Product | Brand | Org | Person | Place | Concept | ...
    salience: float = 0.0                      # 0-1 importance to the page
    canonical_id: str = ""                     # normalized key used to align across pages
    source: str = "llm"                        # llm | schema


class EavTriple(BaseModel):
    """An attribute->value claim (or entity->entity edge) a page asserts."""
    entity: str                                # subject entity (surface form)
    attribute: str                             # normalized attribute / relation name
    value: str                                 # asserted value (or the target entity, for an edge)
    canonical_entity: str = ""                 # canonical_id of the subject entity
    is_edge: bool = False                      # True => value is another entity (a connection)
    source: str = "llm"                        # llm | schema


class PageEntities(BaseModel):
    """The entity/EAV read for one ranking page (or the tracked target)."""
    url: str
    rank: int = 0
    domain: str = ""
    entities: list[EntityMention] = Field(default_factory=list)
    triples: list[EavTriple] = Field(default_factory=list)  # includes edges (is_edge=True)
    # The page's visible body text, kept so the entity-table builder can verify an
    # entity's PRESENCE against the page's actual words — instead of trusting the
    # page's isolated LLM call to have recalled it. Transient/internal: excluded
    # from serialization so it never bloats the stored run JSON.
    body_text: str = Field(default="", exclude=True)


class EntityCell(BaseModel):
    """One brand's cell in an EAV row: did this page assert it, and with what value."""
    rank: int                                  # 0 = the tracked target column
    present: bool = False
    value: str = ""                            # asserted value (View B) — blank for View A
    salience: float = 0.0                      # entity salience (View A)


class EavRow(BaseModel):
    """A row of either matrix.

    View A (entity coverage): ``attribute == ""`` — the row is one entity.
    View B (attribute/value): ``attribute`` set — the row is one (entity, attribute)
    pair, and ``is_edge`` marks an entity->entity connection.
    """
    entity: str
    entity_type: str = ""
    attribute: str = ""                        # "" => View-A entity row
    is_edge: bool = False
    # "subject" => the shared first-party row (each page's own business/service)
    role: str = ""
    cells: dict[int, EntityCell] = Field(default_factory=dict)  # rank -> cell
    consensus: int = 0                         # how many top-N pages cover this row
    # consensus >= MIN_GAP_CONSENSUS: the row is real cross-page consensus and the
    # target is graded against it; ungraded rows are page-specific market intel.
    graded: bool = False
    consensus_value: str = ""                  # modal value across covering pages (View B)
    target_present: bool = False
    target_value: str = ""
    # "" (target covers it) | missing_entity | missing_attribute | missing_value | off_consensus
    gap: str = ""


class EntityTable(BaseModel):
    """The top-N topical EAV comparison: two matrices + the gap summary."""
    keyword: str = ""
    ranks: list[int] = Field(default_factory=list)        # brand columns, in SERP order
    rank_domains: dict[int, str] = Field(default_factory=dict)  # rank -> domain label
    has_target: bool = False
    coverage_rows: list[EavRow] = Field(default_factory=list)   # View A
    eav_rows: list[EavRow] = Field(default_factory=list)        # View B (attribute->value)
    edge_rows: list[EavRow] = Field(default_factory=list)       # View C (entity->relation->entity)
    must_add_entities: list[str] = Field(default_factory=list)  # consensus>=threshold & target absent
    target_completeness: Optional[float] = None   # covered graded pairs / graded pairs, 0-1
    # False when the target page had body text but its entity extraction came
    # back empty (LLM flap) — View B/C gaps are then not graded (View A stays
    # valid: presence there is text-verified).
    target_read_ok: bool = True
    n_entities: int = 0
    n_pairs: int = 0          # attribute->value claims (edges excluded)
    n_edges: int = 0          # entity->entity relationships
    n_graded_pairs: int = 0   # pairs with cross-page consensus (>=2 pages)
    n_graded_edges: int = 0   # edges with cross-page consensus (>=2 pages)
    summary: str = ""
    cost_usd: float = 0.0


# --------------------------------------------------------------------------- #
# Topical authority layer — does the DOMAIN have a supporting content cluster
# for a page's topic? Sitemap inventory + Google site: relevance, joined.
# --------------------------------------------------------------------------- #
class TopicalPage(BaseModel):
    """One page on the domain judged part (or not) of the topic cluster."""
    url: str
    in_sitemap: bool = False            # present in the domain's own sitemap
    slug_match: bool = False            # URL slug matched a core/variation term
    serp_hit: bool = False              # Google returned it for the site: query
    title: str = ""                     # from the site: result, when available
    is_adjacent: bool = False           # matched a supporting subtopic, not the core
    matched_terms: list[str] = Field(default_factory=list)


class TopicalAuthorityReport(BaseModel):
    """Verdict on whether a domain has topical authority for a given topic.

    Joins two evidence sources: the domain's sitemap (what the site *claims* to
    cover) and a Google ``site:`` query with LLM-expanded variations (what Google
    *associates* with the domain). The cluster is the union, scored for breadth,
    focus and indexation.
    """
    domain: str
    topic: str                          # the core topic analyzed
    country: str = "us"
    target_url: Optional[str] = None    # the specific page we're supporting (if any)

    # LLM-expanded term set used for matching + the site: query.
    core: str = ""
    variations: list[str] = Field(default_factory=list)
    adjacent: list[str] = Field(default_factory=list)

    # Inventory + cluster.
    sitemap_total: int = 0              # total pages discovered in the sitemap
    sitemap_found: bool = False         # was a sitemap discovered at all
    cluster: list[TopicalPage] = Field(default_factory=list)  # on-topic pages (union)
    cluster_size: int = 0              # on-core-topic pages in the cluster
    adjacent_pages: int = 0            # supporting-subtopic pages
    adjacent_covered: int = 0          # distinct adjacent subtopics with ≥1 page
    serp_indexed_hits: int = 0         # genuine on-topic site: results (noise-filtered)

    # Target verdict.
    target_in_cluster: bool = False
    target_is_canonical: bool = False  # target is the top site: result for the topic

    # Score + narrative.
    focus_ratio: float = 0.0           # cluster_size / sitemap_total
    score: float = 0.0                 # 0-100 topical-authority score
    band: str = "unknown"             # strong | moderate | thin | unknown
    supports_target: bool = False      # cluster is deep enough to support the page
    summary: str = ""                  # one-line human read
    ai_narrative: Optional[str] = None
    cost_usd: float = 0.0
    generated_at: datetime = Field(default_factory=_now)


# --------------------------------------------------------------------------- #
# Writing brief — the entity/EAV/correlation gaps translated into concrete
# writing recommendations for an SEO (prose) or an LLM (copy-paste prompt).
# --------------------------------------------------------------------------- #
class BriefItem(BaseModel):
    """One writing recommendation, with the evidence that produced it."""
    text: str                              # the instruction, human-readable
    evidence: str = ""                     # "5/19 top pages", "top pages say $61–$94", ...
    entity: str = ""                       # subject entity, when applicable
    attribute: str = ""                    # attribute family, when applicable


class WritingBrief(BaseModel):
    """Deterministic content-writing brief derived from the finished analysis.

    Everything here is grounded in the entity table + correlation roadmap — no
    LLM in the loop, so it is free, instant, and never hallucinates. The
    ``llm_prompt`` block is a self-contained instruction a user can paste into
    any LLM together with their draft/page copy.
    """
    keyword: str = ""
    target_url: str = ""
    entities_to_add: list[BriefItem] = Field(default_factory=list)   # high-consensus, absent
    topics_to_expand: list[BriefItem] = Field(default_factory=list)  # medium-consensus, absent
    facts_to_state: list[BriefItem] = Field(default_factory=list)    # graded attr gaps
    values_to_review: list[BriefItem] = Field(default_factory=list)  # off-consensus values
    relationships_to_make: list[BriefItem] = Field(default_factory=list)  # graded edge gaps
    style_targets: list[BriefItem] = Field(default_factory=list)     # content-shape goals
    # Information gain (Google patent: rank pages that ADD something):
    your_edge: list[BriefItem] = Field(default_factory=list)         # unique target facts — keep & amplify
    information_gain: list[BriefItem] = Field(default_factory=list)  # what NO page covers — the open opportunity
    summary: str = ""
    markdown: str = ""                     # the whole brief as export-ready markdown
    llm_prompt: str = ""                   # copy-paste block for any LLM


# --------------------------------------------------------------------------- #
# Ranking funnel — Google-as-a-cascade diagnostic layers. See
# docs/plans/google-aligned-ranking-funnel_2026-07-11.md. Convention throughout:
# per-page dicts are keyed by SERP rank, and rank 0 means the tracked target
# (same convention as EntityTable.cells).
# --------------------------------------------------------------------------- #
GATE_IDS = [
    "access", "lexical", "semantic", "intent",
    "entities", "quality", "authority", "engagement",
]


class GateScore(BaseModel):
    """One funnel gate: the target's standing vs the SERP on one Google stage."""
    gate: str                              # one of GATE_IDS
    name: str = ""                         # display name, e.g. "Semantic Relevance"
    score: Optional[float] = None          # 0-100 target percentile vs SERP pages; None = not evaluable
    verdict: str = "n/a"                   # pass | weak | fail | n/a
    evidence_tier: str = "computed"        # measured | computed | estimated | mixed
    weight: float = 1.0                    # how much this gate discriminates in THIS SERP
    details: list[str] = Field(default_factory=list)   # short evidence bullets
    per_page: dict[int, float] = Field(default_factory=dict)  # rank -> 0-100 (0 = target)


class FunnelResult(BaseModel):
    """The staged verdict: where in Google's cascade the target is losing."""
    gates: list[GateScore] = Field(default_factory=list)
    bottleneck_gate: str = ""              # earliest/worst failing gate id ("" = none)
    overall_score: Optional[float] = None  # weighted mean x bottleneck penalty, 0-100
    summary: str = ""


class CompetitorCard(BaseModel):
    """Why one competitor outranks the target: its biggest gate deltas."""
    rank: int
    domain: str = ""
    url: str = ""
    reasons: list[BriefItem] = Field(default_factory=list)  # text + evidence per delta


class SubIntent(BaseModel):
    """One sub-intent of the query (LLM fan-out), the unit of semantic coverage."""
    name: str
    description: str = ""


class SemanticReport(BaseModel):
    """Passage-level relevance: the pages x sub-intents coverage matrix."""
    sub_intents: list[SubIntent] = Field(default_factory=list)
    method: str = "tfidf"                  # "embeddings" | "tfidf" (fallback)
    # rank -> sub-intent name -> best-passage similarity 0-1
    coverage: dict[int, dict[str, float]] = Field(default_factory=dict)
    best_passage_sim: dict[int, float] = Field(default_factory=dict)  # rank -> 0-1
    content_focus: dict[int, float] = Field(default_factory=dict)     # rank -> 0-1 mean passage sim
    target_best_passage: str = ""          # the target passage that best answers the query
    open_gaps: list[str] = Field(default_factory=list)  # sub-intents NO page covers well
    summary: str = ""
    cost_usd: float = 0.0


class IntentFit(BaseModel):
    """SERP intent template vs the target's page type (format-fit gate)."""
    page_types: dict[int, str] = Field(default_factory=dict)   # rank -> page_type
    dominant_type: str = ""
    dominant_share: float = 0.0            # 0-1 share of classified pages
    target_type: str = ""
    fit: str = ""                          # match | partial | mismatch | ""
    serp_features: list[str] = Field(default_factory=list)
    is_ymyl: bool = False
    note: str = ""                         # one-sentence human verdict
    cost_usd: float = 0.0


class QualityReport(BaseModel):
    """LLM-judged content effort + helpfulness (the contentEffort proxy)."""
    effort: dict[int, float] = Field(default_factory=dict)        # rank -> 0-100
    helpfulness: dict[int, float] = Field(default_factory=dict)   # rank -> 0-100
    effort_notes: dict[int, str] = Field(default_factory=dict)    # rank -> short rationale
    summary: str = ""
    cost_usd: float = 0.0


class EngagementReport(BaseModel):
    """Simulated SERP engagement (NavBoost proxy) — always labeled estimated."""
    click_share: dict[int, float] = Field(default_factory=dict)   # rank -> 0-1 simulated
    click_reasons: dict[int, str] = Field(default_factory=dict)   # rank -> why the panel picked it
    satisfaction: dict[int, float] = Field(default_factory=dict)  # rank -> 0-100 post-click
    friction: dict[int, str] = Field(default_factory=dict)        # rank -> first friction point
    summary: str = ""
    cost_usd: float = 0.0


class AnalyzeReport(BaseModel):
    request: AnalyzeRequest
    serp: Serp
    page_factors: list[PageFactors] = Field(default_factory=list)
    correlations: list[FactorCorrelation] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    target: Optional[TargetSummary] = None
    ai_narrative: Optional[str] = None
    offpage: Optional[OffPagePanel] = None     # backlink power + quality (opt-in)
    brand: Optional[BrandPanel] = None         # branded search demand (opt-in)
    entity_table: Optional["EntityTable"] = None  # topical entity/EAV comparison (opt-in)
    topical: Optional[TopicalAuthorityReport] = None  # target-domain topical-authority cluster read
    writing_brief: Optional[WritingBrief] = None  # entity/EAV gaps as writing recommendations
    # Ranking-funnel layers (all optional; each degrades to None independently)
    semantic: Optional[SemanticReport] = None      # passage/sub-intent coverage matrix
    intent_fit: Optional[IntentFit] = None         # SERP format template vs target page type
    quality: Optional[QualityReport] = None        # LLM effort/helpfulness rubric
    engagement: Optional[EngagementReport] = None  # simulated SERP engagement
    funnel: Optional[FunnelResult] = None          # the staged gate verdict
    competitor_cards: list[CompetitorCard] = Field(default_factory=list)
    # diagnostics
    n_pages_analyzed: int = 0
    significance_threshold: float = 0.0
    pages_fetched_ok: int = 0
    timings_ms: dict[str, float] = Field(default_factory=dict)
    cost_usd: float = 0.0
    generated_at: datetime = Field(default_factory=_now)


# --------------------------------------------------------------------------- #
# Compare output (before/after algo update)
# --------------------------------------------------------------------------- #
class MoveStatus(str, Enum):
    entered = "entered"
    dropped = "dropped"
    up = "up"
    down = "down"
    same = "same"


class DomainMove(BaseModel):
    domain: str
    before_rank: Optional[int] = None
    after_rank: Optional[int] = None
    status: MoveStatus
    delta: Optional[int] = None            # before_rank - after_rank (positive = improved)
    is_winner: bool = False
    is_mega: bool = False                  # mega-platform (macro lens only)
    authority_score: Optional[float] = None
    traffic_visits: Optional[float] = None
    traffic_trend_pct: Optional[float] = None


class CompareReport(BaseModel):
    request: CompareRequest
    before: Serp
    after: Serp
    before_date: Optional[str] = None
    after_date: Optional[str] = None
    moves: list[DomainMove] = Field(default_factory=list)
    winners: list[DomainMove] = Field(default_factory=list)
    losers: list[DomainMove] = Field(default_factory=list)
    n1_flip: bool = False
    n1_before: Optional[str] = None
    n1_after: Optional[str] = None
    churn_pct: float = 0.0                 # share of top-N that changed
    macro: dict[str, Any] = Field(default_factory=dict)   # mega-platform aggregate lens
    ai_narrative: Optional[str] = None
    cost_usd: float = 0.0
    generated_at: datetime = Field(default_factory=_now)


# --------------------------------------------------------------------------- #
# Job / run wrapper (persistence unit)
# --------------------------------------------------------------------------- #
class RunKind(str, Enum):
    analyze = "analyze"
    compare = "compare"


class RunStatus(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"
    error = "error"


class Run(BaseModel):
    id: str
    kind: RunKind
    status: RunStatus = RunStatus.pending
    request: dict[str, Any] = Field(default_factory=dict)
    result: Optional[dict[str, Any]] = None   # AnalyzeReport | CompareReport dumped to json
    error: Optional[str] = None
    label: str = ""                           # human label e.g. 'analyze: "cordless impact driver"'
    owner_id: Optional[str] = None            # user who started the run; None = legacy/API-key run
    created_at: datetime = Field(default_factory=_now)
    finished_at: Optional[datetime] = None
