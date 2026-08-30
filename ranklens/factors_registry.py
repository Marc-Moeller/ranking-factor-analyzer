"""The factor registry — RankLens's "top ~20% of Cora" factor set.

This is the SHARED CONTRACT. The extraction layer computes a value for each
`FactorDef.id`; the correlation + recommendation layers read the metadata here
(group / phase / difficulty / top200 / direction) to build the roadmap.

Selection rule:
  (i) Cora tags it Top-200 or it sits in the strongest-correlation tier, AND
  (ii) it is computable from page HTML + the SERP we can fetch ourselves
       (no Cora-proprietary "shared factor" corpus).

`source` tells the pipeline where the value comes from:
  "html"      -> parsed from the fetched page HTML
  "serp"      -> derived from the SERP item (no page fetch needed)
  "corpus"    -> needs the 100-page corpus (LSI / TF-IDF)
  "authority" -> optional external backlink/authority API (off by default)
  "entity"    -> optional LLM entity/EAV extraction (off by default)

`direction` is the EXPECTED relationship, used only as a fallback label; the
real direction in a report is taken from the measured correlation sign.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FactorDef:
    id: str
    name: str
    group: str            # display grouping
    phase: str            # Cora-style roadmap phase
    source: str           # html | serp | corpus | authority
    unit: str = "Count"   # Count | Words | Characters | Percent | Tags | Other | Boolean
    category: str = "On Page"   # On Page | Web Development | Academic | Off Page | Technical
    difficulty: str = "Easy"    # Easy | Difficult
    top200: bool = False
    is_bool: bool = False
    direction: str = "more_is_better"   # more_is_better | less_is_better | nonlinear
    how: str = ""         # one-line compute hint for the extractor


REGISTRY: list[FactorDef] = [
    # ---------------- Phase 1: Title & Headings ----------------
    FactorDef("TITLE_LEN", "Title Length", "Title", "Title & Headings", "html", "Characters",
              how="character length of <title> text"),
    FactorDef("TITLE_WORDS", "Title Word Count", "Title", "Title & Headings", "html", "Words",
              how="whitespace tokens in <title>"),
    FactorDef("TITLE_VARS", "Variations in Title Tag", "Title", "Title & Headings", "html", "Variations",
              top200=True, how="count keyword-variation occurrences in title"),
    FactorDef("TITLE_LEAD_VARS", "Leading Variations in Title Tag", "Title", "Title & Headings", "html",
              "Variations", how="1 if title starts with a variation, else 0", is_bool=True),
    FactorDef("TITLE_HAS_DOMAIN", "Has Domain In Title", "Title", "Title & Headings", "html", "Boolean",
              is_bool=True, how="1 if title contains the registrable domain label"),
    FactorDef("OG_TITLE_VARS", "Variations in OpenGraph Title", "Title", "Title & Headings", "html",
              "Variations", how="variation count in og:title meta"),

    FactorDef("HEADING_COUNT", "Number of Heading Tags", "Headings", "Title & Headings", "html", "Tags",
              top200=True, how="count of <h1>..<h6>"),
    FactorDef("H1_COUNT", "Number of H1 Tags", "Headings", "Title & Headings", "html", "Tags",
              how="count of <h1>"),
    FactorDef("H1_VARS", "Variations in H1 Tags", "Headings", "Title & Headings", "html", "Variations",
              top200=True, how="variation count within <h1> text"),
    FactorDef("H2_VARS", "Variations in H2 Tags", "Headings", "Title & Headings", "html", "Variations",
              top200=True, how="variation count within <h2> text"),
    FactorDef("H3_VARS", "Variations in H3 Tags", "Headings", "Title & Headings", "html", "Variations",
              top200=True, how="variation count within <h3> text"),
    FactorDef("H1H6_VARS", "Variations in H1-H6 Tags", "Headings", "Title & Headings", "html", "Variations",
              top200=True, how="variation count across all headings"),
    FactorDef("H1H3_VARS", "Variations in H1, H2, and H3 Tags", "Headings", "Title & Headings", "html",
              "Variations", top200=True, how="variation count across h1-h3"),
    FactorDef("LEAD_VARS_H1H6", "Leading Variations in H1-H6 Tags", "Headings", "Title & Headings", "html",
              "Variations", top200=True, how="count of headings that START with a variation"),
    FactorDef("EXACT_H1H6", "Exact Matches in H1-H6 Tags", "Headings", "Title & Headings", "html",
              "Exact Matches", top200=True, how="count of literal full keyword phrase in headings"),
    FactorDef("QUESTIONS_H1", "Number of Questions in H1 Tags", "Headings", "Title & Headings", "html",
              "Count", how="headings that are questions (end with ? or start with who/what/...)"),

    # ---------------- Phase 2: Content ----------------
    FactorDef("WORD_COUNT", "Word Count", "Content", "Content", "html", "Words", top200=True,
              how="tokens in cleaned visible body text"),
    FactorDef("UNABRIDGED_WORD_COUNT", "Unabridged Word Count", "Content", "Content", "html", "Words",
              top200=True, how="tokens in all rendered text incl. nav/boilerplate"),
    FactorDef("CLEAN_TEXT_KB", "Clean Text Kilobytes", "Content", "Content", "html", "Other", top200=True,
              how="byte size of cleaned visible text / 1024"),
    FactorDef("PAGE_SIZE_KB", "Page Size Kilobytes", "Content", "Content", "html", "Other", top200=True,
              how="byte size of full HTML / 1024"),
    FactorDef("SENTENCES", "Number of Sentences", "Content", "Content", "html", "Sentences", top200=True,
              how="sentence-split count of body text"),
    FactorDef("AVG_WORDS_SENTENCE", "Average Words Per Sentence", "Content", "Content", "html", "Words",
              how="word_count / sentence_count"),
    FactorDef("P_TAGS", "Number of P Tags", "Content", "Content", "html", "Tags", top200=True,
              how="count of <p>"),
    FactorDef("HTML_TAGS", "Number of HTML Tags", "Content", "Content", "html", "Tags", top200=True,
              how="total element count in the DOM"),
    FactorDef("BODY_VARS", "Variations in Body Tags", "Content", "Content", "html", "Variations",
              top200=True, how="variation count in visible body text"),
    FactorDef("HTML_VARS", "Variations in HTML Tags", "Content", "Content", "html", "Variations",
              top200=True, how="variation count in the full HTML source"),
    FactorDef("P_VARS", "Variations in P Tags", "Content", "Content", "html", "Variations", top200=True,
              how="variation count within <p> text"),
    FactorDef("EXACT_HTML", "Exact Matches in the HTML Tag", "Content", "Content", "html", "Exact Matches",
              top200=True, how="count of literal full keyword phrase in HTML"),
    FactorDef("VAR_DENSITY_30KB", "Variation Density in Top 30KB", "Content", "Content", "html", "Percent",
              category="Academic", difficulty="Difficult", top200=True,
              how="variation occurrences / words within first 30KB of HTML, as %"),
    FactorDef("CLEAN_KW_DENSITY", "Clean Keyword Density", "Content", "Content", "html", "Percent",
              category="Academic", difficulty="Difficult", top200=True,
              how="keyword-variation matches / clean word count, as %"),
    FactorDef("TF", "Term Frequency", "Content", "Content", "corpus", "Other", top200=True,
              how="term frequency of keyword on the page"),
    FactorDef("TFIDF", "TF/IDF", "Content", "Content", "corpus", "Other",
              how="TF * IDF, IDF from the N-page corpus"),
    FactorDef("FLESCH_EASE", "Flesch Kincaid Reading Ease", "Content", "Content", "html", "Other",
              direction="nonlinear", how="standard readability ease score on body text"),
    FactorDef("FLESCH_GRADE", "Flesch Kincaid Grade Level", "Content", "Content", "html", "Other",
              direction="nonlinear", how="standard readability grade level on body text"),
    FactorDef("BYTES_TO_FIRST_MATCH", "Number of Bytes to First Match", "Content", "Content", "html",
              "Other", direction="less_is_better",
              how="byte offset of first keyword-variation occurrence in HTML"),

    # ---------------- Phase 4: Diversity (LSI) ----------------
    FactorDef("UNIQUE_VARS", "Number of Unique Variations Used", "Diversity", "Diversity", "html",
              "Variations", category="Academic", difficulty="Difficult", top200=True,
              how="distinct variations appearing anywhere on page"),
    FactorDef("LSI_SENTENCES", "LSI Words in Sentences", "Diversity", "Diversity", "corpus", "LSI",
              top200=True, how="count of corpus-derived LSI terms across sentences"),
    FactorDef("UNIQUE_LSI", "Number of Unique LSI Words Used", "Diversity", "Diversity", "corpus", "LSI",
              category="Academic", difficulty="Difficult", top200=True,
              how="distinct corpus LSI terms present on page"),

    # ---------------- Phase 6: Search Result Presentation (SERP, no fetch) ----------------
    FactorDef("SR_DOMAIN_COM", "Search Result Domain is .com/.net/.org", "SERP Presentation",
              "Search Result Presentation", "serp", "Boolean", difficulty="Difficult", is_bool=True,
              how="1 if result domain TLD in {com,net,org}"),
    FactorDef("SR_DOMAIN_HYPHEN", "Search Result Domain Has Hyphen", "SERP Presentation",
              "Search Result Presentation", "serp", "Boolean", difficulty="Difficult", is_bool=True,
              direction="less_is_better", how="1 if domain contains a hyphen"),
    FactorDef("SR_DOMAIN_LEN", "Search Result Domain Length", "SERP Presentation",
              "Search Result Presentation", "serp", "Characters", difficulty="Difficult",
              how="char length of result domain"),
    FactorDef("SR_URL_HAS_YEAR", "Search Result URL has Year", "SERP Presentation",
              "Search Result Presentation", "serp", "Boolean", difficulty="Difficult", is_bool=True,
              how="1 if URL contains a 19xx/20xx year"),
    FactorDef("SR_SUMMARY_LEN", "Search Result Summary Length", "SERP Presentation",
              "Search Result Presentation", "serp", "Characters", difficulty="Difficult",
              how="char length of SERP snippet"),
    FactorDef("SR_TITLE_VARS", "Variations in Search Result Link Text", "SERP Presentation",
              "Search Result Presentation", "serp", "Variations", difficulty="Difficult", top200=True,
              how="variation count in the SERP title"),
    FactorDef("SR_URL_VARS", "Variations in Search Result Display URL", "SERP Presentation",
              "Search Result Presentation", "serp", "Variations", difficulty="Difficult", top200=True,
              how="variation count in the displayed URL / path"),
    FactorDef("SR_SUMMARY_VARS", "Variations in Search Result Summary", "SERP Presentation",
              "Search Result Presentation", "serp", "Variations", difficulty="Difficult", top200=True,
              how="variation count in the SERP snippet"),

    # ---------------- Phase 7: Outbound Links ----------------
    FactorDef("LINKS", "Number of Links (Content)", "Links", "Outbound Links", "html", "Count",
              category="Web Development", difficulty="Difficult", top200=True,
              how="total in-content <a href> (excludes header/footer/nav)"),
    FactorDef("INTERNAL_LINKS", "Number of Internal Links (Content)", "Links", "Outbound Links", "html",
              "Count", category="On Page", difficulty="Easy", top200=True,
              how="in-content <a href> on same registrable domain (excludes header/footer/nav)"),
    FactorDef("EXTERNAL_LINKS", "Number of External Links (Content)", "Links", "Outbound Links", "html",
              "Count", category="On Page", difficulty="Easy",
              how="in-content <a href> to other domains (excludes header/footer/nav)"),
    FactorDef("HTTPS_LINKS", "Number of HTTPS Links (Content)", "Links", "Outbound Links", "html", "Count",
              category="Web Development", difficulty="Difficult", top200=True,
              how="in-content <a href> using https (excludes header/footer/nav)"),
    FactorDef("DOFOLLOW_EXT_LINKS", "Number of DoFollow External Links (Content)", "Links",
              "Outbound Links", "html", "Count", category="Web Development", difficulty="Difficult",
              how="external in-content <a> without rel=nofollow (excludes header/footer/nav)"),
    FactorDef("NOFOLLOW_LINKS", "Number of NoFollow External Links (Content)", "Links", "Outbound Links",
              "html", "Count", category="Web Development", difficulty="Difficult",
              how="external in-content <a> with rel=nofollow (excludes header/footer/nav)"),
    FactorDef("A_VARS", "Variations in A Tags", "Links", "Outbound Links", "html", "Variations",
              top200=True, how="variation count in anchor text"),
    FactorDef("SOCIAL_LINKS", "Number of Social Pages", "Links", "Social Integration", "html", "Count",
              how="distinct links to facebook/instagram/x/youtube/linkedin/pinterest"),

    # ---------------- Phase 9: Images ----------------
    FactorDef("IMAGES", "Number of Images", "Images", "Images", "html", "Count", top200=True,
              how="count of <img>"),
    FactorDef("IMAGES_ALT", "Number of Images with ALT Text", "Images", "Images", "html", "Count",
              top200=True, how="<img> with non-empty alt"),
    FactorDef("ALT_VARS", "Variations in ALT Attributes", "Images", "Images", "html", "Variations",
              top200=True, how="variation count across all alt text"),

    # ---------------- Phase 12: Schema ----------------
    FactorDef("HAS_JSONLD", "Has JSON-LD Schema Markup", "Schema", "Schema", "html", "Boolean",
              category="Web Development", difficulty="Difficult", top200=True, is_bool=True,
              how="1 if any <script type=application/ld+json> present"),
    FactorDef("SCHEMA_TYPES", "Number of Schema Types", "Schema", "Schema", "html", "Count",
              category="Web Development", difficulty="Difficult", how="distinct @type values in JSON-LD"),
    FactorDef("USES_ORG_LOCALBIZ", "Uses Organization/LocalBusiness Schema", "Schema", "Schema", "html",
              "Boolean", category="Web Development", difficulty="Difficult", is_bool=True,
              how="1 if Organization or LocalBusiness @type present"),
    FactorDef("USES_PRODUCT_OFFER", "Uses Product/Offer Schema", "Schema", "Schema", "html", "Boolean",
              category="Web Development", difficulty="Difficult", is_bool=True,
              how="1 if Product or Offer (price/highPrice/lowPrice) present"),
    FactorDef("USES_AGG_RATING", "Uses AggregateRating Schema", "Schema", "Schema", "html", "Boolean",
              category="Web Development", difficulty="Difficult", is_bool=True,
              how="1 if AggregateRating / ratingValue present"),
    FactorDef("USES_FAQ", "Uses FAQ/Question Schema", "Schema", "Schema", "html", "Boolean",
              category="Web Development", difficulty="Difficult", is_bool=True,
              how="1 if FAQPage or Question present"),
    FactorDef("USES_BREADCRUMB", "Uses BreadcrumbList Schema", "Schema", "Schema", "html", "Boolean",
              category="Web Development", difficulty="Difficult", is_bool=True,
              how="1 if BreadcrumbList present"),
    FactorDef("CLAIMED_BRANDS", "Number of Claimed Brands (sameAs)", "Schema", "Schema", "html", "Count",
              category="Web Development", difficulty="Difficult",
              how="distinct sameAs social/brand URLs in schema"),

    # ---------------- Phase X: Technical ----------------
    FactorDef("META_DESC_LEN", "Meta Description Length", "Technical", "Technical", "html", "Characters",
              how="char length of meta description"),
    FactorDef("META_DESC_WORDS", "Word Count in Meta Description", "Technical", "Technical", "html", "Words",
              how="word count of meta description"),
    FactorDef("META_DESC_VARS", "Variations in Meta Description", "Technical", "Technical", "html",
              "Variations", how="variation count in meta description"),
    FactorDef("META_KEYWORDS", "Number of Meta Keywords", "Technical", "Technical", "html", "Count",
              how="terms in <meta name=keywords>"),
    FactorDef("HAS_DOCTYPE", "Has DocType Tag", "Technical", "Technical", "html", "Boolean", is_bool=True,
              how="1 if <!DOCTYPE> present"),
    FactorDef("HAS_FORM", "Has Form", "Technical", "Technical", "html", "Boolean", is_bool=True,
              how="1 if a <form> is present"),
    FactorDef("HAS_VIDEO", "Has Video or YouTube Embed", "Technical", "Technical", "html", "Boolean",
              is_bool=True, how="1 if <video> or youtube/vimeo iframe present"),
    FactorDef("HAS_PRIVACY", "Has Privacy Policy", "Technical", "Technical", "html", "Boolean",
              is_bool=True, how="1 if a link/anchor mentions privacy policy"),
    FactorDef("HAS_EMAIL", "Has Email", "Technical", "Technical", "html", "Boolean", is_bool=True,
              how="1 if a mailto: or email regex present"),
    FactorDef("LOAD_MS", "Load Time Milliseconds", "Technical", "Technical", "html", "Other",
              direction="less_is_better", how="measured fetch latency in ms"),

    # ---------------- Phase 3: Authority (optional, external API) ----------------
    # Domain-level (bulk-analysis, target_type=root_domain)
    FactorDef("REF_DOMAINS", "Number of Referring Domains", "Authority", "Authority", "backlinks", "Count",
              category="Off Page", difficulty="Difficult", how="domain referring domains (backlink provider)"),
    FactorDef("BACKLINKS", "Number of Backlinks", "Authority", "Authority", "backlinks", "Count",
              category="Off Page", difficulty="Difficult", how="domain total backlinks (backlink provider)"),
    FactorDef("AUTHORITY_SCORE", "Domain Authority Score", "Authority", "Authority", "backlinks", "Other",
              category="Off Page", difficulty="Difficult", how="domain authority score 0-100 (backlink provider)"),
    # Page-level (bulk-analysis, target_type=url) — the exact ranking URL
    FactorDef("PAGE_AUTHORITY", "Page Authority Score", "Authority", "Authority", "backlinks", "Other",
              category="Off Page", difficulty="Difficult", top200=True,
              how="authority score 0-100 of the exact ranking URL (backlink provider)"),
    FactorDef("PAGE_REF_DOMAINS", "Referring Domains to Page", "Authority", "Authority", "backlinks", "Count",
              category="Off Page", difficulty="Difficult", top200=True,
              how="referring domains pointing at the exact URL (backlink provider)"),
    FactorDef("PAGE_BACKLINKS", "Backlinks to Page", "Authority", "Authority", "backlinks", "Count",
              category="Off Page", difficulty="Difficult",
              how="total backlinks pointing at the exact URL (backlink provider)"),
    FactorDef("PAGE_FOLLOW_RATIO", "Page DoFollow Backlink Ratio", "Authority", "Authority", "backlinks",
              "Percent", category="Off Page", difficulty="Difficult",
              how="follow / (follow + nofollow) of backlinks to the URL (backlink provider)"),
    FactorDef("HOMEPAGE_AUTHORITY", "Homepage Authority Score", "Authority", "Authority", "backlinks", "Other",
              category="Off Page", difficulty="Difficult",
              how="authority score 0-100 of the site homepage (backlink provider)"),

    # ---------------- Phase 3 (cont.): Brand (optional, external API) ----------------
    FactorDef("BRAND_VOLUME", "Branded Search Volume", "Brand", "Brand", "backlinks", "Count",
              category="Off Page", difficulty="Difficult", direction="more_is_better",
              how="monthly branded search volume for the ranking domain (backlink provider)"),

    # ---------------- Phase 5: Entities (optional, LLM extraction) ----------------
    FactorDef("ENTITIES_TITLE", "Entities in Title Tag", "Entity", "Entities", "entity", "Entities",
              category="Academic", top200=True,
              how="distinct discovered entities whose surface form appears in <title>"),
    FactorDef("ENTITIES_H1", "Entities in H1 Tags", "Entity", "Entities", "entity", "Entities",
              category="Academic", top200=True, how="distinct entities appearing in <h1> text"),
    FactorDef("ENTITIES_H2", "Entities in H2 Tags", "Entity", "Entities", "entity", "Entities",
              category="Academic", top200=True, how="distinct entities appearing in <h2> text"),
    FactorDef("ENTITIES_H3", "Entities in H3 Tags", "Entity", "Entities", "entity", "Entities",
              category="Academic", how="distinct entities appearing in <h3> text"),
    FactorDef("ENTITIES_BODY", "Entities in the HTML Tag", "Entity", "Entities", "entity", "Entities",
              category="Academic", top200=True, how="distinct entities appearing in the body text"),
    FactorDef("ENTITIES_SENTENCES", "Entities in Sentences", "Entity", "Entities", "entity", "Entities",
              category="Academic", top200=True, how="total entity mentions across body sentences"),
    FactorDef("DISTINCT_ENTITIES", "Number of Distinct Entities Used", "Entity", "Entities", "entity",
              "Entities", category="Academic", top200=True, how="count of distinct canonical entities on page"),
    FactorDef("ENTITY_SALIENCE_SUM", "Total Entity Salience", "Entity", "Entities", "entity", "Other",
              category="Academic", how="sum of discovered-entity salience scores"),
    FactorDef("EAV_COMPLETENESS", "Topical EAV Completeness", "Entity", "Entities", "entity", "Percent",
              category="Academic", top200=True,
              how="covered (entity,attribute) pairs / the top-N union of pairs, as %"),

    # ---------------- Funnel: semantic relevance (passage/sub-intent layer) ----------------
    FactorDef("SALIENT_TERM_COVERAGE", "Salient Term Coverage", "Semantic", "Content", "semantic",
              "Percent", category="Academic", top200=True,
              how="share of the SERP corpus's top salient n-grams present in the page's body"),
    FactorDef("BEST_PASSAGE_SIM", "Best Passage Relevance", "Semantic", "Content", "semantic",
              "Percent", category="Academic", top200=True,
              how="best passage-to-query similarity (passage-indexing proxy), 0-100"),
    FactorDef("SUBINTENT_COVERAGE", "Sub-Intent Coverage", "Semantic", "Content", "semantic",
              "Percent", category="Academic", top200=True,
              how="share of query sub-intents with a passage above the similarity threshold"),
    FactorDef("CONTENT_FOCUS", "Content Focus", "Semantic", "Content", "semantic", "Percent",
              category="Academic",
              how="mean passage-to-query similarity — low = the page dilutes its topic"),

    # ---------------- Funnel: quality / effort / trust ----------------
    FactorDef("CONTENT_EFFORT", "Content Effort", "Quality", "Trust & Experience", "quality",
              "Percent", category="Academic", top200=True, difficulty="Difficult",
              how="LLM-rubric effort estimate 0-100 (originality, depth, first-hand evidence)"),
    FactorDef("HELPFULNESS", "Helpfulness", "Quality", "Trust & Experience", "quality", "Percent",
              category="Academic", difficulty="Difficult",
              how="LLM rubric vs Google's helpful-content self-assessment questions, 0-100"),
    FactorDef("TRUST_BYLINE", "Has Author Byline", "Trust", "Trust & Experience", "trust",
              "Boolean", is_bool=True, how="1 if a visible author byline is present"),
    FactorDef("TRUST_AUTHOR_PAGE", "Links to an Author Page", "Trust", "Trust & Experience",
              "trust", "Boolean", is_bool=True,
              how="1 if the byline links to an author/about page"),
    FactorDef("TRUST_PERSON_SCHEMA", "Person Schema", "Trust", "Trust & Experience", "trust",
              "Boolean", is_bool=True, how="1 if JSON-LD Person markup is present"),
    FactorDef("TRUST_ORG_SCHEMA", "Organization Schema", "Trust", "Trust & Experience", "trust",
              "Boolean", is_bool=True, how="1 if JSON-LD Organization markup is present"),
    FactorDef("TRUST_SAMEAS", "sameAs Profile Links", "Trust", "Trust & Experience", "trust",
              "Count", how="count of schema sameAs links to external profiles"),
    FactorDef("TRUST_OUTBOUND_CITATIONS", "Outbound Citations", "Trust", "Trust & Experience",
              "trust", "Count",
              how="outbound links to authoritative non-social external domains"),
    FactorDef("TRUST_CONTACT_LINK", "Contact/About Reachable", "Trust", "Trust & Experience",
              "trust", "Boolean", is_bool=True,
              how="1 if a contact or about page is linked from the page"),
    FactorDef("TRUST_VISIBLE_DATE", "Visible Publish/Update Date", "Trust", "Trust & Experience",
              "trust", "Boolean", is_bool=True,
              how="1 if a machine-readable or visible publish/modified date is present"),

    # ---------------- Funnel: access / experience (CrUX = real Chrome field data) ----------------
    FactorDef("CRUX_LCP_MS", "CrUX LCP (p75, ms)", "Experience", "Technical", "crux", "Other",
              category="Technical", direction="less_is_better",
              how="origin p75 Largest Contentful Paint from the CrUX API"),
    FactorDef("CRUX_INP_MS", "CrUX INP (p75, ms)", "Experience", "Technical", "crux", "Other",
              category="Technical", direction="less_is_better",
              how="origin p75 Interaction to Next Paint from the CrUX API"),
    FactorDef("CRUX_CLS", "CrUX CLS (p75, x100)", "Experience", "Technical", "crux", "Other",
              category="Technical", direction="less_is_better",
              how="origin p75 Cumulative Layout Shift from the CrUX API (x100)"),
    FactorDef("CRUX_HAS_DATA", "In Chrome UX Report", "Experience", "Technical", "crux",
              "Boolean", is_bool=True,
              how="1 if the origin has CrUX field data at all (a popularity floor)"),

    # ---------------- Funnel: engagement (simulated — NavBoost proxy) ----------------
    FactorDef("CLICK_SHARE", "Simulated SERP Click Share", "Engagement", "Search Result Presentation",
              "engagement", "Percent", category="On Page", top200=True,
              how="share of simulated searcher-panel clicks won by this result's title+snippet"),
    FactorDef("SATISFACTION", "Post-Click Satisfaction", "Engagement", "Search Result Presentation",
              "engagement", "Percent", category="On Page", difficulty="Difficult",
              how="LLM-searcher judgment: does the first viewport satisfy the query, 0-100"),
]

BY_ID: dict[str, FactorDef] = {f.id: f for f in REGISTRY}

# Phase display order for the roadmap.
PHASE_ORDER: list[str] = [
    "Title & Headings",
    "Content",
    "Entities",
    "Diversity",
    "Schema",
    "Images",
    "Outbound Links",
    "Social Integration",
    "Technical",
    "Search Result Presentation",
    "Trust & Experience",
    "Authority",
    "Brand",
]


def html_factor_ids() -> list[str]:
    return [f.id for f in REGISTRY if f.source == "html"]


def serp_factor_ids() -> list[str]:
    return [f.id for f in REGISTRY if f.source == "serp"]


def corpus_factor_ids() -> list[str]:
    return [f.id for f in REGISTRY if f.source == "corpus"]


def authority_factor_ids() -> list[str]:
    return [f.id for f in REGISTRY if f.source == "authority"]


def entity_factor_ids() -> list[str]:
    return [f.id for f in REGISTRY if f.source == "entity"]
