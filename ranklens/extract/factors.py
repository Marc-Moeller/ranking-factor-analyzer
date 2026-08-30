"""The main HTML factor extractor.

``extract_html_factors`` turns a fetched page's HTML into every registry factor
whose ``source == "html"`` (plus the Schema-group factors via ``schema.py``),
returning the factor dict and the cleaned body text (which the corpus layer and
the caller reuse).

Robustness is a hard requirement: one malformed page must never crash a run.
Every factor group is wrapped in try/except and missing numerics default to 0.0.
"""
from __future__ import annotations

import re
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup

from ..keywords import split_sentences, word_count
from .schema import schema_factors

# heading-level question starters
_QUESTION_WORDS = (
    "who", "what", "when", "where", "why", "how", "which",
    "can", "do", "does", "is", "are",
)
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_SOCIAL_HOSTS = (
    "facebook.com", "instagram.com", "twitter.com", "x.com", "youtube.com",
    "youtu.be", "linkedin.com", "pinterest.com", "tiktok.com",
)
_VIDEO_HOSTS = ("youtube.com", "youtu.be", "vimeo.com", "youtube-nocookie.com")
_STRIP_TAGS = ("script", "style", "noscript", "template", "svg")
# Site-wide chrome — links here are boilerplate, not in-content linking, so they
# are excluded from every link count (see the Links section).
_CHROME_TAGS = ("header", "footer", "nav")
_CHROME_ROLES_RE = re.compile(r"^(?:banner|contentinfo|navigation)$", re.I)
_TOP_30KB = 30720


def _registrable_label(domain: str) -> str:
    """The second-level label of a domain, e.g. 'examplestore' from 'www.examplestore.com'."""
    if not domain:
        return ""
    host = domain.lower().strip()
    if host.startswith("www."):
        host = host[4:]
    parts = [p for p in host.split(".") if p]
    if not parts:
        return ""
    # for a.co.uk style, the registrable label is the 3rd-from-last; good enough
    # to take the part before the final 1-2 short TLD pieces.
    if len(parts) >= 3 and len(parts[-2]) <= 3 and len(parts[-1]) <= 3:
        return parts[-3]
    if len(parts) >= 2:
        return parts[-2]
    return parts[0]


def _registrable_domain(domain: str) -> str:
    """Best-effort registrable domain (label + tld) for same-site comparison."""
    if not domain:
        return ""
    host = domain.lower().strip()
    if host.startswith("www."):
        host = host[4:]
    parts = [p for p in host.split(".") if p]
    if len(parts) <= 2:
        return ".".join(parts)
    if len(parts[-2]) <= 3 and len(parts[-1]) <= 3:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def _flesch(text: str) -> tuple[float, float]:
    try:
        import textstat
        if not text or not text.strip():
            return 0.0, 0.0
        ease = float(textstat.flesch_reading_ease(text))
        grade = float(textstat.flesch_kincaid_grade(text))
        return ease, grade
    except Exception:
        return 0.0, 0.0


def extract_html_factors(
    html: str,
    url: str,
    domain: str,
    vset,
    load_ms: float | None,
) -> tuple[dict[str, float], str]:
    factors: dict[str, float] = {}
    html = html or ""

    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        # fall back to the stdlib parser if lxml chokes
        soup = BeautifulSoup(html, "html.parser")

    reg_label = _registrable_label(domain)
    reg_domain = _registrable_domain(domain)

    # ---- clean body text (shared) -------------------------------------------
    clean_body_text = ""
    try:
        body_soup = BeautifulSoup(html, "lxml")
        for tag in body_soup.find_all(_STRIP_TAGS):
            tag.decompose()
        clean_body_text = body_soup.get_text(" ", strip=True)
        clean_body_text = re.sub(r"\s+", " ", clean_body_text).strip()
    except Exception:
        clean_body_text = ""

    # =====================================================================
    # Title + OG
    # =====================================================================
    try:
        title_el = soup.find("title")
        title = title_el.get_text(" ", strip=True) if title_el else ""
        factors["TITLE_LEN"] = float(len(title))
        factors["TITLE_WORDS"] = float(len(title.split()))
        factors["TITLE_VARS"] = float(vset.count(title))
        factors["TITLE_LEAD_VARS"] = 1.0 if vset.starts_with_variation(title) else 0.0
        factors["TITLE_HAS_DOMAIN"] = (
            1.0 if reg_label and reg_label in title.lower() else 0.0
        )
    except Exception:
        for k in ("TITLE_LEN", "TITLE_WORDS", "TITLE_VARS", "TITLE_LEAD_VARS", "TITLE_HAS_DOMAIN"):
            factors.setdefault(k, 0.0)

    try:
        og = soup.find("meta", attrs={"property": re.compile(r"^og:title$", re.I)})
        og_title = (og.get("content") or "") if og else ""
        factors["OG_TITLE_VARS"] = float(vset.count(og_title))
    except Exception:
        factors["OG_TITLE_VARS"] = 0.0

    # =====================================================================
    # Headings
    # =====================================================================
    try:
        headings: dict[int, list[str]] = {lvl: [] for lvl in range(1, 7)}
        for lvl in range(1, 7):
            for h in soup.find_all(f"h{lvl}"):
                txt = h.get_text(" ", strip=True)
                if txt:
                    headings[lvl].append(txt)

        h1_text = " ".join(headings[1])
        h2_text = " ".join(headings[2])
        h3_text = " ".join(headings[3])
        all_heads = [t for lvl in range(1, 7) for t in headings[lvl]]
        all_head_text = " ".join(all_heads)
        h1h3_text = " ".join(headings[1] + headings[2] + headings[3])

        factors["HEADING_COUNT"] = float(sum(len(headings[l]) for l in range(1, 7)))
        factors["H1_COUNT"] = float(len(headings[1]))
        factors["H1_VARS"] = float(vset.count(h1_text))
        factors["H2_VARS"] = float(vset.count(h2_text))
        factors["H3_VARS"] = float(vset.count(h3_text))
        factors["H1H6_VARS"] = float(vset.count(all_head_text))
        factors["H1H3_VARS"] = float(vset.count(h1h3_text))
        factors["LEAD_VARS_H1H6"] = float(
            sum(1 for t in all_heads if vset.starts_with_variation(t))
        )
        factors["EXACT_H1H6"] = float(vset.count_exact(all_head_text))

        def _is_question(t: str) -> bool:
            tl = t.strip().lower()
            if not tl:
                return False
            if tl.endswith("?"):
                return True
            first = tl.split()[0] if tl.split() else ""
            return first in _QUESTION_WORDS

        factors["QUESTIONS_H1"] = float(sum(1 for t in headings[1] if _is_question(t)))
    except Exception:
        for k in ("HEADING_COUNT", "H1_COUNT", "H1_VARS", "H2_VARS", "H3_VARS",
                  "H1H6_VARS", "H1H3_VARS", "LEAD_VARS_H1H6", "EXACT_H1H6", "QUESTIONS_H1"):
            factors.setdefault(k, 0.0)

    # =====================================================================
    # Content / body
    # =====================================================================
    try:
        wc = word_count(clean_body_text)
        factors["WORD_COUNT"] = float(wc)
    except Exception:
        wc = 0
        factors["WORD_COUNT"] = 0.0

    try:
        full_text = soup.get_text(" ", strip=True)
        factors["UNABRIDGED_WORD_COUNT"] = float(word_count(full_text))
    except Exception:
        factors["UNABRIDGED_WORD_COUNT"] = 0.0

    try:
        factors["CLEAN_TEXT_KB"] = len(clean_body_text.encode("utf-8")) / 1024.0
    except Exception:
        factors["CLEAN_TEXT_KB"] = 0.0

    try:
        factors["PAGE_SIZE_KB"] = len(html.encode("utf-8")) / 1024.0
    except Exception:
        factors["PAGE_SIZE_KB"] = 0.0

    try:
        n_sent = len(split_sentences(clean_body_text))
        factors["SENTENCES"] = float(n_sent)
        factors["AVG_WORDS_SENTENCE"] = float(wc) / float(max(n_sent, 1))
    except Exception:
        factors["SENTENCES"] = 0.0
        factors["AVG_WORDS_SENTENCE"] = 0.0

    try:
        p_tags = soup.find_all("p")
        factors["P_TAGS"] = float(len(p_tags))
        p_text = " ".join(p.get_text(" ", strip=True) for p in p_tags)
        factors["P_VARS"] = float(vset.count(p_text))
    except Exception:
        factors["P_TAGS"] = 0.0
        factors["P_VARS"] = 0.0

    try:
        factors["HTML_TAGS"] = float(len(soup.find_all(True)))
    except Exception:
        factors["HTML_TAGS"] = 0.0

    try:
        factors["BODY_VARS"] = float(vset.count(clean_body_text))
    except Exception:
        factors["BODY_VARS"] = 0.0

    try:
        factors["HTML_VARS"] = float(vset.count(html))
    except Exception:
        factors["HTML_VARS"] = 0.0

    try:
        factors["EXACT_HTML"] = float(vset.count_exact(html))
    except Exception:
        factors["EXACT_HTML"] = 0.0

    try:
        slice_html = html.encode("utf-8")[:_TOP_30KB].decode("utf-8", "ignore")
        slice_soup = BeautifulSoup(slice_html, "lxml")
        for tag in slice_soup.find_all(_STRIP_TAGS):
            tag.decompose()
        slice_text = slice_soup.get_text(" ", strip=True)
        slice_wc = word_count(slice_text)
        slice_vars = vset.count(slice_text)
        factors["VAR_DENSITY_30KB"] = (slice_vars / max(slice_wc, 1)) * 100.0
    except Exception:
        factors["VAR_DENSITY_30KB"] = 0.0

    try:
        factors["CLEAN_KW_DENSITY"] = (
            factors.get("BODY_VARS", 0.0) / float(max(wc, 1)) * 100.0
        )
    except Exception:
        factors["CLEAN_KW_DENSITY"] = 0.0

    try:
        ease, grade = _flesch(clean_body_text)
        factors["FLESCH_EASE"] = float(ease)
        factors["FLESCH_GRADE"] = float(grade)
    except Exception:
        factors["FLESCH_EASE"] = 0.0
        factors["FLESCH_GRADE"] = 0.0

    try:
        off = vset.first_match_offset(html)
        factors["BYTES_TO_FIRST_MATCH"] = float(off if off is not None else len(html))
    except Exception:
        factors["BYTES_TO_FIRST_MATCH"] = float(len(html))

    # UNIQUE_VARS (Diversity, source=html)
    try:
        factors["UNIQUE_VARS"] = float(vset.unique_used(html))
    except Exception:
        factors["UNIQUE_VARS"] = 0.0

    # =====================================================================
    # Images
    # =====================================================================
    try:
        imgs = soup.find_all("img")
        factors["IMAGES"] = float(len(imgs))
        alts = []
        n_alt = 0
        for im in imgs:
            alt = im.get("alt")
            if isinstance(alt, list):
                alt = " ".join(alt)
            if alt and alt.strip():
                n_alt += 1
                alts.append(alt)
        factors["IMAGES_ALT"] = float(n_alt)
        factors["ALT_VARS"] = float(vset.count(" ".join(alts)))
    except Exception:
        factors["IMAGES"] = 0.0
        factors["IMAGES_ALT"] = 0.0
        factors["ALT_VARS"] = 0.0

    # =====================================================================
    # Links — counted in the MAIN CONTENT only. Header/footer/nav anchors are
    # site-wide boilerplate (menus, legal, social) that repeat on every page and
    # drown out the editorial in-content links that actually signal relevance, so
    # they are stripped before counting. The "Absolute URLs" factor was dropped
    # in favour of an explicit INTERNAL vs EXTERNAL content split.
    # =====================================================================
    try:
        content_soup = BeautifulSoup(html, "lxml")
        for tag in content_soup.find_all(_STRIP_TAGS):
            tag.decompose()
        for tag in content_soup.find_all(_CHROME_TAGS):
            tag.decompose()
        for tag in content_soup.find_all(attrs={"role": _CHROME_ROLES_RE}):
            tag.decompose()
        anchors = content_soup.find_all("a", href=True)

        links = internal = external = https = dofollow_ext = nofollow = 0
        anchor_texts: list[str] = []

        for a in anchors:
            href = (a.get("href") or "").strip()
            if not href:
                continue
            links += 1
            anchor_texts.append(a.get_text(" ", strip=True))

            rel = a.get("rel")
            if isinstance(rel, (list, tuple)):
                rel_str = " ".join(rel).lower()
            else:
                rel_str = str(rel or "").lower()
            is_nofollow = "nofollow" in rel_str

            if href.lower().startswith("https"):
                https += 1

            try:
                resolved = urljoin(url, href)
                rparts = urlsplit(resolved)
                rhost = rparts.netloc.lower()
                same_site = _registrable_domain(rhost) == reg_domain and reg_domain != ""
            except Exception:
                same_site = False

            if same_site:
                internal += 1
            else:
                external += 1
                if is_nofollow:
                    nofollow += 1
                else:
                    dofollow_ext += 1

        factors["LINKS"] = float(links)
        factors["INTERNAL_LINKS"] = float(internal)
        factors["EXTERNAL_LINKS"] = float(external)
        factors["HTTPS_LINKS"] = float(https)
        factors["DOFOLLOW_EXT_LINKS"] = float(dofollow_ext)
        factors["NOFOLLOW_LINKS"] = float(nofollow)
        factors["A_VARS"] = float(vset.count(" ".join(anchor_texts)))
    except Exception:
        for k in ("LINKS", "INTERNAL_LINKS", "EXTERNAL_LINKS", "HTTPS_LINKS",
                  "DOFOLLOW_EXT_LINKS", "NOFOLLOW_LINKS", "A_VARS"):
            factors.setdefault(k, 0.0)

    # Social profile links live in the header/footer by design, so they are
    # counted across the FULL page (not the content-only slice above).
    try:
        social_hosts_found: set[str] = set()
        for a in soup.find_all("a", href=True):
            low_href = (a.get("href") or "").lower()
            for host in _SOCIAL_HOSTS:
                if host in low_href:
                    social_hosts_found.add(host)
        factors["SOCIAL_LINKS"] = float(len(social_hosts_found))
    except Exception:
        factors.setdefault("SOCIAL_LINKS", 0.0)

    # =====================================================================
    # Meta / technical
    # =====================================================================
    try:
        md = soup.find("meta", attrs={"name": re.compile(r"^description$", re.I)})
        desc = (md.get("content") or "") if md else ""
        factors["META_DESC_LEN"] = float(len(desc))
        factors["META_DESC_WORDS"] = float(len(desc.split()))
        factors["META_DESC_VARS"] = float(vset.count(desc))
    except Exception:
        factors["META_DESC_LEN"] = 0.0
        factors["META_DESC_WORDS"] = 0.0
        factors["META_DESC_VARS"] = 0.0

    try:
        mk = soup.find("meta", attrs={"name": re.compile(r"^keywords$", re.I)})
        kw = (mk.get("content") or "") if mk else ""
        terms = [t for t in re.split(r",", kw) if t.strip()]
        factors["META_KEYWORDS"] = float(len(terms))
    except Exception:
        factors["META_KEYWORDS"] = 0.0

    try:
        factors["HAS_DOCTYPE"] = (
            1.0 if html.lstrip().lower().startswith("<!doctype") else 0.0
        )
    except Exception:
        factors["HAS_DOCTYPE"] = 0.0

    try:
        factors["HAS_FORM"] = 1.0 if soup.find("form") else 0.0
    except Exception:
        factors["HAS_FORM"] = 0.0

    try:
        has_video = bool(soup.find("video"))
        if not has_video:
            for ifr in soup.find_all("iframe", src=True):
                src = (ifr.get("src") or "").lower()
                if any(h in src for h in _VIDEO_HOSTS):
                    has_video = True
                    break
        factors["HAS_VIDEO"] = 1.0 if has_video else 0.0
    except Exception:
        factors["HAS_VIDEO"] = 0.0

    try:
        has_privacy = False
        for a in soup.find_all("a", href=True):
            href = (a.get("href") or "").lower()
            txt = a.get_text(" ", strip=True).lower()
            if "privacy" in href or "privacy" in txt:
                has_privacy = True
                break
        factors["HAS_PRIVACY"] = 1.0 if has_privacy else 0.0
    except Exception:
        factors["HAS_PRIVACY"] = 0.0

    try:
        has_email = "mailto:" in html.lower() or bool(_EMAIL_RE.search(html))
        factors["HAS_EMAIL"] = 1.0 if has_email else 0.0
    except Exception:
        factors["HAS_EMAIL"] = 0.0

    try:
        factors["LOAD_MS"] = float(load_ms) if load_ms is not None else 0.0
    except Exception:
        factors["LOAD_MS"] = 0.0

    # =====================================================================
    # Schema (JSON-LD + microdata)
    # =====================================================================
    try:
        factors.update(schema_factors(soup, html))
    except Exception:
        for k in ("HAS_JSONLD", "SCHEMA_TYPES", "USES_ORG_LOCALBIZ", "USES_PRODUCT_OFFER",
                  "USES_AGG_RATING", "USES_FAQ", "USES_BREADCRUMB", "CLAIMED_BRANDS"):
            factors.setdefault(k, 0.0)

    # ---- final: ensure all values are floats ---------------------------------
    clean: dict[str, float] = {}
    for k, v in factors.items():
        try:
            clean[k] = float(v)
        except Exception:
            clean[k] = 0.0

    return clean, clean_body_text
