"""Regression tests for Danish measurements, SERP metadata and URL identity."""
import asyncio
import unicodedata
from unittest.mock import patch

import numpy as np
import pytest

from ranklens.analyze.semantic import _tokens, _tfidf_matrix
from ranklens.analyze.topical_authority import _fallback_expand, _term_pattern
from ranklens.cli import build_parser, cmd_analyze
from ranklens.clients.dataforseo import _dedup_organic, _location_code, live_serp_advanced
from ranklens.clients.fetch import FetchedPage
from ranklens.clients.serp import fetch_serp
from ranklens.config import Settings
from ranklens.extract.corpus import build_corpus
from ranklens.extract.factors import extract_html_factors
from ranklens.keywords import build_variation_set, word_count
from ranklens.models import AnalyzeRequest, CompareRequest, Serp, SerpItem
from ranklens.pipeline import run_analyze
from ranklens.report.html import render_analyze


def empty_settings():
    return Settings.model_construct(**{k: f.default for k, f in Settings.model_fields.items()})


@pytest.mark.parametrize("country", ["dk", "DK", " dk "])
def test_denmark_location(country):
    assert _location_code(country) == 2208


@pytest.mark.parametrize("country", ["xx", "", "dkk"])
def test_unknown_market_fails_closed(country):
    with pytest.raises(ValueError, match="Unsupported"):
        _location_code(country)


def test_unicode_keyword_counts_and_normalization():
    vset = build_variation_set("trægulv og gulvlægger", "da")
    assert "og" not in vset.variations
    assert "trægulv" in vset.variations
    assert vset.count_exact("TRÆGULV OG GULVLÆGGER") == 1
    assert build_variation_set("blå", "da").count(unicodedata.normalize("NFD", "BLÅ")) == 1
    assert word_count("Æble øl blå gulvlægger") == 4
    assert build_variation_set("gulv", "da").first_match_offset("æøå gulv") == 7
    assert vset.count("gulvlæggeren") == 0


def test_english_defaults_preserved():
    assert build_variation_set("the running shoes").variations == build_variation_set("the running shoes", "en").variations
    assert "the" not in build_variation_set("the running shoes").variations
    assert AnalyzeRequest(keyword="test").language == "en"


@pytest.mark.parametrize("model,extra", [(AnalyzeRequest, {}), (CompareRequest, {"update_date": "2026-10-01"})])
def test_market_default_and_explicit_override(model, extra):
    assert model(keyword="test", country="dk", **extra).language == "da"
    assert model(keyword="test", country="dk", language="en", **extra).language == "en"


def test_corpus_and_semantic_keep_danish_terms():
    vset = build_variation_set("gulvlægger pris", "da")
    corpus = build_corpus(["gulvlægger og trægulv", "gulvlægger og parket", "gulvlægger trægulv", "gulvlægger laminat"], vset)
    assert "og" not in corpus.lsi_terms
    assert "trægulv" in corpus.lsi_terms
    assert _tokens("og på trægulv gulvlægger", "da") == ["trægulv", "gulvlægger"]
    matrix = _tfidf_matrix(["gulvlægger trægulv", "gulvlægger trægulv", "æbler øl"], "da")
    assert np.dot(matrix[0], matrix[1]) > .99
    assert np.dot(matrix[0], matrix[2]) == 0


def test_unknown_language_does_not_use_english_stopwords():
    assert "the" in build_variation_set("the test", "zz").variations


def test_topical_matching_preserves_danish():
    assert _fallback_expand("pris på trægulv", "da")["core"] == "pris trægulv"
    assert _term_pattern("gulvlægger", "da").search("billig-gulvlægger")


def test_clean_body_and_danish_question_heading():
    html = """<html><head><title>Ikke brødtekst</title></head><body>
    <nav>menu gulvlægger</nav><div role="navigation">menu to</div>
    <h1>Hvordan vælger du trægulv</h1><p>Blå trægulve holder længe.</p>
    <footer>Kontakt gulvlægger</footer></body></html>"""
    factors, body = extract_html_factors(html, "https://example.dk", "example.dk", build_variation_set("trægulv", "da"), 2)
    assert body == "Hvordan vælger du trægulv Blå trægulve holder længe."
    assert factors["WORD_COUNT"] == 8
    assert factors["QUESTIONS_H1"] == 1
    assert "FLESCH_GRADE" not in factors and "FLESCH_EASE" not in factors
    assert factors["UNABRIDGED_WORD_COUNT"] > factors["WORD_COUNT"]


def test_dataforseo_snippet_preserved():
    items = _dedup_organic([{"type": "organic", "domain": "www.example.dk", "url": "https://example.dk/gulve", "rank_group": 1, "description": "Trægulv og gulvlægger", "breadcrumb": "example.dk > gulve"}], 20)
    assert items[0].snippet == "Trægulv og gulvlægger"
    assert items[0].displayed_url == "example.dk > gulve"


def test_danish_market_reaches_each_serp_provider():
    import httpx
    bodies = []
    def handler(request):
        import json
        bodies.append(json.loads(request.content))
        if request.url.host == "api.dataforseo.com":
            return httpx.Response(200, json={"tasks": [{"result": [{"items": []}]}]})
        return httpx.Response(200, json={"results": []})
    real_client = httpx.AsyncClient
    def client(**kwargs):
        return real_client(transport=httpx.MockTransport(handler), **kwargs)
    settings = empty_settings().model_copy(update={"serp_api_url": "https://serp.example", "serp_api_key": "test", "dataforseo_login": "test", "dataforseo_password": "test"})
    with patch("httpx.AsyncClient", client):
        asyncio.run(live_serp_advanced("gulvlægger", country="dk", language="da", settings=settings))
        serp = asyncio.run(fetch_serp("gulvlægger", country="dk", language="da", settings=settings))
    assert bodies[0][0]["location_code"] == 2208
    assert bodies[0][0]["language_code"] == "da"
    assert bodies[1]["hl"] == "da"
    assert serp.language == "da"


def snapshot():
    return Serp(keyword="gulvlægger pris", country="dk", language="da", source="test-export", items=[SerpItem(rank=i, url=f"https://p{i}.dk/gulve", domain=f"p{i}.dk", snippet="gulvlægger pris") for i in range(1, 8)])


async def fake_pages(urls, **kwargs):
    return {url: FetchedPage(url=url, final_url=url, status_code=200, html="<html><body><h1>Hvad koster en gulvlægger</h1><p>En gulvlægger lægger blå trægulve.</p></body></html>", load_ms=1, ok=True, error=None) for url in urls}


def request(**kwargs):
    return AnalyzeRequest(keyword="gulvlægger pris", country="dk", include_entities=False, include_topical=False, include_funnel=False, **kwargs)


def test_pipeline_saved_serp_url_identity_and_render():
    serp = snapshot()
    with patch("ranklens.pipeline.settings_for_analyze", return_value=empty_settings()), patch("ranklens.pipeline.fetch_pages", fake_pages), patch("ranklens.pipeline.fetch_serp", side_effect=AssertionError("live SERP forbidden")), patch("ranklens.pipeline.live_serp_advanced", side_effect=AssertionError("live SERP forbidden")):
        report = asyncio.run(run_analyze(request(target_url="https://p1.dk/different-page"), with_ai=False, provided_serp=serp))
        ranked = asyncio.run(run_analyze(request(target_url="https://p1.dk/gulve"), with_ai=False, provided_serp=serp))
    assert report.pages_fetched_ok == 7
    assert report.target.found_in_serp is False
    assert report.target.serp_rank is None
    assert ranked.target.serp_rank == 1
    assert report.page_factors[0].factors["QUESTIONS_H1"] == 1
    assert "gulvlægger pris" in render_analyze(report)
    assert len(serp.items) == 7  # supplied snapshot is not mutated


@pytest.mark.parametrize("field,value", [("country", "us"), ("language", "en"), ("keyword", "different"), ("items", [])])
def test_saved_serp_rejects_wrong_market_or_empty(field, value):
    with patch("ranklens.pipeline.settings_for_analyze", return_value=empty_settings()):
        with pytest.raises(ValueError, match="Saved SERP"):
            asyncio.run(run_analyze(request(), with_ai=False, provided_serp=snapshot().model_copy(update={field: value})))


def test_cli_saved_serp_no_database(tmp_path):
    path = tmp_path / "serp.json"
    path.write_text(snapshot().model_dump_json(), encoding="utf-8")
    settings = empty_settings().model_copy(update={"ranklens_data_dir": str(tmp_path / "output")})
    args = build_parser().parse_args(["analyze", "gulvlægger pris", "--country", "dk", "--serp-file", str(path), "--no-store", "--no-topical", "--no-entities", "--no-funnel", "--no-ai"])
    with patch("ranklens.cli.get_settings", return_value=settings), patch("ranklens.pipeline.settings_for_analyze", return_value=settings), patch("ranklens.pipeline.fetch_pages", fake_pages), patch("ranklens.cli.save_run", side_effect=AssertionError("database forbidden")):
        cmd_analyze(args)
    assert len(list((tmp_path / "output" / "reports").glob("*.html"))) == 1
