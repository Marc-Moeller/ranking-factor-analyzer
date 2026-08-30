"""Acceptance proofs for include_funnel + no-key degrade, and BYOK strip.

No live paid API calls. SERP and page fetch are stubbed. Settings are an
in-memory empty/override copy so process env keys cannot leak into the run.
"""
from __future__ import annotations

import asyncio
import json
import sys
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ranklens.cli import build_parser
from ranklens.clients.fetch import FetchedPage
from ranklens.clients.llm import LLM_NO_KEY, chat, llm_unavailable
from ranklens.config import Settings, settings_for_analyze
from ranklens.models import (
    AnalyzeReport,
    AnalyzeRequest,
    BYOK_REQUEST_FIELDS,
    Run,
    RunKind,
    RunStatus,
    Serp,
    SerpItem,
)
from ranklens.pipeline import run_analyze
from ranklens.store import save_run


def _empty_settings(**overrides) -> Settings:
    """Build Settings that ignore process env / leftover dotenv.

    ``Settings(**kwargs)`` still lets env vars win under pydantic-settings, so
    a developer machine with real keys would make live calls. ``model_construct``
    bypasses env loading.
    """
    dumped = {name: info.default for name, info in Settings.model_fields.items()}
    dumped.update(
        llm_api_url="https://example.invalid/v1",
        llm_api_key="",
        llm_model="",
        serp_api_key="",
        dataforseo_login="",
        dataforseo_password="",
        crux_api_key="",
        embeddings_api_url="",
        embeddings_api_key="",
        zyte_api_key="",
        backlink_api_endpoints="",
        authority_api_key="",
        ranklens_api_key="",
        postgres_password="",
        database_url="",
    )
    dumped.update(overrides)
    return Settings.model_construct(**dumped)


_HTML = """<!doctype html>
<html><head>
<title>Best running shoes 2026</title>
<meta name="description" content="A practical guide to the best running shoes.">
</head><body>
<h1>Best running shoes</h1>
<p>Cushioned running shoes for marathon training, daily miles, and speed work.
Look at stack height, drop, and durability before you buy.</p>
<h2>How to choose</h2>
<p>Match the shoe to your gait, terrain, and weekly mileage. Neutral runners
often prefer a daily trainer; stability shoes help overpronation.</p>
</body></html>
"""


def _stub_serp() -> Serp:
    items = [
        SerpItem(rank=i, url=f"https://p{i}.example/shoes", domain=f"p{i}.example",
                 title=f"Best running shoes #{i}", snippet="guide to running shoes")
        for i in range(1, 8)
    ]
    return Serp(keyword="best running shoes", country="us", language="en",
                source="stub", items=items)


async def _fake_fetch_serp(*_a, **_k):
    return _stub_serp()


async def _fake_live_serp(*_a, **_k):
    return _stub_serp(), 0.0


async def _fake_fetch_pages(urls, concurrency=12, settings=None):
    return {
        u: FetchedPage(url=u, final_url=u, status_code=200, html=_HTML,
                       load_ms=5.0, ok=True, error=None)
        for u in urls
    }


class ForbiddenHTTP:
    """Any live HTTP in these tests is a failure."""

    def __init__(self, *a, **k):
        raise AssertionError("live HTTP is forbidden in this test")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


def _no_network():
    return (
        patch("ranklens.pipeline.fetch_serp", _fake_fetch_serp),
        patch("ranklens.pipeline.live_serp_advanced", _fake_live_serp),
        patch("ranklens.pipeline.fetch_pages", _fake_fetch_pages),
        patch("ranklens.clients.llm.httpx.AsyncClient", ForbiddenHTTP),
        patch("ranklens.clients.serp.httpx.AsyncClient", ForbiddenHTTP),
        patch("ranklens.clients.dataforseo.httpx.AsyncClient", ForbiddenHTTP),
        patch("ranklens.clients.crux.httpx.AsyncClient", ForbiddenHTTP),
        patch("ranklens.clients.embeddings.httpx.AsyncClient", ForbiddenHTTP),
    )


class TestDegradedAnalyze(unittest.TestCase):
    def test_chat_no_key_returns_sentinel_not_raise(self):
        settings = _empty_settings()

        async def _run():
            return await chat(
                [{"role": "user", "content": "hello"}],
                settings=settings,
            )

        text = asyncio.run(_run())
        self.assertEqual(text, LLM_NO_KEY)
        self.assertTrue(llm_unavailable(text))

    def test_cli_exposes_no_funnel(self):
        args = build_parser().parse_args(["analyze", "best running shoes", "--no-funnel"])
        self.assertTrue(args.no_funnel)

    def test_analyze_deterministic_layers_only_does_not_raise(self):
        empty = _empty_settings()
        req = AnalyzeRequest(
            keyword="best running shoes",
            target_url="https://p1.example/shoes",
            country="us",
            language="en",
            max_pages=7,
            include_backlinks=False,
            include_brand=False,
            include_entities=False,
            include_topical=False,
            include_funnel=False,
            serp_source="serpmaster",
        )

        with ExitStack() as stack:
            stack.enter_context(patch("ranklens.pipeline.settings_for_analyze", return_value=empty))
            for cm in _no_network():
                stack.enter_context(cm)
            report = asyncio.run(run_analyze(req, with_ai=True))

        self.assertIsInstance(report, AnalyzeReport)
        self.assertEqual(report.request.keyword, "best running shoes")
        self.assertFalse(report.request.include_funnel)
        self.assertIsNone(report.semantic)
        self.assertIsNone(report.intent_fit)
        self.assertIsNone(report.quality)
        self.assertIsNone(report.engagement)
        self.assertIsNone(report.funnel)
        self.assertEqual(report.competitor_cards, [])
        self.assertIsNone(report.offpage)
        self.assertIsNone(report.brand)
        self.assertIsNone(report.entity_table)
        self.assertIsNone(report.topical)
        self.assertGreaterEqual(report.pages_fetched_ok, 1)
        self.assertGreaterEqual(len(report.page_factors), 1)
        self.assertIsInstance(report.correlations, list)
        # Narrative skipped the sentinel instead of writing it as content.
        if report.ai_narrative:
            self.assertNotIn("AI unavailable", report.ai_narrative)
            self.assertNotIn(LLM_NO_KEY, report.ai_narrative)
        dumped = report.model_dump(mode="json")
        self.assertIn("funnel", dumped)
        self.assertIsNone(dumped["funnel"])


class TestByok(unittest.TestCase):
    def test_per_run_settings_copy_uses_request_key(self):
        base = _empty_settings(llm_api_key="ENV-KEY-SHOULD-NOT-WIN")
        req = AnalyzeRequest(
            keyword="shoes",
            llm_api_key="BYOK-SECRET-KEY",
            llm_api_url="https://byok.example/v1",
            llm_model="byok-model",
        )
        copied = settings_for_analyze(req, base)
        self.assertEqual(copied.llm_api_key, "BYOK-SECRET-KEY")
        self.assertEqual(copied.llm_api_url, "https://byok.example/v1")
        self.assertEqual(copied.llm_model, "byok-model")
        self.assertEqual(base.llm_api_key, "ENV-KEY-SHOULD-NOT-WIN")
        self.assertIsNot(copied, base)

        seen: dict[str, str] = {}

        async def fake_post(url, api_key, model, messages, max_tokens, temperature):
            seen["api_key"] = api_key
            seen["url"] = url
            seen["model"] = model
            return "ok-from-byok"

        async def _run():
            with patch("ranklens.clients.llm._post_completion", fake_post):
                return await chat(
                    [{"role": "user", "content": "hi"}],
                    settings=copied,
                )

        text = asyncio.run(_run())
        self.assertEqual(text, "ok-from-byok")
        self.assertEqual(seen["api_key"], "BYOK-SECRET-KEY")
        self.assertIn("byok.example", seen["url"])
        self.assertEqual(seen["model"], "byok-model")

    def test_run_analyze_uses_byok_key_and_blanks_request(self):
        secret = "sk-BYOK-PIPELINE-SECRET"
        empty = _empty_settings()
        req = AnalyzeRequest(
            keyword="best running shoes",
            target_url="https://p1.example/shoes",
            include_backlinks=False,
            include_brand=False,
            include_entities=False,
            include_topical=False,
            include_funnel=False,
            serp_source="serpmaster",
            llm_api_key=secret,
            llm_api_url="https://byok.example/v1",
            llm_model="byok-model",
        )
        seen: dict[str, str] = {}

        async def fake_post(url, api_key, model, messages, max_tokens, temperature):
            seen["api_key"] = api_key
            seen["url"] = url
            seen["model"] = model
            return "narrative-from-byok"

        with ExitStack() as stack:
            stack.enter_context(patch("ranklens.config.get_settings", return_value=empty))
            stack.enter_context(patch("ranklens.pipeline.get_settings", return_value=empty))
            stack.enter_context(patch("ranklens.clients.llm._post_completion", fake_post))
            for cm in _no_network():
                stack.enter_context(cm)
            report = asyncio.run(run_analyze(req, with_ai=True))

        self.assertEqual(seen.get("api_key"), secret)
        self.assertIn("byok.example", seen.get("url", ""))
        self.assertEqual(seen.get("model"), "byok-model")
        self.assertIsNone(req.llm_api_key)
        self.assertIsNone(report.request.llm_api_key)
        blob = report.model_dump_json()
        self.assertNotIn(secret, blob)
        self.assertEqual(report.ai_narrative, "narrative-from-byok")

    def test_save_run_strips_credentials_from_persisted_row(self):
        secret = "sk-BYOK-PLAINTEXT-SECRET"
        run = Run(
            id="abc123def456",
            kind=RunKind.analyze,
            status=RunStatus.done,
            label='analyze: "shoes"',
            request={
                "keyword": "shoes",
                "llm_api_key": secret,
                "llm_api_url": "https://byok.example/v1",
                "llm_model": "secret-model",
                "dataforseo_login": "dfs-user",
                "dataforseo_password": "dfs-pass-xyz",
                "crux_api_key": "crux-secret-key",
            },
            result={
                "request": {
                    "keyword": "shoes",
                    "llm_api_key": secret,
                    "dataforseo_password": "dfs-pass-xyz",
                },
                "serp": {"keyword": "shoes"},
            },
            error=f"provider failed for {secret}",
        )
        written: dict = {}

        class FakeCursor:
            def execute(self, sql, params):
                written.update(params)

        @contextmanager
        def fake_conn():
            yield FakeCursor()

        with patch("ranklens.store.ensure_init", lambda: None), \
             patch("ranklens.store._conn", fake_conn):
            save_run(run)

        blob = json.dumps(written)
        for needle in (secret, "dfs-pass-xyz", "crux-secret-key", "dfs-user"):
            self.assertNotIn(needle, blob)
        parsed_request = json.loads(written["request"])
        parsed_result = json.loads(written["result"])
        for field in BYOK_REQUEST_FIELDS:
            self.assertIsNone(parsed_request.get(field))
            self.assertIsNone(run.request.get(field))
        self.assertIsNone(parsed_result["request"].get("llm_api_key"))
        self.assertIsNone(parsed_result["request"].get("dataforseo_password"))
        self.assertNotIn(secret, written.get("error") or "")
        self.assertIn("[redacted]", written.get("error") or "")
        leftover = json.dumps(run.model_dump(mode="json"))
        self.assertNotIn(secret, leftover)


if __name__ == "__main__":
    unittest.main()
