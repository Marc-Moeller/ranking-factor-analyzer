"""Smoke tests: the things that must hold before anything else is worth debugging.

These deliberately touch no network and no database. If one of these fails, the
package is broken at import or contract level, not at analysis level.
"""

import pytest

from ranklens import factors_registry
from ranklens.models import AnalyzeRequest


def test_factor_registry_loads():
    factors = factors_registry.REGISTRY
    assert len(factors) > 50, "registry collapsed; expected the full factor set"


def test_factor_ids_are_unique():
    ids = [f.id for f in factors_registry.REGISTRY]
    duplicates = {i for i in ids if ids.count(i) > 1}
    assert not duplicates, f"duplicate factor ids would silently overwrite: {duplicates}"


def test_every_factor_declares_a_source():
    for factor in factors_registry.REGISTRY:
        assert factor.source, f"{factor.id} has no source, so nothing can gate it"


def test_analyze_request_defaults_are_runnable_without_paid_keys():
    """The default request must not silently opt a new user into paid layers."""
    req = AnalyzeRequest(keyword="test keyword")
    assert req.include_backlinks is False
    assert req.include_brand is False


def test_analyze_request_accepts_per_run_credentials():
    """Multi-tenant deployments pass keys per run rather than via the environment."""
    req = AnalyzeRequest(keyword="test keyword", llm_api_key="test-key")
    assert req.llm_api_key == "test-key"


def test_optional_layers_can_all_be_disabled():
    req = AnalyzeRequest(
        keyword="test keyword",
        include_backlinks=False,
        include_brand=False,
        include_entities=False,
        include_topical=False,
        include_funnel=False,
    )
    assert req.include_funnel is False


def test_app_imports_and_exposes_health():
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from api.main import app

    paths = {route.path for route in app.routes}
    assert "/health" in paths
    assert "/analyze" in paths
    del fastapi, TestClient


def test_by_id_index_covers_the_whole_registry():
    """BY_ID is what lookups go through; a gap there silently drops a factor."""
    assert len(factors_registry.BY_ID) == len(factors_registry.REGISTRY)
