"""RankLens — a Cora replacement.

A SERP-driven on-page correlation engine plus a before/after algorithm-update
comparison tool. The `ranklens` package is a pure-Python engine with no web
dependencies, so it can be driven by the CLI, a FastAPI service, or a future
background task-runner without changes. Pydantic models in `ranklens.models`
are the single contract shared by the engine, the API, and any frontend.
"""

__version__ = "0.1.0"
