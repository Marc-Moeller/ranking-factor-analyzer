"""Runtime configuration. All secrets come from the environment / .env.

Nothing is hardcoded in the package — a missing key fails loudly at the point of
use, not silently. `get_settings()` is cached so the whole process shares one
config object.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env from project root (parent of this package) if present.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Live SERP (the configured SERP provider)
    serp_api_url: str = ""
    serp_api_key: str = ""

    # DataForSEO (historical SERP for compare mode)
    dataforseo_login: str = ""
    dataforseo_password: str = ""

    # LLM (OpenAI-compatible chat-completions)
    llm_api_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""

    # Entity/EAV extraction model. None -> inherit the primary ``llm_model``.
    # Set ENTITY_MODEL to pin a specific model the LLM endpoint actually serves.
    entity_model: str | None = None

    # Domain authority (optional)
    authority_api_url: str = ""
    authority_api_key: str = ""

    # Backlink provider endpoints — JSON list of {"base","key"} in failover order.
    # Empty -> off-page enrichment degrades gracefully (engine still runs).
    backlink_api_endpoints: str = ""

    # Zyte scraping fallback (hard/anti-bot pages)
    zyte_api_url: str = "https://api.zyte.com"
    zyte_api_key: str = ""

    # Vision LLM (screenshot analysis) — optional multimodal model id.
    llm_vision_model: str = ""
    ranklens_vision_top_n: int = 3

    # PostgreSQL — REQUIRED. RankLens persists runs/users/sessions here. The app
    # builds its DSN from the parts below so it shares the SAME POSTGRES_PASSWORD
    # the postgres service initialises with (single source of truth — the two
    # cannot drift). A full DATABASE_URL, if set, overrides the parts.
    database_url: str = ""
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_user: str = "ranklens"
    postgres_db: str = "ranklens"
    postgres_password: str = ""

    # API auth — shared key gate on the paid write endpoints (/analyze, /compare).
    # Empty -> endpoints are OPEN (local dev). Set in the deploy env to require it.
    # This stays valid as a programmatic / admin credential even with user accounts on.
    ranklens_api_key: str = ""

    # User accounts (cookie sessions). Signup is closed by default; set
    # RANKLENS_ALLOW_SIGNUP=true only when new registrations are intended.
    # Sessions last ranklens_session_days. Cookies are Secure (served over HTTPS
    # via Traefik).
    ranklens_allow_signup: bool = False
    ranklens_session_days: int = 30

    # CrUX (Chrome UX Report) — real Chrome field data per origin. Free Google
    # API key; empty -> the access gate simply skips CrUX factors.
    crux_api_key: str = ""

    # Embeddings — optional OpenAI-compatible /embeddings endpoint for the
    # semantic gate. Empty url/key -> deterministic TF-IDF fallback (no API).
    embeddings_api_url: str = ""
    embeddings_api_key: str = ""
    embeddings_model: str = "text-embedding-3-small"

    # Funnel layers — page caps (cost control) for the LLM-judged gates.
    ranklens_quality_top_n: int = 10       # effort/helpfulness judged on top-N + target
    ranklens_engagement_top_n: int = 10    # click panel over top-N + target

    # Engine tuning
    ranklens_max_pages: int = 20
    ranklens_fetch_concurrency: int = 12
    ranklens_default_country: str = "us"
    ranklens_data_dir: str = "./data"
    # Entity layer — analyze entities on at most the top-N fetched pages (cost cap),
    # and run at most this many entity-extraction LLM calls concurrently.
    ranklens_entity_top_n: int = 20
    ranklens_entity_concurrency: int = 6

    @property
    def resolved_dsn(self) -> str:
        """Effective PostgreSQL DSN. Prefer an explicit DATABASE_URL; otherwise
        build it from the postgres_* parts (sharing POSTGRES_PASSWORD with the DB)."""
        if self.database_url:
            return self.database_url
        # Omit the password section when empty so the DSN is still valid against a
        # trust-auth Postgres (POSTGRES_HOST_AUTH_METHOD=trust) that takes no password.
        auth = self.postgres_user
        if self.postgres_password:
            auth = f"{self.postgres_user}:{self.postgres_password}"
        return (f"postgresql://{auth}"
                f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}")

    @property
    def data_dir(self) -> Path:
        p = (_PROJECT_ROOT / self.ranklens_data_dir).resolve()
        p.mkdir(parents=True, exist_ok=True)
        (p / "reports").mkdir(exist_ok=True)
        return p


# AnalyzeRequest BYOK fields that map 1:1 onto Settings. Kept here so config
# does not import models (models already depends on nothing in this module).
_BYOK_SETTING_FIELDS: tuple[str, ...] = (
    "llm_api_url",
    "llm_api_key",
    "llm_model",
    "dataforseo_login",
    "dataforseo_password",
    "crux_api_key",
)


@lru_cache
def get_settings() -> Settings:
    return Settings()


def settings_for_analyze(request: object, base: Settings | None = None) -> Settings:
    """Return a per-run Settings COPY with optional BYOK overrides applied.

    ``None`` on the request means "keep the process default". A provided value
    (including empty string) overrides the corresponding Settings field for
    this run only — the process-wide singleton is never mutated.
    """
    base = base or get_settings()
    updates: dict[str, object] = {}
    for name in _BYOK_SETTING_FIELDS:
        value = getattr(request, name, None)
        if value is not None:
            updates[name] = value
    return base.model_copy(update=updates) if updates else base.model_copy()
