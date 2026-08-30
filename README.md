# Ranking Factor Analyzer

Scrape a Google SERP, pull ~113 measurable factors off every page that ranks, correlate each factor
against position, and get back a prioritised list of what to change on your own page.

It answers one question: **for this keyword, what do the pages above me actually have in common,
and which of those things am I missing?**

```
  keyword ──► SERP (top 10-25) ──► fetch every ranking page
                                          │
                                          ▼
                          113 factors × N pages  ──► correlate vs position
                                          │                    │
                          your page graded on the same axes    │
                                          │                    ▼
                                          └──────────►  prioritised roadmap
                                                        + writing brief
```

## What it is not

It does not tell you Google's ranking algorithm. Correlation on a 10-25 page sample is a weak
instrument, and plenty of what correlates with rank is *downstream* of ranking rather than a cause
of it. Long pages rank because good pages are often long, not because length is a lever.

What it is good for: finding the factors where you are an outlier against everyone who outranks you.
Those are worth a look. Treat the output as a list of hypotheses ordered by how unusual you are, not
as a list of instructions.

**New here?** [`SETUP.md`](SETUP.md) is the full install and configuration guide — which API keys
you actually need, in what order they earn their cost, and what each one unlocks. It is written
so you can hand it straight to a coding agent.

## Quickstart

```bash
git clone https://github.com/Marc-Moeller/ranking-factor-analyzer
cd ranking-factor-analyzer
pip install -r requirements.txt
cp .env.example .env          # then edit it, see "Keys" below
python -m ranklens.cli analyze "cordless impact driver" --url https://example.com/drivers
```

Or with Docker:

```bash
docker compose up -d          # app on :4711, Postgres alongside
curl localhost:4711/health
```

## Keys: what you need, and what you can skip

**The zero-key run works.** With no credentials at all and every optional layer off, you still get
the deterministic core: 63 HTML factors, 8 SERP factors, 8 trust factors, 4 corpus factors, the
correlation table and the writing brief. You need a SERP source to get the pages in the first place.

```bash
python -m ranklens.cli analyze "your keyword" \
  --no-funnel --no-entities --no-topical
```

Everything below is additive.

| You want | Set | Cost |
|---|---|---|
| SERP results (required) | `DATAFORSEO_LOGIN` + `DATAFORSEO_PASSWORD` | ~$0.002 per keyword |
| Entity extraction, topical authority, the ranking funnel, the written narrative | `LLM_API_URL` + `LLM_API_KEY` + `LLM_MODEL`, any OpenAI-compatible endpoint | ~$0.12 per run on a cheap model |
| Real Core Web Vitals from Chrome field data | `CRUX_API_KEY` | free from Google |
| Better semantic factors than TF-IDF | `EMBEDDINGS_API_URL` + `EMBEDDINGS_API_KEY` | fractions of a cent |
| Pages that block plain HTTP fetches | `ZYTE_API_KEY` | ~$0.0001 per page, more for browser rendering |
| Backlink and brand panels | `BACKLINK_API_ENDPOINTS` (see below) | depends on your provider |
| Domain authority and traffic | `AUTHORITY_API_URL` + `AUTHORITY_API_KEY` | depends on your provider |

`LLM_API_URL` accepts anything that speaks the OpenAI chat-completions shape — OpenRouter, OpenAI,
Together, a local Ollama, vLLM. OpenRouter is the cheapest way to start: one key, a few dollars,
every model.

Backlink and brand analysis expect a Semrush-compatible JSON array in `BACKLINK_API_ENDPOINTS`. There is no
free path for this one. Leave it empty and those two panels are skipped; nothing else changes.

## How a run works

`POST /analyze` returns a `run_id` immediately and the work happens in the background.
`GET /runs/{id}` polls it. `GET /runs/{id}/report` renders HTML, `GET /runs/{id}/brief.md` gives you
the writing brief as markdown.

Runs are stored in Postgres. Set `DATABASE_URL`, or the individual `POSTGRES_*` variables.

The `/analyze` request also accepts per-run credential overrides (`llm_api_key`, `dataforseo_login`,
and friends), so a multi-tenant deployment can let each user bring their own keys instead of putting
them all in the environment. Those values are stripped before the run is persisted — they never
reach the database.

## Auth

Set `RANKLENS_API_KEY` and every write endpoint requires an `X-API-Key` header matching it.

⚠️ **Leave it empty and writes are open to anyone who can reach the port.** That is convenient on
localhost and dangerous anywhere else. There is also an optional cookie-session layer with
email+password accounts; signup is closed unless you set `RANKLENS_ALLOW_SIGNUP=true`.

## The factors

113 of them, defined in `ranklens/factors_registry.py`:

| Source | Count | Needs |
|---|---|---|
| `html` | 63 | nothing, the fetched page |
| `backlinks` | 9 | `BACKLINK_API_ENDPOINTS` |
| `entity` | 9 | an LLM key |
| `serp` | 8 | the SERP itself |
| `trust` | 8 | nothing |
| `corpus` | 4 | nothing |
| `semantic` | 4 | LLM or embeddings, falls back to TF-IDF |
| `crux` | 4 | `CRUX_API_KEY` |
| `quality` | 2 | an LLM key |
| `engagement` | 2 | an LLM key |

Adding one is a single `FactorDef` — see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Licence

AGPL-3.0, with commercial licences available. Read [`LICENSING.md`](LICENSING.md) before you build a
product on it — the network clause in section 13 is the part that catches people out.
