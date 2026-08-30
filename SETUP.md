# Setup

This document is written to be handed to a coding agent. Give it this file and it should be
able to stand the project up, decide which API keys you actually need, and tell you what each
one costs. A human can follow it just as well.

Nothing here ships with credentials. Every key is yours, set in your own `.env`, and stays on
your machine.

---

## The engine needs one key to do anything, and the rest are optional

There is exactly one hard requirement: **a source of SERP results.** Without it the tool has
no pages to analyse. Everything else is additive, and every optional layer degrades to
"skipped" rather than crashing when its key is blank.

```
  REQUIRED                          OPTIONAL, each independent
  ────────                          ──────────────────────────
  SERP source ──► pages ──┬──► 63 HTML factors      (no key)
                          ├──► 8 SERP factors       (no key)
                          ├──► 8 trust factors      (no key)
                          ├──► 4 corpus factors     (no key)
                          ├──► entities/quality/    LLM_API_KEY
                          │    topical/funnel
                          ├──► Core Web Vitals      CRUX_API_KEY
                          ├──► semantic (better)    EMBEDDINGS_API_KEY
                          ├──► anti-bot page fetch  ZYTE_API_KEY
                          ├──► backlinks + brand    BACKLINK_API_ENDPOINTS
                          └──► domain authority     AUTHORITY_API_KEY
```

A run with only a SERP key still produces the full correlation table, the factor grades for
your page, and the writing brief. It just has no AI narrative and no off-page panels.

---

## Which keys to get, in the order they earn their cost

| Priority | Variable(s) | What it unlocks | Where to get it | Rough cost |
|---|---|---|---|---|
| **1. Required** | `DATAFORSEO_LOGIN` + `DATAFORSEO_PASSWORD` | The SERP itself. Nothing runs without a SERP source. | [dataforseo.com](https://dataforseo.com) | ~$0.002 per keyword |
| **2. Do this one** | `LLM_API_URL` + `LLM_API_KEY` + `LLM_MODEL` | Entity extraction, topical authority, the ranking funnel, the written narrative. This is the single biggest jump in output quality. | [openrouter.ai](https://openrouter.ai) is the easiest: one key, every model | ~$0.12 per run on a cheap model |
| **3. Free, take it** | `CRUX_API_KEY` | Real Chrome field data for Core Web Vitals instead of nothing. | [Google CrUX API](https://developers.google.com/web/tools/chrome-user-experience-report/api) | free |
| 4. Nice to have | `EMBEDDINGS_API_URL` + `EMBEDDINGS_API_KEY` | Real embeddings for the semantic factors. Falls back to TF-IDF when unset, which is worse but works. | Any OpenAI-compatible `/embeddings` endpoint | fractions of a cent |
| 5. Situational | `ZYTE_API_KEY` | Fetching pages that block plain HTTP requests. Skip it until you hit sites that block you. | [zyte.com](https://www.zyte.com) | ~$0.0001/page, more with browser rendering |
| 6. Bring your own | `BACKLINK_API_ENDPOINTS` | The backlink and brand panels (9 factors). | No bundled provider — see below | depends on yours |
| 7. Bring your own | `AUTHORITY_API_URL` + `AUTHORITY_API_KEY` | Domain authority and traffic estimates. | No bundled provider — see below | depends on yours |

**Start with rows 1 and 2.** That is roughly $0.12 a run and gives you almost everything.

### `SERP_API_URL` / `SERP_API_KEY` is an alternative to DataForSEO, not an addition

If you run your own SERP scraper, point `SERP_API_URL` at it and leave DataForSEO blank. The
contract it must speak is documented in `ranklens/clients/serp.py`. Most people should just
use DataForSEO and ignore these two variables.

### Rows 6 and 7 ship with no provider on purpose

The backlink, brand and authority panels talk to HTTP endpoints that you supply. There is no
free or bundled source for this data, and the project does not resell one. The exact request
and response contract each endpoint must satisfy is documented in the module docstrings of
`ranklens/clients/backlinks.py` and `ranklens/clients/authority.py` — an agent can read those
and write an adapter against whatever backlink provider you already pay for.

Leave both blank and those panels are simply absent from the report. Nothing else changes.

---

## Install

```bash
git clone https://github.com/Marc-Moeller/ranking-factor-analyzer
cd ranking-factor-analyzer
cp .env.example .env
```

Now edit `.env`. At minimum set the DataForSEO pair and `POSTGRES_PASSWORD`.

### Docker (recommended — brings its own Postgres)

```bash
docker compose up -d
curl localhost:4711/health
```

The app binds to `127.0.0.1:4711` only. See the security note below before changing that.

### Local Python

You need Python 3.11+ and a reachable PostgreSQL.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m ranklens.migrate                     # create the tables
python -m ranklens.cli analyze "cordless impact driver" --url https://example.com/drivers
```

Set `DATABASE_URL`, or the individual `POSTGRES_*` variables, whichever you prefer. An
explicit `DATABASE_URL` wins.

### Run it with no LLM at all

```bash
python -m ranklens.cli analyze "your keyword" --no-funnel --no-entities --no-topical
```

---

## An empty `RANKLENS_API_KEY` leaves your write endpoints open to anyone who can reach the port

This is the one setting that can genuinely hurt you.

- **Blank** (the default): `/analyze` and `/compare` accept requests from anybody. That is
  fine on localhost, and dangerous the moment the port is exposed — those endpoints spend
  your API credits.
- **Set**: every write endpoint requires a matching `X-API-Key` header.

Set it before you expose the service. `RANKLENS_ALLOW_SIGNUP` is `false` by default and
should stay that way unless you want open registration on the optional cookie-session layer.

---

## Multi-tenant deployments can take each user's keys per request

`POST /analyze` accepts per-run credential overrides in the request body — `llm_api_key`,
`llm_api_url`, `llm_model`, `dataforseo_login`, `dataforseo_password`, `crux_api_key`. When
present they override the environment for that run only; the process-wide config is never
mutated.

**Those values are stripped from the run before it is written to Postgres.** See
`_scrub_run_credentials` in `ranklens/store.py`. A user's key never reaches your database,
your logs, or the stored error text.

This is what makes a "bring your own key" front end possible without holding other people's
credentials.

---

## How a run actually executes

```
  POST /analyze  ──►  returns run_id immediately, work continues in background
                          │
  GET /runs/{id}  ◄───────┘  poll until status is "done" or "error"
                          │
                          ├──►  GET /runs/{id}/report    rendered HTML
                          └──►  GET /runs/{id}/brief.md  the writing brief
```

Runs with the full profile (entities + topical + funnel) commonly take **25 to 35 minutes**.
Set your client's timeout accordingly — a short poll ceiling will time out a run that was
going to succeed. A deterministic run with `--no-funnel --no-entities --no-topical` finishes
in a couple of minutes.

---

## Where to change things

| You want to | Look at |
|---|---|
| Add a ranking factor | `ranklens/factors_registry.py` — one `FactorDef`. See `CONTRIBUTING.md`. |
| Change how a factor is measured | `ranklens/extract/` |
| Change the correlation or scoring | `ranklens/analyze/correlate.py`, `ranklens/analyze/recommend.py` |
| Swap the SERP provider | `ranklens/clients/serp.py` |
| Plug in your backlink provider | `ranklens/clients/backlinks.py` |
| Plug in your authority provider | `ranklens/clients/authority.py` |
| Change the report layout | `ranklens/report/templates/report.html.j2` |
| Change the run orchestration | `ranklens/pipeline.py` |

Config lives in one place: `ranklens/config.py`. Every setting is an environment variable
with the same name uppercased.

---

## Read this before you trust the output

Correlation over a 10 to 25 page sample is a weak instrument, and much of what correlates
with rank is *downstream* of ranking rather than a cause of it. Long pages rank because good
pages tend to be long, not because length is a lever.

What this tool is genuinely good at is finding the factors where **you are an outlier against
everyone who outranks you**. Treat the output as hypotheses ordered by how unusual you are,
not as a list of instructions.

---

## Licence

AGPL-3.0. The section 13 network clause is the part that catches people out: if you run a
modified version as a network service, you must offer your users its source. Read
[`LICENSING.md`](LICENSING.md) before building a product on this. Commercial licences are
available.
