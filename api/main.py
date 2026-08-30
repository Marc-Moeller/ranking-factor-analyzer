"""RankLens API — the thin FastAPI wrapper that the SaaS frontend calls.

The engine (`ranklens.pipeline`) does the work; this layer only:
  - authenticates the caller (email/password cookie session, or the legacy
    ``X-API-Key`` for programmatic / admin access),
  - accepts a job request and runs it in a background task,
  - persists the `Run` (owned by the user who started it), and
  - serves the rendered HTML report + JSON result, scoped to that owner.

Swapping BackgroundTasks for Celery/RQ later is a drop-in change — the `Run`
contract and the store stay identical.

Run:  uvicorn api.main:app --port 4711 --reload
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    Form,
    Header,
    HTTPException,
    Request,
)
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response

from ranklens import auth
from ranklens.clients.dataforseo import COUNTRY_LOCATION
from ranklens.config import get_settings
from ranklens.models import (
    AnalyzeReport,
    AnalyzeRequest,
    CompareReport,
    CompareRequest,
    Run,
    RunKind,
    RunStatus,
)
from ranklens.pipeline import run_analyze, run_compare
from ranklens.report.html import render_analyze, render_compare
from ranklens.store import get_run, init_db, list_runs, save_run

app = FastAPI(title="RankLens API", version="0.3.1",
              description="Cora replacement: SERP factor analysis + algorithm-update comparison.")

# Friendly names for the country picker. Keyed by Google ``gl`` code; the picker
# only offers codes that DataForSEO can actually geo-target (COUNTRY_LOCATION),
# so the dropdown can never silently fall back to US.
_COUNTRY_NAMES: dict[str, str] = {
    "us": "United States",
    "au": "Australia",
    "gb": "United Kingdom",
    "ca": "Canada",
    "de": "Germany",
    "fr": "France",
    "in": "India",
    "nz": "New Zealand",
}


def _country_options(selected: str = "us") -> str:
    """Render <option> tags for every supported country, one pre-selected."""
    sel = (selected or "us").lower()
    out = []
    for code in COUNTRY_LOCATION:
        name = _COUNTRY_NAMES.get(code, code.upper())
        mark = " selected" if code == sel else ""
        out.append(f'<option value="{code}"{mark}>{name} ({code.upper()})</option>')
    return "".join(out)


@app.on_event("startup")
def _startup() -> None:
    from ranklens import db, migrate
    from ranklens.store import mark_interrupted_runs
    db.ensure_init()
    migrated = migrate.auto_migrate_from_sqlite()
    if any(migrated.values()):
        print(f"[startup] migrated legacy SQLite -> PostgreSQL: {migrated}")
    interrupted = mark_interrupted_runs()
    if interrupted:
        print(f"[startup] marked {interrupted} interrupted run(s) as error")


# --------------------------------------------------------------------------- #
# Auth helpers
# --------------------------------------------------------------------------- #
def _session_user(request: Request) -> Optional[auth.User]:
    return auth.user_for_session(request.cookies.get(auth.SESSION_COOKIE))


def _api_key_ok(x_api_key: Optional[str]) -> bool:
    """True when the legacy shared key matches, OR no key is configured (open dev)."""
    expected = get_settings().ranklens_api_key
    if not expected:
        return True            # local dev: endpoints open, exactly as before
    return x_api_key == expected


def write_access(request: Request, x_api_key: Optional[str] = Header(default=None)) -> Optional[str]:
    """Gate the paid write endpoints. Returns the owner_id to stamp on the run.

    - logged-in user  -> allowed, owner = user.id
    - valid X-API-Key  -> allowed, owner = None (admin / programmatic run)
    - no key configured (local dev) -> allowed, owner = None
    - otherwise        -> 401
    """
    user = _session_user(request)
    if user:
        return user.id
    if _api_key_ok(x_api_key):
        return None
    raise HTTPException(401, "log in or send a valid X-API-Key")


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        auth.SESSION_COOKIE,
        token,
        max_age=get_settings().ranklens_session_days * 86400,
        httponly=True,
        samesite="lax",
        # Not forcing Secure: Traefik terminates TLS, so the app sees http
        # internally; the edge is always https + CF, so the cookie still rides
        # https in practice while staying usable for local dev.
        secure=False,
        path="/",
    )


# --------------------------------------------------------------------------- #
# Run helpers
# --------------------------------------------------------------------------- #
def _new_run(kind: RunKind, label: str, request: dict, owner_id: Optional[str]) -> Run:
    run = Run(id=uuid.uuid4().hex[:12], kind=kind, status=RunStatus.running,
              label=label, request=request, owner_id=owner_id)
    save_run(run)
    return run


async def _do_analyze(run_id: str, req: AnalyzeRequest) -> None:
    run = get_run(run_id)
    try:
        report = await run_analyze(req, with_ai=True)
        run.status = RunStatus.done
        run.result = report.model_dump(mode="json")
    except Exception as e:  # pragma: no cover
        run.status = RunStatus.error
        run.error = str(e)
    run.finished_at = datetime.now(timezone.utc)
    save_run(run)


async def _do_compare(run_id: str, req: CompareRequest) -> None:
    run = get_run(run_id)
    try:
        report = await run_compare(req, with_ai=True)
        run.status = RunStatus.done
        run.result = report.model_dump(mode="json")
    except Exception as e:  # pragma: no cover
        run.status = RunStatus.error
        run.error = str(e)
    run.finished_at = datetime.now(timezone.utc)
    save_run(run)


def _can_view_run(run: Run, request: Request, x_api_key: Optional[str]) -> bool:
    """A run is viewable by: its owner, an admin (valid key / open dev), or anyone
    when it has no owner (legacy run or admin-created — keeps shared links alive)."""
    if run.owner_id is None:
        return True
    user = _session_user(request)
    if user and user.id == run.owner_id:
        return True
    expected = get_settings().ranklens_api_key
    return bool(expected) and x_api_key == expected


# --------------------------------------------------------------------------- #
# JSON API (programmatic clients keep working via X-API-Key)
# --------------------------------------------------------------------------- #
@app.post("/analyze")
async def analyze(req: AnalyzeRequest, bg: BackgroundTasks, owner_id: Optional[str] = Depends(write_access)):
    run = _new_run(RunKind.analyze, f'analyze: "{req.keyword}"', req.model_dump(mode="json"), owner_id)
    bg.add_task(_do_analyze, run.id, req)
    return {"run_id": run.id, "status": run.status.value, "poll": f"/runs/{run.id}"}


@app.post("/compare")
async def compare(req: CompareRequest, bg: BackgroundTasks, owner_id: Optional[str] = Depends(write_access)):
    run = _new_run(RunKind.compare, f'compare: "{req.keyword}"', req.model_dump(mode="json"), owner_id)
    bg.add_task(_do_compare, run.id, req)
    return {"run_id": run.id, "status": run.status.value, "poll": f"/runs/{run.id}"}


@app.get("/runs")
async def runs(request: Request, limit: int = 50, x_api_key: Optional[str] = Header(default=None)):
    user = _session_user(request)
    if user:
        scope = user.id
    elif _api_key_ok(x_api_key):
        scope = None           # admin / open dev: all runs
    else:
        raise HTTPException(401, "log in or send a valid X-API-Key")
    return [
        {"id": r.id, "kind": r.kind.value, "status": r.status.value, "label": r.label,
         "created_at": r.created_at.isoformat()}
        for r in list_runs(limit, owner_id=scope)
    ]


@app.get("/runs/{run_id}")
async def run_json(run_id: str, request: Request, x_api_key: Optional[str] = Header(default=None)):
    run = get_run(run_id)
    if not run:
        raise HTTPException(404, "run not found")
    if not _can_view_run(run, request, x_api_key):
        raise HTTPException(403, "not your run")
    return run.model_dump(mode="json")


@app.get("/runs/{run_id}/report", response_class=HTMLResponse)
async def run_report(run_id: str, request: Request, x_api_key: Optional[str] = Header(default=None)):
    run = get_run(run_id)
    if not run:
        raise HTTPException(404, "run not found")
    if not _can_view_run(run, request, x_api_key):
        raise HTTPException(403, "not your run")
    if run.status != RunStatus.done or not run.result:
        return HTMLResponse(
            f"<h2>Run {run_id} is {run.status.value}.</h2>"
            f"<p>{run.error or 'Still processing — refresh shortly.'}</p>",
            status_code=202 if run.status == RunStatus.running else 200,
        )
    if run.kind == RunKind.analyze:
        return HTMLResponse(render_analyze(AnalyzeReport.model_validate(run.result)))
    return HTMLResponse(render_compare(CompareReport.model_validate(run.result)))


@app.get("/runs/{run_id}/brief.md")
async def run_brief_md(run_id: str, request: Request, x_api_key: Optional[str] = Header(default=None)):
    """The analyze run's writing brief as export-ready markdown (SEO handoff)."""
    run = get_run(run_id)
    if not run:
        raise HTTPException(404, "run not found")
    if not _can_view_run(run, request, x_api_key):
        raise HTTPException(403, "not your run")
    if run.status != RunStatus.done or not run.result or run.kind != RunKind.analyze:
        raise HTTPException(404, "no finished analyze report for this run")
    report = AnalyzeReport.model_validate(run.result)
    if not report.writing_brief or not report.writing_brief.markdown:
        raise HTTPException(404, "this run has no writing brief")
    return PlainTextResponse(report.writing_brief.markdown, media_type="text/markdown; charset=utf-8")


# --------------------------------------------------------------------------- #
# Web auth pages (signup / login / logout)
# --------------------------------------------------------------------------- #
_CSS = """
*{box-sizing:border-box}
body{font:15px/1.5 system-ui,Segoe UI,Roboto,sans-serif;background:#0f1115;color:#e6e8ee;margin:0;padding:0}
a{color:#52d6c9;text-decoration:none}a:hover{text-decoration:underline}
.wrap{max-width:980px;margin:0 auto;padding:32px}
.card{background:#161a22;border:1px solid #232838;border-radius:12px;padding:24px;margin:0 auto}
.auth{max-width:380px;margin:8vh auto}
h1{color:#7c8cff;margin:0 0 4px}h2{margin:0 0 16px;font-weight:600}
label{display:block;font-size:13px;color:#9aa3b8;margin:14px 0 6px}
input,select{width:100%;padding:10px 12px;background:#0f1115;border:1px solid #2b3142;border-radius:8px;color:#e6e8ee;font-size:14px}
button{margin-top:18px;width:100%;padding:11px;background:#7c8cff;border:0;border-radius:8px;color:#0f1115;font-weight:700;font-size:14px;cursor:pointer}
button:hover{background:#94a1ff}
.row{display:flex;gap:12px}.row>div{flex:1}
.muted{color:#9aa3b8;font-size:13px}
.err{background:#3a1d22;border:1px solid #6b2b35;color:#ffb4bd;padding:10px 12px;border-radius:8px;margin:12px 0;font-size:13px}
.topbar{display:flex;justify-content:space-between;align-items:center;margin-bottom:24px}
table{border-collapse:collapse;width:100%;margin-top:8px}
td,th{padding:9px 12px;border-bottom:1px solid #232838;text-align:left;font-size:13px}
th{color:#9aa3b8;font-weight:600}
.pill{display:inline-block;padding:2px 9px;border-radius:999px;font-size:12px;background:#232838}
.pill.done{background:#15392f;color:#5ce6c0}.pill.running{background:#3a3415;color:#e6d35c}
.pill.error{background:#3a1d22;color:#ffb4bd}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}
@media(max-width:760px){.grid{grid-template-columns:1fr}}
"""


def _page(title: str, body: str) -> str:
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{title}</title><style>{_CSS}</style></head><body>{body}</body></html>"
    )


def _auth_form(kind: str, error: str = "", email: str = "") -> str:
    """kind = 'login' | 'signup'."""
    is_signup = kind == "signup"
    title = "Create your account" if is_signup else "Welcome back"
    action = "/signup" if is_signup else "/login"
    cta = "Sign up" if is_signup else "Log in"
    alt = (
        '<p class="muted" style="margin-top:18px">Already have an account? <a href="/login">Log in</a></p>'
        if is_signup else
        '<p class="muted" style="margin-top:18px">New here? <a href="/signup">Create an account</a></p>'
    )
    err_html = f'<div class="err">{error}</div>' if error else ""
    pw_hint = '<div class="muted" style="margin-top:6px">At least 8 characters.</div>' if is_signup else ""
    return _page(
        cta + " · RankLens",
        '<div class="wrap"><div class="card auth">'
        '<h1>RankLens</h1>'
        f'<h2>{title}</h2>'
        f'{err_html}'
        f'<form method="post" action="{action}">'
        '<label>Email</label>'
        f'<input name="email" type="email" autocomplete="email" required value="{email}">'
        '<label>Password</label>'
        f'<input name="password" type="password" autocomplete="{"new-password" if is_signup else "current-password"}" required>'
        f'{pw_hint}'
        f'<button type="submit">{cta}</button>'
        '</form>'
        f'{alt}'
        '</div></div>'
    )


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if _session_user(request):
        return RedirectResponse("/", status_code=303)
    return HTMLResponse(_auth_form("login"))


@app.post("/login")
async def login_submit(email: str = Form(...), password: str = Form(...)):
    user = auth.authenticate(email, password)
    if not user:
        return HTMLResponse(_auth_form("login", "Wrong email or password.", auth.normalize_email(email)),
                            status_code=401)
    resp = RedirectResponse("/", status_code=303)
    _set_session_cookie(resp, auth.create_session(user.id))
    return resp


@app.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request):
    if _session_user(request):
        return RedirectResponse("/", status_code=303)
    if not get_settings().ranklens_allow_signup:
        return HTMLResponse(_auth_form("login", "Sign-ups are currently closed."), status_code=403)
    return HTMLResponse(_auth_form("signup"))


@app.post("/signup")
async def signup_submit(email: str = Form(...), password: str = Form(...)):
    if not get_settings().ranklens_allow_signup:
        return HTMLResponse(_auth_form("login", "Sign-ups are currently closed."), status_code=403)
    email = auth.normalize_email(email)
    err = auth.validate_signup(email, password)
    if err:
        return HTMLResponse(_auth_form("signup", err, email), status_code=400)
    try:
        user = auth.create_user(email, password)
    except ValueError:
        return HTMLResponse(_auth_form("signup", "That email is already registered.", email), status_code=409)
    resp = RedirectResponse("/", status_code=303)
    _set_session_cookie(resp, auth.create_session(user.id))
    return resp


@app.post("/logout")
async def logout(request: Request):
    auth.destroy_session(request.cookies.get(auth.SESSION_COOKIE))
    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie(auth.SESSION_COOKIE, path="/")
    return resp


# --------------------------------------------------------------------------- #
# Dashboard + UI run launcher (logged-in users)
# --------------------------------------------------------------------------- #
@app.post("/ui/analyze")
async def ui_analyze(
    request: Request,
    bg: BackgroundTasks,
    keyword: str = Form(...),
    target_url: str = Form(""),
    country: str = Form("us"),
    include_backlinks: bool = Form(False),
    include_brand: bool = Form(False),
):
    user = _session_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    keyword = keyword.strip()
    if not keyword:
        return RedirectResponse("/", status_code=303)
    gl = (country.strip() or "us").lower()
    if gl not in COUNTRY_LOCATION:
        gl = "us"
    req = AnalyzeRequest(
        keyword=keyword,
        target_url=(target_url.strip() or None),
        country=gl,
        include_backlinks=include_backlinks,
        include_brand=include_brand,
    )
    run = _new_run(RunKind.analyze, f'analyze: "{req.keyword}"', req.model_dump(mode="json"), user.id)
    bg.add_task(_do_analyze, run.id, req)
    return RedirectResponse("/", status_code=303)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    user = _session_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)

    items = list_runs(50, owner_id=user.id)
    if items:
        rows = "".join(
            f'<tr><td>{r.created_at:%Y-%m-%d %H:%M}</td><td>{r.kind.value}</td>'
            f'<td>{r.label}</td><td><span class="pill {r.status.value}">{r.status.value}</span></td>'
            f'<td><a href="/runs/{r.id}/report">report</a> · <a href="/runs/{r.id}">json</a></td></tr>'
            for r in items
        )
        table = (
            '<table><tr><th>When</th><th>Kind</th><th>Label</th><th>Status</th><th></th></tr>'
            f'{rows}</table>'
        )
    else:
        table = '<p class="muted">No analyses yet. Start one on the left.</p>'

    body = (
        '<div class="wrap">'
        '<div class="topbar"><h1>RankLens</h1>'
        f'<div class="muted">{user.email} · '
        '<form method="post" action="/logout" style="display:inline">'
        '<button type="submit" style="width:auto;margin:0;padding:6px 12px;background:#232838;color:#e6e8ee">Log out</button>'
        '</form></div></div>'
        '<div class="grid">'
        # New analysis form
        '<div class="card"><h2>New analysis</h2>'
        '<form method="post" action="/ui/analyze">'
        '<label>Keyword</label>'
        '<input name="keyword" placeholder="best running shoes" required>'
        '<label>Your target URL <span class="muted">(optional)</span></label>'
        '<input name="target_url" placeholder="https://example.com/page">'
        '<label>Country to search from</label>'
        f'<select name="country">{_country_options("us")}</select>'
        '<label style="display:flex;align-items:center;gap:8px;margin-top:12px;font-weight:400">'
        '<input type="checkbox" name="include_backlinks" value="true" checked style="width:auto;margin:0">'
        'Backlinks &amp; off-page authority <span class="muted">(needs a backlink provider)</span></label>'
        '<label style="display:flex;align-items:center;gap:8px;font-weight:400">'
        '<input type="checkbox" name="include_brand" value="true" checked style="width:auto;margin:0">'
        'Brand strength / branded search volume <span class="muted">(needs a backlink provider)</span></label>'
        '<button type="submit">Run analysis</button>'
        '<div class="muted" style="margin-top:10px">Takes ~30–60s. The run appears on the right; refresh to see it finish.</div>'
        '</form></div>'
        # Runs list
        f'<div class="card"><h2>Your runs</h2>{table}</div>'
        '</div></div>'
    )
    return HTMLResponse(_page("RankLens", body))


@app.get("/health")
async def health():
    return JSONResponse({"ok": True})
