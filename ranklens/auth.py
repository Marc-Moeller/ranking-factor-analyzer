"""Email/password user accounts + cookie sessions. Pure stdlib — no new deps.

Design goals: simple, secure-enough, and Postgres-portable (same one-file swap
story as ``store.py``). Passwords are PBKDF2-HMAC-SHA256 with a per-user random
salt. Sessions are opaque random tokens kept server-side; the cookie only carries
the token, never anything derivable into a password.

The web layer (``api/main.py``) is the only caller. It resolves the current user
from the session cookie and gates the dashboard + paid endpoints on it, while the
legacy ``X-API-Key`` keeps working for programmatic / admin access.
"""
from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from ranklens.config import get_settings
from ranklens.db import _conn, ensure_init

_PBKDF2_ROUNDS = 200_000
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
SESSION_COOKIE = "ranklens_session"


@dataclass
class User:
    id: str
    email: str
    created_at: str


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
def init_auth_db() -> None:
    """Schema now lives in ranklens.db.init_schema; kept as a thin alias so the
    existing call sites stay valid."""
    ensure_init()


# --------------------------------------------------------------------------- #
# Password hashing
# --------------------------------------------------------------------------- #
def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PBKDF2_ROUNDS)
    return f"pbkdf2_sha256${_PBKDF2_ROUNDS}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, rounds, salt_hex, hash_hex = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), int(rounds))
        return hmac.compare_digest(dk.hex(), hash_hex)
    except (ValueError, AttributeError):
        return False


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def validate_signup(email: str, password: str) -> Optional[str]:
    """Return an error string, or None when the credentials are acceptable."""
    if not _EMAIL_RE.match(email):
        return "Enter a valid email address."
    if len(password) < 8:
        return "Password must be at least 8 characters."
    return None


# --------------------------------------------------------------------------- #
# User CRUD
# --------------------------------------------------------------------------- #
def get_user_by_email(email: str) -> Optional[User]:
    init_auth_db()
    with _conn() as c:
        row = c.execute("SELECT * FROM users WHERE email = %s", (normalize_email(email),)).fetchone()
    return User(row["id"], row["email"], row["created_at"]) if row else None


def get_user_by_id(user_id: str) -> Optional[User]:
    init_auth_db()
    with _conn() as c:
        row = c.execute("SELECT * FROM users WHERE id = %s", (user_id,)).fetchone()
    return User(row["id"], row["email"], row["created_at"]) if row else None


def create_user(email: str, password: str) -> User:
    """Create a user. Raises ValueError('email_taken') on a duplicate email."""
    init_auth_db()
    email = normalize_email(email)
    if get_user_by_email(email):
        raise ValueError("email_taken")
    user = User(id=uuid.uuid4().hex[:12], email=email, created_at=datetime.now(timezone.utc).isoformat())
    with _conn() as c:
        c.execute(
            "INSERT INTO users (id, email, password_hash, created_at) VALUES (%s, %s, %s, %s)",
            (user.id, user.email, hash_password(password), user.created_at),
        )
    return user


def authenticate(email: str, password: str) -> Optional[User]:
    init_auth_db()
    with _conn() as c:
        row = c.execute("SELECT * FROM users WHERE email = %s", (normalize_email(email),)).fetchone()
    if not row or not verify_password(password, row["password_hash"]):
        return None
    return User(row["id"], row["email"], row["created_at"])


# --------------------------------------------------------------------------- #
# Sessions
# --------------------------------------------------------------------------- #
def create_session(user_id: str) -> str:
    init_auth_db()
    token = secrets.token_urlsafe(32)
    now = datetime.now(timezone.utc)
    expires = now + timedelta(days=get_settings().ranklens_session_days)
    with _conn() as c:
        c.execute(
            "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (%s, %s, %s, %s)",
            (token, user_id, now.isoformat(), expires.isoformat()),
        )
    return token


def user_for_session(token: Optional[str]) -> Optional[User]:
    if not token:
        return None
    init_auth_db()
    with _conn() as c:
        row = c.execute("SELECT * FROM sessions WHERE token = %s", (token,)).fetchone()
        if not row:
            return None
        try:
            expired = datetime.fromisoformat(row["expires_at"]) < datetime.now(timezone.utc)
        except ValueError:
            expired = True
        if expired:
            c.execute("DELETE FROM sessions WHERE token = %s", (token,))
            return None
    return get_user_by_id(row["user_id"])


def destroy_session(token: Optional[str]) -> None:
    if not token:
        return
    init_auth_db()
    with _conn() as c:
        c.execute("DELETE FROM sessions WHERE token = %s", (token,))
