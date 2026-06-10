"""Credit accounts + ledger for the ticker-analyser product.

Monetisation model: each FRESH panel analysis costs credits; cached
same-day analyses are free (the analysis of a ticker on a given day is
identical for every user, so the second request costs ~$0 to serve).
New accounts get a signup bonus so the funnel is
sign up → run a few free analyses → buy more.

Design notes:
  * Same SQLite DB as the experiment store (one file to back up, same
    pattern as paper_trading/store.py). Each function opens its own
    connection; schema creation is idempotent and lazy.
  * ``charge`` is atomic and race-safe: a conditional UPDATE
    (``balance = balance - ? WHERE balance >= ?``) either applies fully
    or not at all — two concurrent charges against a balance of 1 can
    never both succeed.
  * Every balance change is journalled in ``credit_events`` with the
    post-event balance, so support questions ("where did my credits
    go?") are answerable from the ledger alone.
  * Enforcement is gated behind the CREDITS_ENFORCEMENT env var
    (default OFF) so the schema + plumbing can ship and run in
    production before the paywall is turned on. While OFF, charges are
    still attempted and journalled when possible — giving real usage
    data — but a failed charge never blocks an analysis.

Env knobs:
  CREDITS_SIGNUP_BONUS       — credits granted on first account touch (default 3)
  CREDITS_COST_PER_ANALYSIS  — credits per fresh panel run (default 1)
  CREDITS_ENFORCEMENT        — "1" to block runs on insufficient credits (default "0")
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS credit_accounts (
    user_id     TEXT PRIMARY KEY,
    balance     INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS credit_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       TEXT NOT NULL,
    delta         INTEGER NOT NULL,
    reason        TEXT NOT NULL,
    ref_id        TEXT,
    balance_after INTEGER NOT NULL,
    created_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_credit_events_user ON credit_events(user_id, created_at DESC);
"""


def signup_bonus() -> int:
    try:
        return max(0, int(os.environ.get("CREDITS_SIGNUP_BONUS", "3")))
    except ValueError:
        return 3


def cost_per_analysis() -> int:
    try:
        return max(0, int(os.environ.get("CREDITS_COST_PER_ANALYSIS", "1")))
    except ValueError:
        return 1


def enforcement_enabled() -> bool:
    return os.environ.get("CREDITS_ENFORCEMENT", "").strip() in {"1", "true", "yes"}


def normalize_user_id(user_id: str) -> str:
    """Canonical form for account keys. Identity comes from the web
    tier as an email; lowercase + strip so 'Alice@X.com' and
    'alice@x.com ' are one account."""
    return (user_id or "").strip().lower()


def _db_path() -> str:
    """Resolve the ledger DB path at call time (not import time) so
    tests can point FINAGENT_EXPERIMENT_DB at a temp file."""
    env = os.environ.get("FINAGENT_EXPERIMENT_DB", "").strip()
    if env:
        return env
    from finagent.experiments import _DEFAULT_PATH

    return str(_DEFAULT_PATH)


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _journal(
    conn: sqlite3.Connection,
    user_id: str,
    delta: int,
    reason: str,
    ref_id: Optional[str],
    balance_after: int,
) -> None:
    conn.execute(
        "INSERT INTO credit_events (user_id, delta, reason, ref_id, balance_after, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, delta, reason, ref_id, balance_after, time.time()),
    )


def ensure_account(user_id: str) -> dict[str, Any]:
    """Idempotent account creation with signup bonus.

    First call for a user creates the account and grants the signup
    bonus (journalled as 'signup_bonus'); every later call is a no-op
    that returns the current balance. Safe to call on every login —
    the INSERT OR IGNORE + rowcount check means the bonus can only be
    granted once even under concurrent first-logins.
    """
    uid = normalize_user_id(user_id)
    if not uid:
        raise ValueError("user_id is required")
    bonus = signup_bonus()
    now = time.time()
    with _conn() as conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO credit_accounts (user_id, balance, created_at, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (uid, bonus, now, now),
        )
        created = cur.rowcount > 0
        if created and bonus > 0:
            _journal(conn, uid, bonus, "signup_bonus", None, bonus)
        (balance,) = conn.execute(
            "SELECT balance FROM credit_accounts WHERE user_id = ?", (uid,)
        ).fetchone()
    return {"user_id": uid, "balance": int(balance), "created": created}


def get_balance(user_id: str) -> int:
    """Current balance; 0 for unknown accounts (no implicit creation)."""
    uid = normalize_user_id(user_id)
    with _conn() as conn:
        row = conn.execute(
            "SELECT balance FROM credit_accounts WHERE user_id = ?", (uid,)
        ).fetchone()
    return int(row["balance"]) if row else 0


def grant(
    user_id: str,
    amount: int,
    reason: str,
    ref_id: Optional[str] = None,
) -> int:
    """Add credits (purchase, promo, refund). Returns the new balance.
    Creates the account WITHOUT a signup bonus if it doesn't exist —
    a grant to a fresh account shouldn't stack with the bonus."""
    uid = normalize_user_id(user_id)
    if not uid:
        raise ValueError("user_id is required")
    if amount <= 0:
        raise ValueError("grant amount must be positive")
    now = time.time()
    with _conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO credit_accounts (user_id, balance, created_at, updated_at) "
            "VALUES (?, 0, ?, ?)",
            (uid, now, now),
        )
        conn.execute(
            "UPDATE credit_accounts SET balance = balance + ?, updated_at = ? WHERE user_id = ?",
            (amount, now, uid),
        )
        (balance,) = conn.execute(
            "SELECT balance FROM credit_accounts WHERE user_id = ?", (uid,)
        ).fetchone()
        _journal(conn, uid, amount, reason, ref_id, int(balance))
    return int(balance)


def charge(
    user_id: str,
    amount: int,
    reason: str,
    ref_id: Optional[str] = None,
) -> tuple[bool, int]:
    """Atomically deduct ``amount`` if the balance covers it.

    Returns ``(ok, balance_after)``. The conditional UPDATE makes this
    race-safe: with balance=1 and two concurrent charge(1) calls,
    exactly one succeeds. Unknown accounts always fail (ok=False, 0).
    """
    uid = normalize_user_id(user_id)
    if not uid:
        return False, 0
    if amount <= 0:
        return True, get_balance(uid)  # zero-cost charge is a no-op
    now = time.time()
    with _conn() as conn:
        cur = conn.execute(
            "UPDATE credit_accounts SET balance = balance - ?, updated_at = ? "
            "WHERE user_id = ? AND balance >= ?",
            (amount, now, uid, amount),
        )
        ok = cur.rowcount > 0
        row = conn.execute(
            "SELECT balance FROM credit_accounts WHERE user_id = ?", (uid,)
        ).fetchone()
        balance = int(row["balance"]) if row else 0
        if ok:
            _journal(conn, uid, -amount, reason, ref_id, balance)
    return ok, balance


def refund(user_id: str, amount: int, ref_id: Optional[str] = None) -> int:
    """Give back credits for a failed run. Journalled distinctly from
    purchases so the ledger shows refunds as refunds."""
    return grant(user_id, amount, "refund:failed_run", ref_id=ref_id)


def history(user_id: str, limit: int = 50) -> list[dict[str, Any]]:
    """Most-recent-first ledger entries for one account."""
    uid = normalize_user_id(user_id)
    limit = max(1, min(int(limit), 500))
    with _conn() as conn:
        rows = conn.execute(
            "SELECT delta, reason, ref_id, balance_after, created_at "
            "FROM credit_events WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (uid, limit),
        ).fetchall()
    return [dict(r) for r in rows]
