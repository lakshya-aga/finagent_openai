"""Write + read helpers for the visits table.

Schema is created on first call via ``_ensure_schema``. Reads return
plain dicts so the API layer stays free of sqlite-specific types.

Aggregation queries are deliberately simple — SQLite group-by + a bit
of Python arithmetic. At <100k rows/day this is fine; the bottleneck
is the dashboard re-fetch, not the query.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import time
import urllib.parse
from contextlib import contextmanager
from typing import Any, Iterator

from .schema import SCHEMA

logger = logging.getLogger(__name__)


def _db_path():
    """Resolve the experiments DB path. Lazy-import to keep this module
    loadable without the finagent.experiments stack."""
    from finagent.experiments import _DEFAULT_PATH
    return _DEFAULT_PATH


_SCHEMA_CREATED = False


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    c = sqlite3.connect(str(_db_path()))
    c.row_factory = sqlite3.Row
    try:
        global _SCHEMA_CREATED
        if not _SCHEMA_CREATED:
            c.executescript(SCHEMA)
            _SCHEMA_CREATED = True
        yield c
        c.commit()
    finally:
        c.close()


# ── UA classification ─────────────────────────────────────────────


# Lightweight UA bucketing. Matches common cases; doesn't try to
# enumerate every device. Anything that doesn't hit a regex becomes
# 'unknown' (still counted in totals; not split out in breakdowns).
_BOT_RE = re.compile(
    r"bot|crawler|spider|crawling|googlebot|bingbot|yandex|baiduspider"
    r"|duckduckbot|slurp|facebookexternalhit|twitterbot|linkedinbot|"
    r"discordbot|telegrambot|whatsapp|slackbot|preview|pingdom|uptimerobot"
    r"|headlesschrome|phantomjs",
    re.IGNORECASE,
)
_MOBILE_RE = re.compile(r"mobi|iphone|ipod|android.*mobile|blackberry|opera mini", re.IGNORECASE)
_TABLET_RE = re.compile(r"ipad|tablet|android(?!.*mobile)", re.IGNORECASE)


def classify_ua(ua: str | None) -> str:
    """Coarse bucket: bot | mobile | tablet | desktop | unknown."""
    if not ua:
        return "unknown"
    if _BOT_RE.search(ua):
        return "bot"
    if _TABLET_RE.search(ua):
        return "tablet"
    if _MOBILE_RE.search(ua):
        return "mobile"
    # Anything that's not bot/mobile/tablet but is a real browser is desktop.
    if re.search(r"mozilla|chrome|safari|firefox|edge", ua, re.IGNORECASE):
        return "desktop"
    return "unknown"


# ── Path sanitisation ─────────────────────────────────────────────


def sanitise_path(raw: str | None) -> str:
    """Strip query strings + fragments so we never leak tokens or
    session IDs into the visits log. Falls back to '/' on bad input."""
    if not raw or not isinstance(raw, str):
        return "/"
    try:
        parsed = urllib.parse.urlparse(raw)
        path = parsed.path or "/"
    except Exception:
        return "/"
    # Defensively cap path length — pathological tests can blow the
    # SQLite TEXT cell.
    return path[:300]


# ── Referrer normalisation ────────────────────────────────────────


def normalise_referrer(raw: str | None) -> str | None:
    """Reduce a full referrer URL to just its origin (host) so the
    breakdown stays clean. e.g. 'https://google.com/search?q=...' →
    'google.com'. Empty / same-origin refs are dropped to null."""
    if not raw:
        return None
    try:
        parsed = urllib.parse.urlparse(raw)
        host = (parsed.netloc or "").lower()
    except Exception:
        return None
    if not host:
        return None
    # Strip leading www. for cleaner grouping.
    return host[4:] if host.startswith("www.") else host


# ── Writes ────────────────────────────────────────────────────────


def record_visit(
    *,
    path: str,
    anonymous_id: str | None = None,
    user_id: str | None = None,
    referrer: str | None = None,
    ua: str | None = None,
) -> int:
    """Insert one visit row. Returns the row id (zero on insert error)."""
    sanitised = sanitise_path(path)
    ref = normalise_referrer(referrer)
    ua_class = classify_ua(ua)
    with _conn() as c:
        cur = c.execute(
            """
            INSERT INTO visits (ts, path, anonymous_id, user_id, referrer, ua_class)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (time.time(), sanitised, anonymous_id, user_id, ref, ua_class),
        )
        return cur.lastrowid or 0


# ── Reads ─────────────────────────────────────────────────────────


def _row(r) -> dict[str, Any]:
    return {k: r[k] for k in r.keys()}


def summary(*, exclude_bots: bool = True) -> dict:
    """Headline numbers: total + unique visitors over today / 7d / 30d /
    all-time. ``unique`` is the count of distinct anonymous_id values
    (rows without an id contribute 1 each, since each null anonymous
    is treated as an independent visit — conservative)."""
    now = time.time()
    cutoffs = {
        "today": now - 86_400,
        "last_7d": now - 7 * 86_400,
        "last_30d": now - 30 * 86_400,
    }
    bot_filter = "AND ua_class != 'bot'" if exclude_bots else ""
    out: dict[str, dict] = {}
    with _conn() as c:
        for label, cutoff in cutoffs.items():
            row = c.execute(
                f"""
                SELECT
                  COUNT(*) AS pageviews,
                  COUNT(DISTINCT COALESCE(anonymous_id, 'null_' || id)) AS uniques
                FROM visits
                WHERE ts >= ? {bot_filter}
                """,
                (cutoff,),
            ).fetchone()
            out[label] = {"pageviews": row["pageviews"], "uniques": row["uniques"]}
        # All-time
        row = c.execute(
            f"""
            SELECT
              COUNT(*) AS pageviews,
              COUNT(DISTINCT COALESCE(anonymous_id, 'null_' || id)) AS uniques
            FROM visits
            WHERE 1=1 {bot_filter}
            """
        ).fetchone()
        out["all_time"] = {"pageviews": row["pageviews"], "uniques": row["uniques"]}
        # Bot count (always, so the operator can see the noise level)
        bots = c.execute("SELECT COUNT(*) AS n FROM visits WHERE ua_class = 'bot'").fetchone()
        out["bot_count_all_time"] = bots["n"]
    return out


def timeline(*, days: int = 30, exclude_bots: bool = True) -> list[dict]:
    """One row per UTC date over the last ``days``, with pageviews +
    unique visitor count. Missing days are emitted with zeroes so the
    chart doesn't gap."""
    from datetime import date, timedelta, datetime, timezone

    end_date = date.today()
    start_date = end_date - timedelta(days=days - 1)
    bot_filter = "AND ua_class != 'bot'" if exclude_bots else ""
    with _conn() as c:
        rows = c.execute(
            f"""
            SELECT
              date(ts, 'unixepoch') AS day,
              COUNT(*) AS pageviews,
              COUNT(DISTINCT COALESCE(anonymous_id, 'null_' || id)) AS uniques
            FROM visits
            WHERE ts >= ? {bot_filter}
            GROUP BY day
            ORDER BY day
            """,
            (datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc).timestamp(),),
        ).fetchall()
        observed = {r["day"]: (r["pageviews"], r["uniques"]) for r in rows}
    out: list[dict] = []
    cur = start_date
    while cur <= end_date:
        pv, uq = observed.get(cur.isoformat(), (0, 0))
        out.append({"date": cur.isoformat(), "pageviews": pv, "uniques": uq})
        cur += timedelta(days=1)
    return out


def top_pages(*, days: int = 7, limit: int = 20, exclude_bots: bool = True) -> list[dict]:
    """Top N most-visited paths over the last ``days``."""
    cutoff = time.time() - days * 86_400
    bot_filter = "AND ua_class != 'bot'" if exclude_bots else ""
    with _conn() as c:
        rows = c.execute(
            f"""
            SELECT
              path,
              COUNT(*) AS pageviews,
              COUNT(DISTINCT COALESCE(anonymous_id, 'null_' || id)) AS uniques
            FROM visits
            WHERE ts >= ? {bot_filter}
            GROUP BY path
            ORDER BY pageviews DESC
            LIMIT ?
            """,
            (cutoff, int(limit)),
        ).fetchall()
        return [_row(r) for r in rows]


def top_referrers(*, days: int = 7, limit: int = 20, exclude_bots: bool = True) -> list[dict]:
    cutoff = time.time() - days * 86_400
    bot_filter = "AND ua_class != 'bot'" if exclude_bots else ""
    with _conn() as c:
        rows = c.execute(
            f"""
            SELECT referrer, COUNT(*) AS pageviews
            FROM visits
            WHERE ts >= ? AND referrer IS NOT NULL {bot_filter}
            GROUP BY referrer
            ORDER BY pageviews DESC
            LIMIT ?
            """,
            (cutoff, int(limit)),
        ).fetchall()
        return [_row(r) for r in rows]


def device_breakdown(*, days: int = 30, exclude_bots: bool = False) -> list[dict]:
    """Pageviews bucketed by ua_class. Bots are INCLUDED by default
    here because the operator wants to see how much bot noise the
    site is getting."""
    cutoff = time.time() - days * 86_400
    bot_filter = "AND ua_class != 'bot'" if exclude_bots else ""
    with _conn() as c:
        rows = c.execute(
            f"""
            SELECT ua_class, COUNT(*) AS pageviews
            FROM visits
            WHERE ts >= ? {bot_filter}
            GROUP BY ua_class
            ORDER BY pageviews DESC
            """,
            (cutoff,),
        ).fetchall()
        return [_row(r) for r in rows]
