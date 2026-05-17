"""Visits table — single source of truth for the analytics dashboard.

CREATE IF NOT EXISTS so the experiments DB self-heals on first call
from store.py — no separate migration step.
"""

from __future__ import annotations

SCHEMA = """
CREATE TABLE IF NOT EXISTS visits (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           REAL NOT NULL,                -- UTC epoch seconds, server-set
    path         TEXT NOT NULL,                -- '/app/projects' (query stripped)
    anonymous_id TEXT,                         -- random UUID from browser localStorage; null if blocked
    user_id      TEXT,                         -- session.user.email if signed in; null otherwise
    referrer     TEXT,                         -- document.referrer at fire time
    ua_class     TEXT NOT NULL DEFAULT 'unknown'  -- 'desktop' | 'mobile' | 'tablet' | 'bot' | 'unknown'
);
CREATE INDEX IF NOT EXISTS idx_visits_ts        ON visits(ts DESC);
CREATE INDEX IF NOT EXISTS idx_visits_path      ON visits(path);
CREATE INDEX IF NOT EXISTS idx_visits_anonymous ON visits(anonymous_id);
CREATE INDEX IF NOT EXISTS idx_visits_ua_class  ON visits(ua_class);
"""
