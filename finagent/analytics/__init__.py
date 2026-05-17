"""Lightweight self-hosted visit analytics.

Single SQLite table + a public ``track`` endpoint + admin aggregations.
Pattern mirrors the existing ``finagent.paper_trading`` module so the
operational story is the same: no external dependencies, schema
self-heals on first call, idempotent indexes, admin surface gated by
``finagent.experiments``'s existing auth.

Privacy posture:
  - Anonymous-only by default. The browser drops a random UUID into
    localStorage; visits are keyed off that UUID for "unique visitor"
    counts. No IP address is stored. The raw User-Agent string is
    classified into a coarse bucket (desktop/mobile/tablet/bot) and
    only the bucket is persisted — full UAs are not.
  - Path is stripped of query strings before storage so we don't
    leak session tokens in shareable URLs.
  - Bots are recorded but excluded from "human" counts by default.

What the dashboard surfaces:
  - Total page-views and unique visitors over today / 7d / 30d / all
  - Daily timeline (line chart)
  - Top pages by view count
  - Referrer breakdown
"""

from . import schema, store  # noqa: F401
