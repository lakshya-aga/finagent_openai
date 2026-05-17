"""SQLite schema for the paper-trading book.

Created idempotently on first read via ``CREATE TABLE IF NOT EXISTS``
so we don't need a separate migration step — the experiments DB
self-heals when the paper_trading module first writes.

Tables:
  predictions          : raw direction calls, one row per (date, ticker)
  portfolio_snapshots  : daily account state per (date, strategy)
  position_snapshots   : per-ticker MTM per (date, strategy, ticker)
  trades               : open-then-close trade lifecycle
  nifty50_universe     : ticker → name, sector, last_market_cap
"""

from __future__ import annotations

SCHEMA = """
-- ── predictions ──────────────────────────────────────────────────
-- Direction call for one ticker on one trading date. Direction
-- semantics: -1 = short, 0 = neutral (close any open position),
-- +1 = long. UNIQUE(date, ticker) so an "override" path can UPDATE
-- without producing duplicates.
CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    date            TEXT NOT NULL,                  -- 'YYYY-MM-DD' (trading day)
    ticker          TEXT NOT NULL,
    direction       INTEGER NOT NULL,               -- -1 / 0 / +1
    confidence      REAL,                           -- 0..1 from upstream analyst
    reasoning       TEXT,
    target_price    REAL,                           -- optional
    stop_loss_price REAL,                           -- optional
    source          TEXT NOT NULL DEFAULT 'manual', -- 'manual' / 'debate:<id>' / 'agent:v1'
    created_at      REAL NOT NULL,
    UNIQUE(date, ticker)
);
CREATE INDEX IF NOT EXISTS idx_predictions_date    ON predictions(date);
CREATE INDEX IF NOT EXISTS idx_predictions_ticker  ON predictions(ticker);


-- ── portfolio_snapshots ──────────────────────────────────────────
-- One row per (date, strategy). The equity curve is a scan ordered
-- by date. UNIQUE(date, strategy) so EOD-close is idempotent —
-- re-running it overwrites the previous snapshot rather than
-- producing duplicates.
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    date                 TEXT NOT NULL,
    strategy             TEXT NOT NULL,       -- 'equal_weight' | 'market_cap'
    equity_value         REAL NOT NULL,       -- INR (positions MTM + cash)
    cash                 REAL NOT NULL,       -- INR
    gross_exposure       REAL NOT NULL,       -- Σ |notional_i| / equity
    net_exposure         REAL NOT NULL,       -- Σ notional_i / equity (signed)
    daily_pnl            REAL NOT NULL,       -- INR
    daily_return_pct     REAL NOT NULL,       -- decimal (0.0125 = 1.25%)
    transaction_costs    REAL NOT NULL DEFAULT 0,
    n_long               INTEGER NOT NULL DEFAULT 0,
    n_short              INTEGER NOT NULL DEFAULT 0,
    n_neutral            INTEGER NOT NULL DEFAULT 0,
    created_at           REAL NOT NULL,
    UNIQUE(date, strategy)
);
CREATE INDEX IF NOT EXISTS idx_psnaps_date     ON portfolio_snapshots(date);
CREATE INDEX IF NOT EXISTS idx_psnaps_strategy ON portfolio_snapshots(strategy);


-- ── position_snapshots ───────────────────────────────────────────
-- Per-ticker MTM for a given date + strategy. Drives the "open
-- positions" table on the dashboard. Filter date=max(date) to get
-- "today's positions".
CREATE TABLE IF NOT EXISTS position_snapshots (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    date                TEXT NOT NULL,
    strategy            TEXT NOT NULL,
    ticker              TEXT NOT NULL,
    direction           INTEGER NOT NULL,     -- -1 / +1 (0 not stored)
    weight              REAL NOT NULL,        -- signed (-1..+1)
    notional            REAL NOT NULL,        -- INR notional at snapshot price
    entry_date          TEXT NOT NULL,
    entry_price         REAL NOT NULL,
    current_price       REAL NOT NULL,
    days_held           INTEGER NOT NULL,
    unrealized_pnl      REAL NOT NULL,        -- INR
    unrealized_pnl_pct  REAL NOT NULL,        -- decimal
    UNIQUE(date, strategy, ticker)
);
CREATE INDEX IF NOT EXISTS idx_possnaps_dstrat ON position_snapshots(date, strategy);
CREATE INDEX IF NOT EXISTS idx_possnaps_ticker ON position_snapshots(ticker);


-- ── trades ───────────────────────────────────────────────────────
-- Lifecycle of one trade — opened, then later closed. Open trades
-- have closed_at = NULL. Filter is_open = (closed_at IS NULL).
--
-- v2 columns (target_price, stop_loss_price, opened_via, closed_ts)
-- support intraday execution. target/stoploss are copied from the
-- prediction at open-time so a later prediction edit doesn't move
-- the trigger on an already-open position. opened_via tags the
-- entry source so the UI can label market-open vs eod fills.
-- closed_ts is the wall-clock instant of the trigger fire (UTC
-- epoch seconds) — distinct from closed_at which stays a trading-
-- date 'YYYY-MM-DD'.
CREATE TABLE IF NOT EXISTS trades (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy          TEXT NOT NULL,
    ticker            TEXT NOT NULL,
    direction         INTEGER NOT NULL,
    opened_at         TEXT NOT NULL,             -- 'YYYY-MM-DD'
    open_price        REAL NOT NULL,
    open_weight       REAL NOT NULL,
    closed_at         TEXT,
    close_price       REAL,
    realized_pnl      REAL,                      -- INR
    close_reason      TEXT,                      -- 'direction_change' | 'neutral' | 'stop_loss' | 'target' | 'manual' | 'market_close'
    transaction_cost  REAL NOT NULL DEFAULT 20.0,

    -- v2 — added by store._migrate_v2_trade_columns on first run.
    -- The CREATE TABLE keeps them for fresh DBs; the migration
    -- ALTERs existing DBs in place.
    target_price       REAL,
    stop_loss_price    REAL,
    opened_via         TEXT DEFAULT 'eod_close', -- 'market_open' | 'eod_close' | 'manual'
    closed_ts          REAL                       -- UTC epoch seconds (intraday precision)
);
CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy);
CREATE INDEX IF NOT EXISTS idx_trades_open     ON trades(strategy, ticker, closed_at);


-- ── nifty50_universe ─────────────────────────────────────────────
-- Reference data: name + sector + last-refreshed market cap. Seeded
-- on first use by paper_trading.universe.refresh_market_caps().
CREATE TABLE IF NOT EXISTS nifty50_universe (
    ticker           TEXT PRIMARY KEY,
    name             TEXT,
    sector           TEXT,
    market_cap       REAL,                       -- INR
    refreshed_at     REAL                        -- unix ts
);
"""
