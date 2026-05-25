"""SQLite read/write helpers for the paper-trading book.

Every function opens a fresh connection — SQLite is fine with this
at our scale (~hundreds of writes per day). Schema is auto-created
on first call via ``_ensure_schema()``.

Convention: functions returning rows return plain dicts, not Row
objects — keeps the API layer free of sqlite-specific imports.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Iterable, Iterator

from .schema import SCHEMA

logger = logging.getLogger(__name__)


def _db_path():
    """Resolve the experiments DB path. Imported lazily so this module
    stays loadable in environments where finagent.experiments isn't
    importable (CI, tests with sys.path overrides)."""
    from finagent.experiments import _DEFAULT_PATH

    return _DEFAULT_PATH


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    c = sqlite3.connect(str(_db_path()))
    c.row_factory = sqlite3.Row
    try:
        _ensure_schema(c)
        yield c
        c.commit()
    finally:
        c.close()


_SCHEMA_CREATED = False


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Idempotent. Runs once per process; second+ calls short-circuit.

    SCHEMA contains CREATE TABLE IF NOT EXISTS / CREATE INDEX IF
    NOT EXISTS so it's safe to re-run on existing DBs. For columns
    added after v1 we run ALTER TABLE … ADD COLUMN inside
    ``_migrate_v2_trade_columns`` — SQLite tolerates this
    idempotently when wrapped in PRAGMA table_info checks.
    """
    global _SCHEMA_CREATED
    if _SCHEMA_CREATED:
        return
    conn.executescript(SCHEMA)
    _migrate_v2_trade_columns(conn)
    _SCHEMA_CREATED = True


def _migrate_v2_trade_columns(conn: sqlite3.Connection) -> None:
    """Add intraday-execution + triple-barrier columns to legacy tables.

    No-op on fresh DBs (CREATE TABLE already includes the columns).
    """
    # trades — v2 intraday execution + v3 triple-barrier time column
    existing_trades = {
        r[1] for r in conn.execute("PRAGMA table_info(trades)").fetchall()
    }
    for col, ddl in [
        ("target_price", "REAL"),
        ("stop_loss_price", "REAL"),
        ("opened_via", "TEXT DEFAULT 'eod_close'"),
        ("closed_ts", "REAL"),
        ("max_hold_days", "INTEGER"),
    ]:
        if col not in existing_trades:
            conn.execute(f"ALTER TABLE trades ADD COLUMN {col} {ddl}")
            logger.info("paper_trading: migrated trades — added column %s", col)

    # predictions — v3 triple-barrier columns (time_horizon + max_hold_days)
    existing_preds = {
        r[1] for r in conn.execute("PRAGMA table_info(predictions)").fetchall()
    }
    for col, ddl in [
        ("time_horizon", "TEXT"),
        ("max_hold_days", "INTEGER"),
    ]:
        if col not in existing_preds:
            conn.execute(f"ALTER TABLE predictions ADD COLUMN {col} {ddl}")
            logger.info("paper_trading: migrated predictions — added column %s", col)


def _row_to_dict(r) -> dict[str, Any]:
    return {k: r[k] for k in r.keys()}


# ── predictions ────────────────────────────────────────────────────


def upsert_prediction(
    *,
    date: str,
    ticker: str,
    direction: int,
    confidence: float | None = None,
    reasoning: str | None = None,
    target_price: float | None = None,
    stop_loss_price: float | None = None,
    time_horizon: str | None = None,
    max_hold_days: int | None = None,
    source: str = "manual",
) -> int:
    """Insert-or-merge. UNIQUE(date, ticker) so a second call with the
    same key updates rather than duplicating.

    IMPORTANT: COALESCE semantics on ON CONFLICT — if the caller passes
    None for any of the optional fields, the existing value is PRESERVED.
    This stops the portfolio-manager agent (which commits direction only)
    from blowing away the target_price/stop_loss/time_horizon that the
    stock_analyst wrote upstream. Direction + source always overwrite —
    those are the fields the override path means to change.
    """
    if direction not in (-1, 0, 1):
        raise ValueError(f"direction must be -1/0/+1, got {direction!r}")
    with _conn() as c:
        cur = c.execute(
            """
            INSERT INTO predictions (
                date, ticker, direction, confidence, reasoning,
                target_price, stop_loss_price, time_horizon, max_hold_days,
                source, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(date, ticker) DO UPDATE SET
                direction       = excluded.direction,
                confidence      = COALESCE(excluded.confidence,      predictions.confidence),
                reasoning       = COALESCE(excluded.reasoning,       predictions.reasoning),
                target_price    = COALESCE(excluded.target_price,    predictions.target_price),
                stop_loss_price = COALESCE(excluded.stop_loss_price, predictions.stop_loss_price),
                time_horizon    = COALESCE(excluded.time_horizon,    predictions.time_horizon),
                max_hold_days   = COALESCE(excluded.max_hold_days,   predictions.max_hold_days),
                source          = excluded.source
            """,
            (
                date,
                ticker,
                direction,
                confidence,
                reasoning,
                target_price,
                stop_loss_price,
                time_horizon,
                max_hold_days,
                source,
                time.time(),
            ),
        )
        return cur.lastrowid or 0


def list_predictions(date: str | None = None, ticker: str | None = None) -> list[dict]:
    sql = "SELECT * FROM predictions WHERE 1=1"
    args: list[Any] = []
    if date is not None:
        sql += " AND date = ?"
        args.append(date)
    if ticker is not None:
        sql += " AND ticker = ?"
        args.append(ticker)
    sql += " ORDER BY date DESC, ticker ASC"
    with _conn() as c:
        return [_row_to_dict(r) for r in c.execute(sql, args)]


def latest_prediction_date() -> str | None:
    with _conn() as c:
        r = c.execute("SELECT MAX(date) AS d FROM predictions").fetchone()
        return r["d"] if r and r["d"] else None


# ── portfolio_snapshots ────────────────────────────────────────────


def upsert_portfolio_snapshot(snap: dict) -> None:
    """Idempotent. Re-running the EOD close routine overwrites the
    snapshot for that date+strategy rather than producing duplicates."""
    keys = (
        "date",
        "strategy",
        "equity_value",
        "cash",
        "gross_exposure",
        "net_exposure",
        "daily_pnl",
        "daily_return_pct",
        "transaction_costs",
        "n_long",
        "n_short",
        "n_neutral",
    )
    vals = tuple(snap[k] for k in keys) + (time.time(),)
    placeholders = ", ".join(["?"] * (len(keys) + 1))
    cols = ", ".join(keys + ("created_at",))
    update_set = ", ".join(f"{k} = excluded.{k}" for k in keys)
    with _conn() as c:
        c.execute(
            f"""
            INSERT INTO portfolio_snapshots ({cols})
            VALUES ({placeholders})
            ON CONFLICT(date, strategy) DO UPDATE SET {update_set}
            """,
            vals,
        )


def list_snapshots(
    strategy: str,
    *,
    start: str | None = None,
    end: str | None = None,
) -> list[dict]:
    sql = "SELECT * FROM portfolio_snapshots WHERE strategy = ?"
    args: list[Any] = [strategy]
    if start:
        sql += " AND date >= ?"
        args.append(start)
    if end:
        sql += " AND date <= ?"
        args.append(end)
    sql += " ORDER BY date ASC"
    with _conn() as c:
        return [_row_to_dict(r) for r in c.execute(sql, args)]


def latest_snapshot(strategy: str) -> dict | None:
    with _conn() as c:
        r = c.execute(
            "SELECT * FROM portfolio_snapshots WHERE strategy = ? "
            "ORDER BY date DESC LIMIT 1",
            (strategy,),
        ).fetchone()
        return _row_to_dict(r) if r else None


def previous_snapshot(strategy: str, before_date: str) -> dict | None:
    """The most recent snapshot strictly before ``before_date`` — used
    to anchor the next-day diff (yesterday's equity = today's baseline)."""
    with _conn() as c:
        r = c.execute(
            "SELECT * FROM portfolio_snapshots WHERE strategy = ? AND date < ? "
            "ORDER BY date DESC LIMIT 1",
            (strategy, before_date),
        ).fetchone()
        return _row_to_dict(r) if r else None


# ── position_snapshots ─────────────────────────────────────────────


def upsert_position_snapshot(snap: dict) -> None:
    keys = (
        "date",
        "strategy",
        "ticker",
        "direction",
        "weight",
        "notional",
        "entry_date",
        "entry_price",
        "current_price",
        "days_held",
        "unrealized_pnl",
        "unrealized_pnl_pct",
    )
    vals = tuple(snap[k] for k in keys)
    placeholders = ", ".join(["?"] * len(keys))
    cols = ", ".join(keys)
    update_set = ", ".join(f"{k} = excluded.{k}" for k in keys)
    with _conn() as c:
        c.execute(
            f"""
            INSERT INTO position_snapshots ({cols})
            VALUES ({placeholders})
            ON CONFLICT(date, strategy, ticker) DO UPDATE SET {update_set}
            """,
            vals,
        )


def clear_position_snapshots(date: str, strategy: str) -> None:
    """Used before re-running EOD-close for a given (date, strategy)
    so a smaller direction set doesn't leave stale positions hanging."""
    with _conn() as c:
        c.execute(
            "DELETE FROM position_snapshots WHERE date = ? AND strategy = ?",
            (date, strategy),
        )


def list_positions(strategy: str, date: str | None = None) -> list[dict]:
    """Open positions for a strategy at a date (default: most recent
    snapshot date). Returns sorted by weight desc."""
    if date is None:
        with _conn() as c:
            r = c.execute(
                "SELECT MAX(date) AS d FROM position_snapshots WHERE strategy = ?",
                (strategy,),
            ).fetchone()
            date = r["d"] if r and r["d"] else None
        if not date:
            return []
    with _conn() as c:
        rows = c.execute(
            """
            SELECT * FROM position_snapshots
            WHERE strategy = ? AND date = ?
            ORDER BY ABS(weight) DESC, ticker ASC
            """,
            (strategy, date),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]


# ── trades ─────────────────────────────────────────────────────────


def open_trade(
    *,
    strategy: str,
    ticker: str,
    direction: int,
    opened_at: str,
    open_price: float,
    open_weight: float,
    transaction_cost: float = 20.0,
    target_price: float | None = None,
    stop_loss_price: float | None = None,
    max_hold_days: int | None = None,
    opened_via: str = "eod_close",
) -> int:
    """Open a new trade. target_price + stop_loss_price + max_hold_days
    are snapshotted at open-time so a later prediction edit doesn't move
    the trigger on this already-open position."""
    with _conn() as c:
        cur = c.execute(
            """
            INSERT INTO trades (
                strategy, ticker, direction, opened_at, open_price,
                open_weight, transaction_cost,
                target_price, stop_loss_price, max_hold_days, opened_via
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                strategy,
                ticker,
                direction,
                opened_at,
                open_price,
                open_weight,
                transaction_cost,
                target_price,
                stop_loss_price,
                max_hold_days,
                opened_via,
            ),
        )
        return cur.lastrowid or 0


def close_trade(
    trade_id: int,
    *,
    closed_at: str,
    close_price: float,
    realized_pnl: float,
    close_reason: str,
    closed_ts: float | None = None,
) -> None:
    """Close a trade. closed_at is the trading-date 'YYYY-MM-DD';
    closed_ts is the wall-clock UTC epoch (set when an intraday
    trigger fires, NULL when close is part of an EOD batch)."""
    with _conn() as c:
        c.execute(
            """
            UPDATE trades
               SET closed_at = ?, close_price = ?, realized_pnl = ?,
                   close_reason = ?, closed_ts = ?
             WHERE id = ?
            """,
            (closed_at, close_price, realized_pnl, close_reason, closed_ts, trade_id),
        )


def list_open_trades(strategy: str) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM trades WHERE strategy = ? AND closed_at IS NULL "
            "ORDER BY opened_at DESC",
            (strategy,),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]


def list_trades(
    strategy: str, *, limit: int = 100, only_closed: bool = False
) -> list[dict]:
    sql = "SELECT * FROM trades WHERE strategy = ?"
    if only_closed:
        sql += " AND closed_at IS NOT NULL"
    sql += " ORDER BY COALESCE(closed_at, opened_at) DESC LIMIT ?"
    with _conn() as c:
        rows = c.execute(sql, (strategy, int(limit))).fetchall()
        return [_row_to_dict(r) for r in rows]


# ── market caps (Nifty 50 universe) ────────────────────────────────


def upsert_market_cap(
    ticker: str, market_cap: float, sector: str | None = None
) -> None:
    with _conn() as c:
        c.execute(
            """
            INSERT INTO nifty50_universe (ticker, name, sector, market_cap, refreshed_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                market_cap   = excluded.market_cap,
                sector       = COALESCE(excluded.sector, nifty50_universe.sector),
                refreshed_at = excluded.refreshed_at
            """,
            (ticker, None, sector, market_cap, time.time()),
        )


def get_market_caps(tickers: Iterable[str] | None = None) -> dict[str, dict]:
    """Return {ticker: {market_cap, refreshed_at, sector}} for the
    given tickers (default: all rows in the table)."""
    with _conn() as c:
        if tickers is None:
            rows = c.execute("SELECT * FROM nifty50_universe").fetchall()
        else:
            tickers = list(tickers)
            placeholders = ",".join("?" * len(tickers))
            rows = c.execute(
                f"SELECT * FROM nifty50_universe WHERE ticker IN ({placeholders})",
                tickers,
            ).fetchall()
        return {r["ticker"]: _row_to_dict(r) for r in rows}


# ── derived helpers for the dashboard ──────────────────────────────


def portfolio_overview(strategy: str) -> dict:
    """Single-payload header stats for Surface A.

    Combines latest snapshot + all-time aggregations (sharpe,
    max-drawdown) into one dict the UI can render without
    coordinating multiple queries.
    """
    from . import STARTING_CAPITAL

    latest = latest_snapshot(strategy)
    snaps = list_snapshots(strategy)
    if not latest:
        return {
            "strategy": strategy,
            "equity_value": STARTING_CAPITAL,
            "cash": STARTING_CAPITAL,
            "starting_capital": STARTING_CAPITAL,
            "all_time_return_pct": 0.0,
            "today": {
                "pnl": 0.0,
                "pnl_pct": 0.0,
                "transaction_costs": 0.0,
            },
            "sharpe_30d": None,
            "max_drawdown": 0.0,
            "exposure": {
                "gross": 0.0,
                "net": 0.0,
                "n_long": 0,
                "n_short": 0,
                "n_neutral": 0,
            },
            "as_of": None,
            "n_snapshots": 0,
        }

    # All-time return
    all_time_return = (latest["equity_value"] - STARTING_CAPITAL) / STARTING_CAPITAL

    # 30-day Sharpe (annualised). Skip if fewer than 5 obs.
    sharpe_30d: float | None = None
    if len(snaps) >= 5:
        tail = snaps[-30:]
        rets = [
            s["daily_return_pct"] for s in tail if s["daily_return_pct"] is not None
        ]
        if rets:
            import statistics

            mean = statistics.fmean(rets)
            std = statistics.pstdev(rets) if len(rets) > 1 else 0.0
            sharpe_30d = (mean / std) * (252**0.5) if std > 0 else None

    # Max drawdown over the full history
    peak = STARTING_CAPITAL
    max_dd = 0.0
    for s in snaps:
        v = s["equity_value"]
        if v > peak:
            peak = v
        dd = (v - peak) / peak if peak > 0 else 0.0
        if dd < max_dd:
            max_dd = dd

    return {
        "strategy": strategy,
        "equity_value": latest["equity_value"],
        "cash": latest["cash"],
        "starting_capital": STARTING_CAPITAL,
        "all_time_return_pct": all_time_return,
        "today": {
            "pnl": latest["daily_pnl"],
            "pnl_pct": latest["daily_return_pct"],
            "transaction_costs": latest["transaction_costs"],
        },
        "sharpe_30d": sharpe_30d,
        "max_drawdown": max_dd,
        "exposure": {
            "gross": latest["gross_exposure"],
            "net": latest["net_exposure"],
            "n_long": latest["n_long"],
            "n_short": latest["n_short"],
            "n_neutral": latest["n_neutral"],
        },
        "as_of": latest["date"],
        "n_snapshots": len(snaps),
    }
