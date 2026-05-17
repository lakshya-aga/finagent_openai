"""Paper trading — Nifty 50 daily directional book.

End-to-end pipeline:

  predictions(date, ticker, direction)
        │
        ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ daily EOD close routine                                     │
   │                                                             │
   │   for each strategy in {equal_weight, market_cap}:          │
   │     1. compute target weights from today's directions       │
   │     2. diff vs yesterday's positions → open + close trades  │
   │     3. mark-to-market via close prices (yfinance)           │
   │     4. write portfolio_snapshots + position_snapshots       │
   │                                                             │
   │   ₹20 transaction cost per opened/closed trade              │
   └─────────────────────────────────────────────────────────────┘
        │
        ▼
   dashboard at /app/paper-trading

Money math is simple by design — daily portfolio_return = Σ wᵢ × rᵢ.
SL/TP / intra-day stop triggers are NOT modeled in v1 (every
position is held until the next day's predictions overwrite it).
Cash is folded into equity_value — we don't separately track margin
since this is a notional paper book.

Public API:
  - universe.NIFTY50_TICKERS               : list[str]
  - universe.get_sector(ticker)            : str | None
  - universe.refresh_market_caps()         : async
  - engine.run_eod_close(date)             : async — daily snapshot
  - store.list_predictions(date)           : reader for the UI
  - store.list_positions(strategy, date)
  - store.list_snapshots(strategy, start, end)
  - store.portfolio_overview(strategy)
  - predictions.record_prediction(...)
  - predictions.seed_from_debates(date)    : populate predictions
                                              from existing debate
                                              verdicts (no LLM cost)
"""

# NB: constants are defined BEFORE the submodule imports so engine.py
# can `from . import STARTING_CAPITAL` without hitting a circular-import
# error (the submodules read them at module-load time).
STARTING_CAPITAL = 100_000.0       # INR — anchor for equity curve
TRANSACTION_COST = 20.0            # INR per position change

STRATEGIES = ("equal_weight", "market_cap")

from . import universe, store, engine, predictions, schema  # noqa: E402, F401
