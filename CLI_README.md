# Common Console Commands

Quick reference for the Python commands used most often in this project.

## Paper Trading

Run the live paper-trading strategy using the combined Donchian + pullback system.

- **From directory:**
  - `/Users/mmorr/Documents/GitHub/algo-trading/engine/live`
- **Command:**
  - `python3 run_ib_paper.py --port 4002 --max-positions 10 --target-pct 0.1 --n-symbols 50`

Parameters:
- `--port 4002` — IB Gateway paper port (change to 7497 for TWS paper).
- `--max-positions 10` — Maximum concurrent positions.
- `--target-pct 0.1` — Target fraction of equity per new position (10%).
- `--n-symbols 50` — Number of symbols sampled from the universe.

## Daily Historical Backfill (IBKR)

Backfill daily OHLCV candles from IBKR into the Parquet store.

- **From directory:**
  - `/Users/mmorr/Documents/GitHub/algo-trading/historical_data`
- **Command (explicit dates):**
  - `python3 backfill_ibkr_daily.py --start 2026-02-09 --end 2026-02-14`

Notes:
- `--start` is inclusive, `--end` is exclusive, so the above covers 2026-02-09 through 2026-02-13.
- If you omit `--start` and `--end`, the script defaults to `[yesterday, today)` for easy daily scheduling:
  - `python3 backfill_ibkr_daily.py --port 4002`

## Tips

- Ensure IB Gateway or TWS is running and logged into the **paper** account with API enabled before running any of the above.
- You can adjust arguments (e.g., `--max-positions`, `--target-pct`, date range) as your research or risk preferences change.
