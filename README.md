# Algo Trading

A Python algorithmic trading framework supporting backtesting and live paper trading via Interactive Brokers (IBKR). Includes multiple long-only equity strategies, a portfolio-level backtest engine, and historical data management utilities.

## Repository Structure

```
algo-trading/
├── engine/
│   ├── backtest.py          # Core backtest engine (next-open fill model)
│   ├── run_strategy.py      # Backtest entry point for combined strategy
│   ├── broker.py            # Broker abstraction
│   ├── datasource.py        # Data source abstraction
│   ├── portfolio.py         # Portfolio management
│   ├── metrics.py           # Performance metrics (CAGR, drawdown, win rate, etc.)
│   ├── strategies/
│   │   ├── donchian/        # Donchian channel breakout strategy
│   │   ├── pullback_sma/    # SMA pullback continuation strategy
│   │   ├── mean_reversion/  # Mean reversion strategy (RSI-based)
│   │   └── combine/         # Combined multi-strategy sleeve
│   └── live/
│       ├── run_ib_paper.py  # Live paper trading runner (IBKR)
│       └── execution/       # IBKR order execution
├── historical_data/
│   ├── backfill_ibkr_daily.py       # Backfill OHLCV from IBKR
│   ├── backfill_databento_r1000.py  # Backfill OHLCV from Databento
│   ├── bars_ohlcv_1d/               # Parquet bar store (partitioned by symbol/year)
│   └── metadata/                    # Universe CSV files
├── live_results/
│   └── fills.csv            # Live paper trade fill log
└── CLI_README.md            # Quick-reference CLI commands
```

## Strategies

| Strategy | Description |
|---|---|
| **Donchian Breakout** | Buys when price breaks the prior N-day high; exits below the prior M-day low. Entries ranked by breakout × trend strength. |
| **SMA Pullback Continuation** | Buys a pullback to a fast SMA while price remains above a slow SMA trend filter. Entries ranked by distance from trend. |
| **Mean Reversion** | Buys oversold conditions (RSI-2 ≤ 5) above the 200-day SMA. Exits when RSI recovers or price reclaims the 20-day SMA. |
| **Combined** | Runs Donchian + Pullback SMA as two sleeves under a shared position cap. |

## Backtest Engine

- **Fill model:** next trading day's open (no lookahead)
- **Long-only** by default
- **Ranked entries:** strategies can return a score alongside a BUY signal; the engine executes sells first, then buys ranked by score descending
- **Data format:** Parquet files partitioned as `bars_ohlcv_1d/symbol=<SYM>/year=<YYYY>.parquet`

### Running a Backtest

```bash
cd engine
python3 run_strategy.py
```

Edit the parameters at the top of `run_strategy.py` to adjust the universe, backtest window, cash, and position sizing.

## Live Paper Trading

Requires IB Gateway or TWS running with API enabled on the paper account.

```bash
cd engine/live
python3 run_ib_paper.py --port 4002 --max-positions 10 --target-pct 0.1 --n-symbols 50
```

| Argument | Default | Description |
|---|---|---|
| `--port` | `4002` | IB Gateway paper port (use `7497` for TWS paper) |
| `--max-positions` | `10` | Maximum concurrent positions |
| `--target-pct` | `0.1` | Fraction of equity per new position (10%) |
| `--n-symbols` | `50` | Number of symbols sampled from the universe |

## Historical Data

### Backfill from IBKR

```bash
cd historical_data
# Explicit date range:
python3 backfill_ibkr_daily.py --start 2026-02-09 --end 2026-02-14

# Default (yesterday → today):
python3 backfill_ibkr_daily.py --port 4002
```

### Backfill from Databento

```bash
cd historical_data
python3 backfill_databento_r1000.py
```

## Dependencies

```bash
pip install pandas pyarrow numpy
```

Additional dependencies for live trading: `ib_insync` (or equivalent IBKR Python API).

## Performance Metrics

The `metrics.py` module computes:
- Total return, CAGR, max drawdown
- Number of trades and round-trips
- Win rate, average win/loss, profit factor
- Average holding period

## Notes

- Ensure IB Gateway or TWS is running and logged in to the **paper** account before running live strategies.
- The universe CSV (`metadata/universe_r1000_asof_*.csv`) must have a `symbol` column.
- All signals are generated on today's close and filled at the next day's open to avoid lookahead bias.
