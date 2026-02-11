from pathlib import Path
import random
import pandas as pd

from backtest import BacktestConfig, Backtester
from strategies.donchian.donchian  import donchian_breakout
from strategies.pullback_sma.pullback_sma import sma_pullback_continuation
from strategies.combine.combine_caps import combine_with_caps
from metrics import (
    summarize_performance, 
    print_summary,
    compute_exposure,
    print_exposure
)


# ---- Paths (edit if needed) ----
BARS_ROOT = "../historical_data/bars_ohlcv_1d"
UNIVERSE_CSV = "../historical_data/metadata/universe_r1000_asof_20260201.csv"  # adjust if yours lives elsewhere

# ---- Test size ----
N = 50                 # try 25, 50, 100
RANDOM_SAMPLE = True   # False = take first N
SEED = 42

# ---- Backtest window ----
START = "2024-07-01"
END   = "2026-02-04"   # exclusive

# ---- Portfolio settings ----
INITIAL_CASH = 800
MAX_POSITIONS = 10
TARGET_PCT = 1.0 / MAX_POSITIONS  # equal-weight when fully invested

# ---- Costs ----
COMMISSION = 1.00
SLIPPAGE_BPS = 1.0


def load_universe(path: str) -> list[str]:
    df = pd.read_csv(path)
    if "symbol" not in df.columns:
        raise ValueError(f"Universe CSV missing 'symbol' column: {df.columns.tolist()}")
    syms = (
        df["symbol"].astype(str).str.strip().str.upper()
        .replace({"": None, "NAN": None, "NONE": None, "NULL": None})
        .dropna()
        .drop_duplicates()
        .tolist()
    )
    return syms


def main():
    universe = load_universe(UNIVERSE_CSV)
    if N > len(universe):
        raise ValueError(f"N={N} > universe size {len(universe)}")

    if RANDOM_SAMPLE:
        random.seed(SEED)
        symbols = random.sample(universe, N)
    else:
        symbols = universe[:N]

    print(f"Running Strategy on {len(symbols)} symbols")
    print(f"Date range: {START} → {END} (exclusive)")
    print(f"Max positions: {MAX_POSITIONS} | Target pct: {TARGET_PCT:.4f}")

    cfg = BacktestConfig(
        bars_root=BARS_ROOT,
        symbols=symbols,
        start=START,
        end=END,
        initial_cash=INITIAL_CASH,
        commission_per_trade=COMMISSION,
        slippage_bps=SLIPPAGE_BPS,
        max_positions=MAX_POSITIONS,
        target_position_pct=TARGET_PCT,
    )
    
    donchian = donchian_breakout(
        entry_lookback=20,
        exit_lookback=10,
    )
    
    pullback = sma_pullback_continuation(
        trend_len=50,
        pullback_len=20,
        slope_lookback=10,
        min_slope=0.0,
    )
    
    strategy = combine_with_caps(donchian, pullback, tag1='donchian', tag2='pullback', max_s1_positions=4, max_s2_positions=6)

    bt = Backtester(cfg, strategy)
    res = bt.run()

    summary, roundtrips = summarize_performance(res.trades, res.equity)
    print_summary(summary)
    
    expo = compute_exposure(res.equity, res.positions)
    print_exposure(expo)

    outdir = Path("results_pullback_sma_portfolio")
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"symbol": symbols}).to_csv(outdir / "symbols_used.csv", index=False)
    res.trades.to_csv(outdir / "trades.csv", index=False)
    res.equity.to_csv(outdir / "equity.csv", index=False)
    roundtrips.to_csv(outdir / "roundtrips.csv", index=False)

    print(f"Wrote outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
