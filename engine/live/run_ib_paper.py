from __future__ import annotations

from pathlib import Path
import random
from typing import Dict, List, Any
import sys

import pandas as pd


# ---- Paths / imports (robust to CWD) ----
ENGINE_DIR = Path(__file__).resolve().parents[1]
if str(ENGINE_DIR) not in sys.path:
    sys.path.append(str(ENGINE_DIR))

HIST_DIR = ENGINE_DIR.parent / "historical_data"

from backtest import ParquetDailyDataSource, _parse_signal
from strategies.donchian.donchian import donchian_breakout
from strategies.pullback_sma.pullback_sma import sma_pullback_continuation
from strategies.combine.combine_caps import combine_with_caps
from live.execution.ikbr import IBKRExecutor

BARS_ROOT = str(HIST_DIR / "bars_ohlcv_1d")
UNIVERSE_CSV = str(HIST_DIR / "metadata" / "universe_r1000_asof_20260201.csv")

N = 50                 # number of symbols to trade
RANDOM_SAMPLE = True   # False = take first N
SEED = 42

MAX_POSITIONS = 10
TARGET_PCT = 1.0 / MAX_POSITIONS


def load_universe(path: str) -> List[str]:
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


def position_size_shares(equity: float, price: float, target_pct: float, allow_fractional: bool = False) -> int:
    target_value = equity * target_pct
    if target_value <= 0 or price <= 0:
        return 0
    if allow_fractional:
        return int(target_value / price)
    return int(target_value // price)


def run_ib_paper(
    bars_root: str = BARS_ROOT,
    universe_csv: str = UNIVERSE_CSV,
    n_symbols: int = N,
    random_sample: bool = RANDOM_SAMPLE,
    seed: int = SEED,
    max_positions: int = MAX_POSITIONS,
    target_pct: float = TARGET_PCT,
    assumed_cash: float | None = None,
    host: str = "127.0.0.1",
    port: int = 7497,
    client_id: int = 7,
) -> None:
    universe = load_universe(universe_csv)
    if n_symbols > len(universe):
        raise ValueError(f"n_symbols={n_symbols} > universe size {len(universe)}")

    if random_sample:
        random.seed(seed)
        symbols = random.sample(universe, n_symbols)
    else:
        symbols = universe[:n_symbols]

    print(f"IB paper run on {len(symbols)} symbols")
    print(f"Max positions: {max_positions} | Target pct: {target_pct:.4f}")

    data = ParquetDailyDataSource(bars_root)

    with IBKRExecutor(host=host, port=port, client_id=client_id) as ex:
        ib_positions: Dict[str, int] = ex.get_positions()

        # Load symbol data and get latest close for equity estimate
        symbol_bars: Dict[str, pd.DataFrame] = {}
        last_close: Dict[str, float] = {}

        for sym in symbols:
            df = data.load_symbol(sym, start="2010-01-01", end="2099-12-31")
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            symbol_bars[sym] = df
            last_close[sym] = float(df["close"].iloc[-1])

        # Decide cash source: IB account by default, or CLI override
        if assumed_cash is None:
            cash = ex.get_cash()
            cash_source = "IB account TotalCashValue"
        else:
            cash = float(assumed_cash)
            cash_source = "CLI assumed_cash"

        # Simple equity estimate from cash + IB positions marked to last close
        eq = float(cash)
        for sym, qty in ib_positions.items():
            px = last_close.get(sym)
            if px is not None and qty != 0:
                eq += qty * px

        print(f"Approx equity for sizing: {eq:,.2f} (cash source={cash_source}, cash={cash:,.2f})")

        # Build combined strategy
        donchian = donchian_breakout(entry_lookback=20, exit_lookback=10)
        pullback = sma_pullback_continuation(
            trend_len=50,
            pullback_len=20,
            slope_lookback=10,
            min_slope=0.0,
        )
        strategy = combine_with_caps(donchian, pullback, tag1="donchian", tag2="pullback", max_s1_positions=4, max_s2_positions=6)

        # Current open positions count (for global max_positions constraint)
        open_count = sum(1 for q in ib_positions.values() if q != 0)

        sell_list: List[Dict[str, Any]] = []
        buy_candidates: List[Dict[str, Any]] = []

        for sym, df in symbol_bars.items():
            hist = df
            pos = int(ib_positions.get(sym, 0))
            dt = pd.to_datetime(hist["date"].iloc[-1])
            ctx = {
                "date": dt,
                "symbol": sym,
                "bar": hist.iloc[-1].to_dict(),
                "history": hist,
                "position": pos,
                "entry_tag": None,         # unknown for existing IB positions
                "open_by_tag": {},         # caps will only apply to new entries
                "cash": cash,
                "equity": eq,
            }

            action, score, tag = _parse_signal(strategy(ctx))

            if action == "SELL" and pos > 0:
                sell_list.append({"symbol": sym, "qty": pos})
            elif action == "BUY" and pos == 0:
                px = last_close[sym]
                buy_candidates.append({
                    "score": float(score),
                    "symbol": sym,
                    "est_price": px,
                    "tag": tag,
                })

        # Rank buys by score desc
        buy_candidates.sort(key=lambda d: d["score"], reverse=True)

        orders: List[Dict[str, Any]] = []

        # Sells first (full liquidation for each SELL signal)
        for s in sell_list:
            sym = s["symbol"]
            qty = int(s["qty"])
            if qty <= 0:
                continue
            orders.append({"symbol": sym, "side": "SELL", "qty": qty})
            open_count = max(open_count - 1, 0)

        # Then ranked buys, subject to max_positions
        for b in buy_candidates:
            if open_count >= max_positions:
                break

            sym = b["symbol"]
            px = float(b["est_price"])
            qty = position_size_shares(eq, px, target_pct, allow_fractional=False)
            if qty <= 0:
                continue

            orders.append({"symbol": sym, "side": "BUY", "qty": qty})
            open_count += 1

        if not orders:
            print("No orders to submit.")
            return

        print("Placing MOO orders:")
        for o in orders:
            print(f"  {o['side']} {o['qty']} {o['symbol']}")

        trades = []
        for o in orders:
            tr = ex.place_moo(o["symbol"], o["side"], int(o["qty"]))
            trades.append(tr)

        fills = ex.wait_for_fills(trades, timeout_sec=120)

        outdir = Path("live_results")
        outdir.mkdir(parents=True, exist_ok=True)
        fills_df = pd.DataFrame([
            {
                "symbol": f.symbol,
                "side": f.side,
                "qty": f.qty,
                "avg_fill_price": f.avg_fill_price,
            }
            for f in fills
        ])
        fills_df.to_csv(outdir / "fills.csv", index=False)

        print("Fills:")
        for f in fills:
            print(f"  {f.side} {f.qty} {f.symbol} @ {f.avg_fill_price:.4f}")

        print(f"Wrote fills to: {outdir.resolve() / 'fills.csv'}")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Run combined strategy and send MOO orders to IBKR paper.")
    ap.add_argument("--bars-root", default=BARS_ROOT)
    ap.add_argument("--universe-csv", default=UNIVERSE_CSV)
    ap.add_argument("--n-symbols", type=int, default=N)
    ap.add_argument("--random-sample", action="store_true", default=RANDOM_SAMPLE)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--max-positions", type=int, default=MAX_POSITIONS)
    ap.add_argument("--target-pct", type=float, default=TARGET_PCT)
    ap.add_argument("--assumed-cash", type=float, default=None, help="Override IB cash for sizing; if omitted, use IB TotalCashValue")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7497)
    ap.add_argument("--client-id", type=int, default=7)

    args = ap.parse_args()

    run_ib_paper(
        bars_root=args.bars_root,
        universe_csv=args.universe_csv,
        n_symbols=args.n_symbols,
        random_sample=args.random_sample,
        seed=args.seed,
        max_positions=args.max_positions,
        target_pct=args.target_pct,
        assumed_cash=args.assumed_cash,
        host=args.host,
        port=args.port,
        client_id=args.client_id,
    )


if __name__ == "__main__":
    main()
