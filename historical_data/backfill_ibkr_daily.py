#!/usr/bin/env python3
"""Backfill missing daily OHLCV bars from IBKR into Parquet layout.

- Uses IB's historical data API via ib_insync.
- Writes to bars_ohlcv_1d/symbol=SYM/year=YYYY.parquet
- Intended for small incremental top-ups (e.g., last few missing days).

Requirements:
  pip install ib-insync pandas pyarrow

You must have IB Gateway or TWS running and logged in (paper or live),
with API enabled and the configured host/port open.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Dict

import pandas as pd
from ib_insync import IB, Stock, util


ROOT = Path(__file__).resolve().parent
BARS_ROOT_DEFAULT = ROOT / "bars_ohlcv_1d"
UNIVERSE_CSV_DEFAULT = ROOT / "metadata" / "universe_r1000_asof_20260201.csv"


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def atomic_write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(out_path)


def load_universe(path: Path) -> List[str]:
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


@dataclass
class BackfillConfig:
    host: str
    port: int
    client_id: int
    bars_root: Path
    universe_csv: Path
    start: date
    end_exclusive: date


def fetch_symbol_daily(ib: IB, symbol: str, start: date, end_exclusive: date) -> pd.DataFrame:
    """Fetch daily bars for [start, end_exclusive) from IB for one symbol."""
    contract = Stock(symbol, "SMART", "USD")

    # Request a bit more than needed so we can filter exactly.
    days = max(1, (end_exclusive - start).days + 2)
    bars = ib.reqHistoricalData(
        contract,
        endDateTime=end_exclusive.strftime("%Y%m%d 23:59:59"),
        durationStr=f"{days} D",
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=True,
        formatDate=1,
    )
    if not bars:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "symbol"])

    df = util.df(bars)
    # IB daily bars typically have a 'date' column like 'YYYYMMDD' or datetime
    df["date"] = pd.to_datetime(df["date"]).dt.date

    out = pd.DataFrame(
        {
            "date": df["date"],
            "open": df["open"].astype(float),
            "high": df["high"].astype(float),
            "low": df["low"].astype(float),
            "close": df["close"].astype(float),
            "volume": df["volume"].astype(float),
            "symbol": symbol,
        }
    )

    # Filter to the exact window [start, end_exclusive)
    out = out[(out["date"] >= start) & (out["date"] < end_exclusive)].copy()
    return out.sort_values("date").reset_index(drop=True)


def merge_into_parquet(bars_root: Path, sym: str, df_new: pd.DataFrame) -> None:
    if df_new.empty:
        return

    df_new = df_new.copy()

    # Group by year and merge per-year parquet
    df_new["year"] = pd.to_datetime(df_new["date"]).dt.year
    for year, chunk in df_new.groupby("year"):
        out_path = bars_root / f"symbol={sym}" / f"year={year}.parquet"
        if out_path.exists():
            existing = pd.read_parquet(out_path)
            # Ensure same columns
            cols = ["date", "open", "high", "low", "close", "volume", "symbol"]
            existing = existing[cols]
            combined = pd.concat([existing, chunk[cols]], ignore_index=True)
            combined = combined.drop_duplicates(subset=["date"], keep="last").sort_values("date")
        else:
            combined = chunk.drop(columns=["year"]).sort_values("date")
        atomic_write_parquet(combined, out_path)


def backfill(cfg: BackfillConfig) -> None:
    ib = IB()
    ib.connect(cfg.host, cfg.port, clientId=cfg.client_id)

    try:
        universe = load_universe(cfg.universe_csv)
        print(f"Backfilling {len(universe)} symbols from {cfg.start} to {cfg.end_exclusive} (exclusive)")

        for i, sym in enumerate(universe, start=1):
            print(f"[{i}/{len(universe)}] {sym} ...", end="", flush=True)
            try:
                df_sym = fetch_symbol_daily(ib, sym, cfg.start, cfg.end_exclusive)
                merge_into_parquet(cfg.bars_root, sym, df_sym)
                print(f" ok ({len(df_sym)} rows)")
            except Exception as e:
                print(f" error: {e}")
    finally:
        ib.disconnect()


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Backfill daily OHLCV bars from IBKR into Parquet store.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=4002, help="IB Gateway/TWS port (e.g., 4002 for paper Gateway, 7497 for paper TWS)")
    ap.add_argument("--client-id", type=int, default=9)
    ap.add_argument("--bars-root", type=str, default=str(BARS_ROOT_DEFAULT))
    ap.add_argument("--universe-csv", type=str, default=str(UNIVERSE_CSV_DEFAULT))
    ap.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD (inclusive); default = yesterday")
    ap.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD (exclusive); default = today")

    args = ap.parse_args()

    # If dates are not provided, default to [yesterday, today)
    today = date.today()
    if args.start is None:
        start_d = today - timedelta(days=1)
    else:
        start_d = parse_date(args.start)

    if args.end is None:
        end_d = today
    else:
        end_d = parse_date(args.end)

    cfg = BackfillConfig(
        host=args.host,
        port=args.port,
        client_id=args.client_id,
        bars_root=Path(args.bars_root),
        universe_csv=Path(args.universe_csv),
        start=start_d,
        end_exclusive=end_d,
    )

    backfill(cfg)


if __name__ == "__main__":
    main()
