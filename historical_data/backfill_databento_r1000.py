#!/usr/bin/env python3
"""
Databento Russell 1000 daily OHLCV backfill (2024 -> cutoff), resume-safe, Parquet output.

What it does
- Reads metadata/universe_r1000_asof_20260201.csv (single column: symbol)
- Downloads daily bars from Databento (dataset=EQUS.SUMMARY, schema=ohlcv-1d)
- Writes Parquet partitioned by symbol/year:
    data/bars_1d/symbol=AAPL/year=2024.parquet
- Generates:
    metadata/coverage_report.csv (first/last date, rows per symbol)
    metadata/gaps_report.csv     (missing business days within each symbol’s coverage)

Requirements
- pip install databento pandas pyarrow python-dateutil

Auth
- Set env var: DATABENTO_API_KEY="..."
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Dict, Tuple

import pandas as pd
import databento as db


# ---------------------------
# Config defaults
# ---------------------------

DEFAULT_DATASET = "EQUS.SUMMARY"
DEFAULT_SCHEMA = "ohlcv-1d"
DEFAULT_START = date(2024, 7, 1)

# Databento limit: up to 2,000 symbols per request (docs)
DEFAULT_SYMBOL_BATCH = 500

# Keep output vendor-agnostic and small
KEEP_COLS = ["date", "open", "high", "low", "close", "volume", "symbol"]


# ---------------------------
# Helpers
# ---------------------------

def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def atomic_write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(out_path)


def normalize_symbol(sym: str) -> str:
    sym = str(sym).strip().upper()
    sym = re.sub(r"\s+", " ", sym)
    if sym in {"", "NAN", "NONE", "NULL", "NA", "N/A"}:
        return ""
    return sym


def read_universe_csv(path: Path) -> List[str]:
    df = pd.read_csv(path)
    if "symbol" not in df.columns:
        raise ValueError(f"Universe file must have 'symbol' column. Found: {df.columns.tolist()}")
    syms = [normalize_symbol(x) for x in df["symbol"].tolist()]
    syms = [s for s in syms if s]
    # de-dupe while preserving order
    seen = set()
    out = []
    for s in syms:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def year_ranges(start: date, end_exclusive: date) -> List[Tuple[int, date, date]]:
    """Return [(year, start, end_exclusive), ...] covering [start, end_exclusive)."""
    ranges = []
    y = start.year
    while True:
        y_start = date(y, 1, 1)
        y_end = date(y + 1, 1, 1)
        seg_start = max(start, y_start)
        seg_end = min(end_exclusive, y_end)
        if seg_start < seg_end:
            ranges.append((y, seg_start, seg_end))
        if y_end >= end_exclusive:
            break
        y += 1
    return ranges


def chunked(xs: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def coerce_ohlcv_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make a consistent frame with date + ohlcv + volume + symbol.
    Databento .to_df() often yields a DatetimeIndex OR a ts_event column.
    """
    out = df.copy()

    # Ensure we have a symbol column
    if "symbol" not in out.columns:
        # some outputs may use 'raw_symbol' or similar; try best effort
        for cand in ("raw_symbol", "sym", "ticker"):
            if cand in out.columns:
                out["symbol"] = out[cand]
                break
    if "symbol" not in out.columns:
        raise ValueError("Databento response missing 'symbol' column after to_df()")

    # Build a 'date' column from index or ts_event
    if isinstance(out.index, pd.DatetimeIndex):
        dt_index = out.index
    elif "ts_event" in out.columns:
        ts = out["ts_event"]
        # Databento timestamps are typically ns; but be robust by magnitude.
        # ns since epoch ~ 1e18; ms ~ 1e12; s ~ 1e9
        m = int(pd.to_numeric(ts.iloc[0], errors="coerce"))
        if m > 10**16:
            dt_index = pd.to_datetime(ts, unit="ns", utc=True)
        elif m > 10**13:
            dt_index = pd.to_datetime(ts, unit="ms", utc=True)
        else:
            dt_index = pd.to_datetime(ts, unit="s", utc=True)
    else:
        raise ValueError("Cannot determine timestamps: no DatetimeIndex and no ts_event column")

    out["date"] = pd.to_datetime(dt_index).date

    # Standardize expected OHLCV field names
    # (Databento OHLCV schema uses open/high/low/close/volume)
    needed = {"open", "high", "low", "close", "volume"}
    missing = needed - set(out.columns)
    if missing:
        raise ValueError(f"Missing expected OHLCV columns: {sorted(missing)}. Columns: {out.columns.tolist()}")

    out["symbol"] = out["symbol"].astype(str).map(normalize_symbol)
    out = out[out["symbol"] != ""]

    out = out[KEEP_COLS].sort_values(["symbol", "date"]).reset_index(drop=True)
    return out


def parquet_path(root: Path, symbol: str, year: int) -> Path:
    # Hive-style partitions:
    return root / f"symbol={symbol}" / f"year={year}.parquet"


def existing_coverage(root: Path, symbol: str) -> Tuple[date | None, date | None]:
    """Return (min_date, max_date) for symbol if any parquet exists, else (None, None)."""
    sym_dir = root / f"symbol={symbol}"
    if not sym_dir.exists():
        return None, None
    files = sorted(sym_dir.glob("year=*.parquet"))
    if not files:
        return None, None
    mins = []
    maxs = []
    for f in files:
        try:
            df = pd.read_parquet(f, columns=["date"])
            if len(df):
                mins.append(min(df["date"]))
                maxs.append(max(df["date"]))
        except Exception:
            continue
    if not mins:
        return None, None
    return min(mins), max(maxs)


def write_reports(
    bars_root: Path,
    universe: List[str],
    out_coverage: Path,
    out_gaps: Path,
) -> None:
    """
    coverage_report: symbol,first_date,last_date,row_count
    gaps_report: symbol,missing_date
    """
    coverage_rows = []
    gaps_rows = []

    for sym in universe:
        sym_dir = bars_root / f"symbol={sym}"
        files = sorted(sym_dir.glob("year=*.parquet"))
        if not files:
            continue

        all_dates = []
        row_count = 0
        for f in files:
            df = pd.read_parquet(f, columns=["date"])
            if len(df):
                all_dates.append(df["date"])
                row_count += len(df)

        if not all_dates:
            continue

        dates = pd.concat(all_dates).drop_duplicates().sort_values()
        first = dates.iloc[0]
        last = dates.iloc[-1]

        coverage_rows.append(
            {"symbol": sym, "first_date": first, "last_date": last, "row_count": row_count}
        )

        # Gaps: use business-day calendar (Mon-Fri). (Holidays will appear as "gaps"—that’s okay.)
        expected = pd.bdate_range(first, last).date
        actual = set(dates.tolist())
        for d in expected:
            if d not in actual:
                gaps_rows.append({"symbol": sym, "missing_date": d})

    ensure_dir(out_coverage.parent)
    pd.DataFrame(coverage_rows).sort_values(["symbol"]).to_csv(out_coverage, index=False)
    pd.DataFrame(gaps_rows).sort_values(["symbol", "missing_date"]).to_csv(out_gaps, index=False)


# ---------------------------
# Main backfill
# ---------------------------

@dataclass
class RunConfig:
    dataset: str
    schema: str
    start: date
    end_exclusive: date
    symbol_batch: int
    universe_csv: Path
    bars_root: Path
    meta_root: Path
    estimate_only: bool


def estimate_cost(client: db.Historical, cfg: RunConfig, symbols: List[str]) -> float:
    """
    Use Databento metadata.get_cost to estimate request cost (in USD).
    We estimate per (year_chunk × symbol_batch) request and sum.
    """
    total = 0.0
    for (yr, seg_start, seg_end) in year_ranges(cfg.start, cfg.end_exclusive):
        for batch in chunked(symbols, cfg.symbol_batch):
            cost = client.metadata.get_cost(
                dataset=cfg.dataset,
                schema=cfg.schema,
                symbols=batch,
                start=seg_start.isoformat(),
                end=seg_end.isoformat(),
            )
            total += float(cost)
    return total


def backfill(cfg: RunConfig) -> None:
    universe = read_universe_csv(cfg.universe_csv)
    print(f"Universe loaded: {len(universe)} symbols")

    client = db.Historical()  # uses DATABENTO_API_KEY env var

    # Optional pre-flight: list schemas (helps catch typos early)
    try:
        schemas = client.list_schemas(cfg.dataset)
        if cfg.schema not in [str(s) for s in schemas]:
            print(f"WARNING: schema {cfg.schema!r} not in list_schemas({cfg.dataset}). Found: {schemas}")
    except Exception as e:
        print(f"NOTE: list_schemas check skipped due to error: {e}")

    if cfg.estimate_only:
        est = estimate_cost(client, cfg, universe)
        print(f"Estimated total cost for full backfill: ${est:,.2f}")
        return

    bars_root = cfg.bars_root
    ensure_dir(bars_root)
    ensure_dir(cfg.meta_root)

    # Determine which (symbol, year) are missing so we don't re-download
    missing_pairs: List[Tuple[str, int, date, date]] = []
    for sym in universe:
        for (yr, seg_start, seg_end) in year_ranges(cfg.start, cfg.end_exclusive):
            out_path = parquet_path(bars_root, sym, yr)
            if out_path.exists():
                continue
            missing_pairs.append((sym, yr, seg_start, seg_end))

    print(f"Missing partitions to fetch: {len(missing_pairs)} (symbol/year files)")

    # Fetch by year-range and symbol-batches to reduce number of API calls,
    # but still write one parquet per symbol/year.
    # We group missing pairs by year segment to pull multiple symbols at once.
    segs = year_ranges(cfg.start, cfg.end_exclusive)
    seg_by_year = {yr: (seg_start, seg_end) for (yr, seg_start, seg_end) in segs}

    missing_by_year: Dict[int, List[str]] = {}
    for sym, yr, _, _ in missing_pairs:
        missing_by_year.setdefault(yr, []).append(sym)

    for yr in sorted(missing_by_year.keys()):
        seg_start, seg_end = seg_by_year[yr]
        syms = missing_by_year[yr]
        # de-dupe (if any)
        syms = list(dict.fromkeys(syms))
        print(f"\n=== Year {yr}: need {len(syms)} symbols, range [{seg_start} .. {seg_end}) ===")

        for batch in chunked(syms, cfg.symbol_batch):
            print(f"Fetching {len(batch)} symbols for {yr}...")
            store = client.timeseries.get_range(
                dataset=cfg.dataset,
                schema=cfg.schema,
                symbols=batch,
                stype_in="raw_symbol",
                # stype_out omitted -> defaults to instrument_id for historical requests
                start=seg_start.isoformat(),
                end=seg_end.isoformat(),
            )
            df_raw = store.to_df(map_symbols=True)  # adds 'symbol' mapped from instrument_id

            df_raw = store.to_df()
            df = coerce_ohlcv_df(df_raw)

            # Write one parquet per symbol/year
            for sym, sdf in df.groupby("symbol", sort=False):
                out_path = parquet_path(bars_root, sym, yr)
                if out_path.exists():
                    continue  # just in case
                atomic_write_parquet(sdf, out_path)

    # Reports
    coverage_path = cfg.meta_root / "coverage_report.csv"
    gaps_path = cfg.meta_root / "gaps_report.csv"
    write_reports(bars_root, universe, coverage_path, gaps_path)
    print("\nDone.")
    print(f"Wrote reports:\n- {coverage_path}\n- {gaps_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="metadata/universe_r1000_asof_20260201.csv")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--schema", default=DEFAULT_SCHEMA)
    ap.add_argument("--start", default=DEFAULT_START.isoformat(), help="YYYY-MM-DD")
    ap.add_argument(
        "--end",
        default=None,
        help="YYYY-MM-DD (exclusive). Default: tomorrow (so 'today' is included if available).",
    )
    ap.add_argument("--symbol-batch", type=int, default=DEFAULT_SYMBOL_BATCH)
    ap.add_argument("--bars-root", default="data/bars_1d")
    ap.add_argument("--meta-root", default="metadata")
    ap.add_argument("--estimate-only", action="store_true", help="Estimate Databento cost and exit (no downloads).")

    args = ap.parse_args()

    start = parse_date(args.start)
    if args.end:
        end_excl = parse_date(args.end)
    else:
        end_excl = date.today()

    cfg = RunConfig(
        dataset=args.dataset,
        schema=args.schema,
        start=start,
        end_exclusive=end_excl,
        symbol_batch=args.symbol_batch,
        universe_csv=Path(args.universe),
        bars_root=Path(args.bars_root),
        meta_root=Path(args.meta_root),
        estimate_only=args.estimate_only,
    )

    backfill(cfg)


if __name__ == "__main__":
    main()
