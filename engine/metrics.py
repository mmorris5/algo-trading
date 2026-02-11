# metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd


@dataclass
class PerformanceSummary:
    total_return: float
    cagr: float
    max_drawdown: float
    num_trades: int
    num_round_trips: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    avg_hold_days: float
    start_date: str
    end_date: str
    start_equity: float
    end_equity: float


def _to_equity_series(equity_df: pd.DataFrame) -> pd.DataFrame:
    df = equity_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    df["equity"] = df["equity"].astype(float)
    return df


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())  # negative number


def _cagr(start_equity: float, end_equity: float, start_date: pd.Timestamp, end_date: pd.Timestamp) -> float:
    days = (end_date - start_date).days
    if days <= 0 or start_equity <= 0:
        return 0.0
    years = days / 365.25
    return float((end_equity / start_equity) ** (1 / years) - 1.0)


def _round_trips_and_pnl(trades: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Converts trade stream into per-round-trip PnL using FIFO per symbol.
    Assumes long-only with full exits (your engine sells full qty).
    Returns: (roundtrips_df, num_round_trips)

    roundtrips_df columns:
      symbol, entry_date, exit_date, qty, entry_price, exit_price, pnl, hold_days
    """
    if trades is None or trades.empty:
        cols = ["symbol", "entry_date", "exit_date", "qty", "entry_price", "exit_price", "pnl", "hold_days"]
        return pd.DataFrame(columns=cols), 0

    t = trades.copy()
    t["date"] = pd.to_datetime(t["date"])
    t = t.sort_values(["symbol", "date"]).reset_index(drop=True)

    open_pos: Dict[str, Dict[str, Any]] = {}
    rows = []

    for _, r in t.iterrows():
        sym = str(r["symbol"])
        side = str(r["side"]).upper()
        qty = int(r["qty"])
        px = float(r["price"])
        dt = pd.to_datetime(r["date"])

        comm = float(r["commission"]) if "commission" in r and pd.notna(r["commission"]) else 0.0

        if side == "BUY":
            # If we somehow get multiple buys without a sell, treat as overwrite (shouldn't happen in your current engine)
            open_pos[sym] = {
                "entry_date": dt,
                "qty": qty,
                "entry_price": px,
                "entry_commission": comm,
            }
        elif side == "SELL":
            if sym not in open_pos:
                # Unmatched sell — skip (or raise if you prefer strict)
                continue
            entry = open_pos.pop(sym)
            entry_dt = entry["entry_date"]
            hold_days = (dt - entry_dt).days
            entry_px = float(entry["entry_price"])
            entry_qty = int(entry["qty"])
            entry_comm = float(entry.get("entry_commission", 0.0))

            # PnL: qty*(exit-entry) - commissions (entry+exit)
            pnl = entry_qty * (px - entry_px) - (entry_comm + comm)

            rows.append({
                "symbol": sym,
                "entry_date": entry_dt.date().isoformat(),
                "exit_date": dt.date().isoformat(),
                "qty": entry_qty,
                "entry_price": entry_px,
                "exit_price": px,
                "pnl": float(pnl),
                "hold_days": float(hold_days),
            })

    rt = pd.DataFrame(rows)
    return rt, len(rt)


def summarize_performance(trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> Tuple[PerformanceSummary, pd.DataFrame]:
    """
    Returns:
      - PerformanceSummary (key headline metrics)
      - roundtrips_df (per-trade PnL and hold times)
    """
    eq = _to_equity_series(equity_df)
    start_equity = float(eq.iloc[0]["equity"])
    end_equity = float(eq.iloc[-1]["equity"])
    start_date = pd.to_datetime(eq.iloc[0]["date"])
    end_date = pd.to_datetime(eq.iloc[-1]["date"])

    total_return = (end_equity / start_equity) - 1.0 if start_equity > 0 else 0.0
    cagr = _cagr(start_equity, end_equity, start_date, end_date)
    mdd = _max_drawdown(eq["equity"])

    roundtrips_df, num_round_trips = _round_trips_and_pnl(trades_df)

    # Win/Loss stats
    if num_round_trips > 0:
        wins = roundtrips_df[roundtrips_df["pnl"] > 0]["pnl"]
        losses = roundtrips_df[roundtrips_df["pnl"] < 0]["pnl"]

        win_rate = float(len(wins) / num_round_trips)
        avg_win = float(wins.mean()) if len(wins) else 0.0
        avg_loss = float(losses.mean()) if len(losses) else 0.0  # negative
        gross_profit = float(wins.sum()) if len(wins) else 0.0
        gross_loss = float((-losses).sum()) if len(losses) else 0.0
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
        avg_hold_days = float(roundtrips_df["hold_days"].mean()) if len(roundtrips_df) else 0.0
    else:
        win_rate = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        profit_factor = 0.0
        avg_hold_days = 0.0

    summary = PerformanceSummary(
        total_return=float(total_return),
        cagr=float(cagr),
        max_drawdown=float(mdd),
        num_trades=int(0 if trades_df is None else len(trades_df)),
        num_round_trips=int(num_round_trips),
        win_rate=float(win_rate),
        avg_win=float(avg_win),
        avg_loss=float(avg_loss),
        profit_factor=float(profit_factor),
        avg_hold_days=float(avg_hold_days),
        start_date=start_date.date().isoformat(),
        end_date=end_date.date().isoformat(),
        start_equity=float(start_equity),
        end_equity=float(end_equity),
    )

    return summary, roundtrips_df

def print_summary(summary: PerformanceSummary) -> None:
    """
    Nicely formatted console output for PerformanceSummary
    """
    def pct(x): 
        return f"{x*100:6.2f}%"

    def money(x):
        return f"${x:,.2f}"

    print("\n" + "=" * 50)
    print(" BACKTEST PERFORMANCE SUMMARY")
    print("=" * 50)

    print(f"Period:          {summary.start_date} → {summary.end_date}")
    print(f"Start Equity:    {money(summary.start_equity)}")
    print(f"End Equity:      {money(summary.end_equity)}")
    print("-" * 50)

    print(f"Total Return:    {pct(summary.total_return)}")
    print(f"CAGR:            {pct(summary.cagr)}")
    print(f"Max Drawdown:    {pct(summary.max_drawdown)}")
    print("-" * 50)

    print(f"Trades (fills):  {summary.num_trades}")
    print(f"Round Trips:     {summary.num_round_trips}")
    print(f"Win Rate:        {pct(summary.win_rate)}")
    print(f"Avg Win:         {money(summary.avg_win)}")
    print(f"Avg Loss:        {money(summary.avg_loss)}")
    print(f"Profit Factor:   {summary.profit_factor:6.2f}")
    print(f"Avg Hold (days): {summary.avg_hold_days:6.1f}")

    print("=" * 50 + "\n")

def compute_exposure(equity_df: pd.DataFrame, positions_df: pd.DataFrame) -> dict:
    """
    Exposure / time-in-market metrics.

    Assumptions:
    - equity_df has columns: date, equity, cash
    - positions_df has columns: date, symbol, shares
    - One row per (date, symbol) when shares != 0
    """

    eq = equity_df.copy()
    eq["date"] = pd.to_datetime(eq["date"])
    eq = eq.sort_values("date").reset_index(drop=True)

    # Total trading days
    total_days = len(eq)

    # Days with at least one open position
    pos_days = (
        positions_df.groupby("date")
        .size()
        .rename("num_positions")
        .reset_index()
    )
    pos_days["date"] = pd.to_datetime(pos_days["date"])

    eq = eq.merge(pos_days, on="date", how="left")
    eq["num_positions"] = eq["num_positions"].fillna(0)

    invested_days = int((eq["num_positions"] > 0).sum())
    pct_days_in_market = invested_days / total_days if total_days else 0.0

    avg_positions = float(eq["num_positions"].mean())
    avg_cash = float(eq["cash"].mean())
    avg_equity = float(eq["equity"].mean())

    avg_cash_pct = avg_cash / avg_equity if avg_equity > 0 else 0.0
    avg_exposure_pct = 1.0 - avg_cash_pct

    return {
        "total_days": total_days,
        "invested_days": invested_days,
        "pct_days_in_market": pct_days_in_market,
        "avg_positions": avg_positions,
        "avg_cash_pct": avg_cash_pct,
        "avg_exposure_pct": avg_exposure_pct,
    }

def print_exposure(expo: dict) -> None:
    def pct(x): 
        return f"{x*100:6.2f}%"

    print("\n" + "=" * 50)
    print(" MARKET EXPOSURE SUMMARY")
    print("=" * 50)

    print(f"Trading days:        {expo['total_days']}")
    print(f"Days invested:       {expo['invested_days']}")
    print(f"Time in market:      {pct(expo['pct_days_in_market'])}")
    print("-" * 50)
    print(f"Avg positions held:  {expo['avg_positions']:.2f}")
    print(f"Avg exposure:        {pct(expo['avg_exposure_pct'])}")
    print(f"Avg cash drag:       {pct(expo['avg_cash_pct'])}")

    print("=" * 50 + "\n")
