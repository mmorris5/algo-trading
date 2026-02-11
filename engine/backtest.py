# backtest.py
"""
Minimal daily-bar backtest runner (next-day open fills) over Parquet bars.

Assumptions / design choices
- Daily bars only
- Strategy generates signals using today's bar (no lookahead)
- Orders fill on NEXT trading day's OPEN (next-open fill model)
- Long-only by default (easy to extend)
- Parquet layout supported:
    data/bars_1d/symbol=AAPL/year=2010.parquet
    data/bars_1d/symbol=AAPL/year=2011.parquet
  or any files under symbol=<SYM>/year=*.parquet

Ranked entries upgrade
- Strategies may return:
    None
    "BUY" / "SELL"
    ("BUY", score) / ("SELL", score)
    {"action": "BUY", "score": 1.23}
- Engine executes SELLS first (next open), then BUYS ranked by score desc.

Dependencies:
  pip install pandas pyarrow
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any, Tuple

import pandas as pd


def _parse_signal(sig):
    """
    Backward-compatible signal parser.

    Accepts:
      - None
      - "BUY" / "SELL"
      - ("BUY", score) / ("SELL", score)
      - ("BUY", score, tag) / ("SELL", score, tag)   <-- tag used only for BUY
      - {"action": "BUY", "score": 1.23, "tag": "mr"} <-- tag optional

    Returns: (action: str|None, score: float, tag: Optional[str])
    """
    if sig is None:
        return None, 0.0, None

    if isinstance(sig, str):
        s = sig.upper()
        if s in ("BUY", "SELL"):
            return s, 0.0, None
        raise ValueError(f"Invalid signal string: {sig!r}")

    if isinstance(sig, tuple):
        if len(sig) == 2:
            action, score = sig
            if not isinstance(action, str):
                raise ValueError(f"Invalid signal tuple action: {sig!r}")
            a = action.upper()
            if a not in ("BUY", "SELL"):
                raise ValueError(f"Invalid signal tuple action: {sig!r}")
            return a, float(score), None

        if len(sig) == 3:
            action, score, tag = sig
            if not isinstance(action, str):
                raise ValueError(f"Invalid signal tuple action: {sig!r}")
            a = action.upper()
            if a not in ("BUY", "SELL"):
                raise ValueError(f"Invalid signal tuple action: {sig!r}")
            if tag is not None and not isinstance(tag, str):
                raise ValueError(f"Invalid signal tuple tag: {sig!r}")
            return a, float(score), (tag if tag else None)

        raise ValueError(f"Invalid signal tuple length: {len(sig)} for {sig!r}")

    if isinstance(sig, dict):
        action = sig.get("action")
        score = sig.get("score", 0.0)
        tag = sig.get("tag", None)
        if not isinstance(action, str):
            raise ValueError(f"Invalid signal dict action: {sig!r}")
        a = action.upper()
        if a not in ("BUY", "SELL"):
            raise ValueError(f"Invalid signal dict action: {sig!r}")
        if tag is not None and not isinstance(tag, str):
            raise ValueError(f"Invalid signal dict tag: {sig!r}")
        return a, float(score), (tag if tag else None)

    raise ValueError(f"Invalid signal type: {type(sig).__name__}: {sig!r}")



Signal = Optional[str]  # For legacy strategies; ranked strategies may return tuple/dict too.
StrategyFn = Callable[[Dict[str, Any]], Any]


@dataclass(frozen=True)
class BacktestConfig:
    bars_root: str
    symbols: List[str]
    start: str  # inclusive "YYYY-MM-DD"
    end: str    # exclusive "YYYY-MM-DD"
    initial_cash: float = 100_000.0
    commission_per_trade: float = 0.0
    slippage_bps: float = 0.0
    max_positions: int = 10
    target_position_pct: float = 0.10  # position value as % of equity
    allow_fractional: bool = False


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.DataFrame
    positions: pd.DataFrame
    config: BacktestConfig


class ParquetDailyDataSource:
    def __init__(self, bars_root: str):
        self.root = Path(bars_root)

    def _symbol_dir(self, symbol: str) -> Path:
        return self.root / f"symbol={symbol}"

    def list_symbol_files(self, symbol: str) -> List[Path]:
        d = self._symbol_dir(symbol)
        if not d.exists():
            return []
        return sorted(d.glob("year=*.parquet"))

    def load_symbol(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        files = self.list_symbol_files(symbol)
        if not files:
            raise FileNotFoundError(f"No parquet files found for symbol={symbol} under {self.root}")

        dfs = []
        for f in files:
            df = pd.read_parquet(f, columns=["date", "open", "high", "low", "close", "volume", "symbol"])
            dfs.append(df)

        out = pd.concat(dfs, ignore_index=True)
        out["date"] = pd.to_datetime(out["date"])
        out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

        # Clip to [start, end)
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        out = out[(out["date"] >= start_dt) & (out["date"] < end_dt)].reset_index(drop=True)

        if out.empty:
            raise ValueError(f"{symbol}: no rows in requested range [{start}, {end})")

        return out


class Portfolio:
    def __init__(self, initial_cash: float):
        self.entry_tag: Dict[str, str] = {}  # symbol -> strategy tag
        self.cash: float = float(initial_cash)
        self.positions: Dict[str, int] = {}  # symbol -> shares (long-only)
        self.avg_price: Dict[str, float] = {}  # optional

    def shares(self, symbol: str) -> int:
        return int(self.positions.get(symbol, 0))

    def has_position(self, symbol: str) -> bool:
        return self.shares(symbol) != 0

    def num_positions(self) -> int:
        return sum(1 for _, q in self.positions.items() if q != 0)
    
    def get_entry_tag(self, symbol: str) -> Optional[str]:
        t = self.entry_tag.get(symbol)
        return str(t) if t is not None else None


    def equity(self, prices: Dict[str, float]) -> float:
        eq = self.cash
        for sym, qty in self.positions.items():
            if qty == 0:
                continue
            px = prices.get(sym)
            if px is None:
                continue
            eq += qty * px
        return float(eq)
    
    def open_by_tag(self) -> Dict[str, int]:
        # counts open positions by entry_tag
        counts: Dict[str, int] = {}
        for sym, qty in self.positions.items():
            if qty == 0:
                continue
            tag = self.entry_tag.get(sym)
            if not tag:
                continue
            counts[str(tag)] = counts.get(str(tag), 0) + 1
        return counts

    def buy(self, symbol: str, qty: int, price: float, commission: float, tag: Optional[str] = None) -> None:
        cost = qty * price + commission
        if cost > self.cash + 1e-9:
            raise ValueError(f"Insufficient cash to buy {qty} {symbol} at {price} (cost={cost}, cash={self.cash})")

        self.cash -= cost
        prev_qty = self.positions.get(symbol, 0)
        prev_avg = self.avg_price.get(symbol, 0.0)
        new_qty = prev_qty + qty
        new_avg = ((prev_qty * prev_avg) + (qty * price)) / new_qty if new_qty != 0 else 0.0

        self.positions[symbol] = new_qty
        self.avg_price[symbol] = float(new_avg)
        if tag is not None:
            self.entry_tag[symbol] = str(tag)

    def sell(self, symbol: str, qty: int, price: float, commission: float) -> None:
        prev_qty = self.positions.get(symbol, 0)
        if qty > prev_qty:
            raise ValueError(f"Insufficient shares to sell {qty} {symbol} (have {prev_qty})")

        proceeds = qty * price - commission
        self.cash += proceeds
        new_qty = prev_qty - qty
        self.positions[symbol] = new_qty
        if new_qty == 0:
            self.avg_price.pop(symbol, None)
            self.entry_tag.pop(symbol, None)


class Backtester:
    def __init__(self, cfg: BacktestConfig, strategy: StrategyFn):
        self.cfg = cfg
        self.strategy = strategy
        self.data = ParquetDailyDataSource(cfg.bars_root)

    @staticmethod
    def _apply_slippage(price: float, side: str, slippage_bps: float) -> float:
        if slippage_bps <= 0:
            return float(price)
        slip = float(price) * (slippage_bps / 10_000.0)
        return float(price + slip) if side == "BUY" else float(price - slip)

    def _position_size_shares(self, equity: float, next_open: float) -> int:
        target_value = equity * self.cfg.target_position_pct
        if target_value <= 0 or next_open <= 0:
            return 0
        if self.cfg.allow_fractional:
            return int(target_value / next_open)
        return int(target_value // next_open)

    def run(self) -> BacktestResult:
        cfg = self.cfg
        portfolio = Portfolio(cfg.initial_cash)

        # Load data upfront
        symbol_bars: Dict[str, pd.DataFrame] = {}
        for sym in cfg.symbols:
            symbol_bars[sym] = self.data.load_symbol(sym, cfg.start, cfg.end)

        # Master calendar (union)
        all_dates = sorted(set(pd.concat([df["date"] for df in symbol_bars.values()]).dt.normalize().tolist()))
        if len(all_dates) < 2:
            raise ValueError("Not enough dates across symbols to run (need at least 2).")

        # date->row index maps
        idx_map: Dict[str, Dict[pd.Timestamp, int]] = {}
        for sym, df in symbol_bars.items():
            idx_map[sym] = {d.normalize(): i for i, d in enumerate(df["date"])}

        trades: List[Dict[str, Any]] = []
        equity_rows: List[Dict[str, Any]] = []
        pos_rows: List[Dict[str, Any]] = []
        
        for di in range(len(all_dates) - 1):
            dt = all_dates[di]
            next_dt = all_dates[di + 1]

            # Mark-to-market using today's closes
            close_prices: Dict[str, float] = {}
            for sym, df in symbol_bars.items():
                i = idx_map[sym].get(dt)
                if i is not None:
                    close_prices[sym] = float(df.loc[i, "close"])

            equity = portfolio.equity(close_prices)
            equity_rows.append({"date": dt.date().isoformat(), "equity": equity, "cash": portfolio.cash})

            # Position snapshot (debugging)
            for sym in cfg.symbols:
                qty = portfolio.shares(sym)
                if qty != 0:
                    pos_rows.append({"date": dt.date().isoformat(), "symbol": sym, "shares": qty})

            # ----------------------------
            # Ranked signal collection
            # ----------------------------
            sell_list: List[Tuple[str, float]] = []                  # (sym, next_open)
            buy_list: List[Tuple[float, str, float]] = []            # (score, sym, next_open)

            for sym, df in symbol_bars.items():
                i = idx_map[sym].get(dt)
                ni = idx_map[sym].get(next_dt)
                if i is None or ni is None:
                    continue

                next_open = float(df.loc[ni, "open"])
                if next_open <= 0:
                    continue

                ctx = {
                    "date": dt,
                    "symbol": sym,
                    "bar": df.loc[i].to_dict(),
                    "history": df.iloc[: i + 1],
                    "position": portfolio.shares(sym),
                    "entry_tag": portfolio.get_entry_tag(sym),
                    "open_by_tag": portfolio.open_by_tag(),  # 👈 NEW
                    "cash": portfolio.cash,
                    "equity": equity,
                }

                action, score, tag = _parse_signal(self.strategy(ctx))

                if action == "SELL":
                    if portfolio.shares(sym) > 0:
                        sell_list.append((sym, next_open))
                elif action == "BUY":
                    buy_list.append((float(score), sym, next_open, tag))

            # ----------------------------
            # Execute SELLS first
            # ----------------------------
            for sym, next_open in sell_list:
                qty = portfolio.shares(sym)
                if qty <= 0:
                    continue

                fill_px = self._apply_slippage(next_open, "SELL", cfg.slippage_bps)
                portfolio.sell(sym, qty, fill_px, cfg.commission_per_trade)

                trades.append({
                    "date": next_dt.date().isoformat(),
                    "symbol": sym,
                    "side": "SELL",
                    "qty": qty,
                    "price": fill_px,
                    "commission": cfg.commission_per_trade,
                    "slippage_bps": cfg.slippage_bps,
                    "fill_model": "next_open",
                })

            # ----------------------------
            # Execute BUYS ranked by score desc
            # ----------------------------
            buy_list.sort(key=lambda x: x[0], reverse=True)

            for score, sym, next_open, tag in buy_list:
                if portfolio.has_position(sym):
                    continue
                if portfolio.num_positions() >= cfg.max_positions:
                    break

                qty = self._position_size_shares(equity, next_open)
                if qty <= 0:
                    continue

                fill_px = self._apply_slippage(next_open, "BUY", cfg.slippage_bps)
                try:
                    portfolio.buy(sym, qty, fill_px, cfg.commission_per_trade, tag=tag)
                except ValueError:
                    continue

                trades.append({
                    "date": next_dt.date().isoformat(),
                    "symbol": sym,
                    "side": "BUY",
                    "qty": qty,
                    "price": fill_px,
                    "commission": cfg.commission_per_trade,
                    "slippage_bps": cfg.slippage_bps,
                    "fill_model": "next_open",
                    "entry_score": float(score),
                    "entry_tag": tag
                })

        # Final mark-to-market
        last_dt = all_dates[-1]
        close_prices = {}
        for sym, df in symbol_bars.items():
            i = idx_map[sym].get(last_dt)
            if i is not None:
                close_prices[sym] = float(df.loc[i, "close"])
        equity = portfolio.equity(close_prices)
        equity_rows.append({"date": last_dt.date().isoformat(), "equity": equity, "cash": portfolio.cash})
        
        # ----------------------------
        # Force liquidation at final close (so metrics include unrealized PnL)
        # ----------------------------
        for sym, qty in list(portfolio.positions.items()):
            if qty == 0:
                continue
            px = close_prices.get(sym)
            if px is None:
                continue

            fill_px = self._apply_slippage(float(px), "SELL", cfg.slippage_bps)
            portfolio.sell(sym, int(qty), fill_px, cfg.commission_per_trade)

            trades.append({
                "date": last_dt.date().isoformat(),
                "symbol": sym,
                "side": "SELL",
                "qty": int(qty),
                "price": float(fill_px),
                "commission": cfg.commission_per_trade,
                "slippage_bps": cfg.slippage_bps,
                "fill_model": "final_close",
            })
            
        # Final equity AFTER liquidation (should be ~all cash)
        equity = portfolio.equity({})
        equity_rows.append({"date": last_dt.date().isoformat(), "equity": equity, "cash": portfolio.cash})

        return BacktestResult(
            trades=pd.DataFrame(trades),
            equity=pd.DataFrame(equity_rows),
            positions=pd.DataFrame(pos_rows),
            config=cfg,
        )


# -----------------------------
# Optional: simple CLI example
# -----------------------------
def _example_strategy(ctx: Dict[str, Any]) -> Signal:
    """
    Tiny deterministic strategy:
    - BUY if today's close > yesterday's close and you don't already hold
    - SELL if today's close < yesterday's close and you do hold
    """
    hist: pd.DataFrame = ctx["history"]
    if len(hist) < 2:
        return None
    c0 = float(hist.iloc[-2]["close"])
    c1 = float(hist.iloc[-1]["close"])
    pos = int(ctx["position"])
    if pos == 0 and c1 > c0:
        return "BUY"
    if pos != 0 and c1 < c0:
        return "SELL"
    return None


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--bars-root", default="data/bars_1d")
    ap.add_argument("--symbols", default="AAPL,MSFT")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2020-01-01")
    ap.add_argument("--cash", type=float, default=100000.0)
    ap.add_argument("--commission", type=float, default=1.0)
    ap.add_argument("--slippage-bps", type=float, default=1.0)
    ap.add_argument("--max-positions", type=int, default=10)
    ap.add_argument("--target-pct", type=float, default=0.10)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    cfg = BacktestConfig(
        bars_root=args.bars_root,
        symbols=[s.strip().upper() for s in args.symbols.split(",") if s.strip()],
        start=args.start,
        end=args.end,
        initial_cash=args.cash,
        commission_per_trade=args.commission,
        slippage_bps=args.slippage_bps,
        max_positions=args.max_positions,
        target_position_pct=args.target_pct,
    )

    bt = Backtester(cfg, _example_strategy)
    res = bt.run()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    res.trades.to_csv(outdir / "trades.csv", index=False)
    res.equity.to_csv(outdir / "equity.csv", index=False)
    res.positions.to_csv(outdir / "positions.csv", index=False)

    print(f"Done. Trades: {len(res.trades)} | Equity rows: {len(res.equity)}")
    print(f"Wrote: {outdir/'trades.csv'}")
    print(f"Wrote: {outdir/'equity.csv'}")
    print(f"Wrote: {outdir/'positions.csv'}")
