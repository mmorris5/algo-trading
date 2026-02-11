# strategies/pullback_sma.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import pandas as pd


def sma_pullback_continuation(
    trend_len: int = 50,
    pullback_len: int = 20,
    slope_lookback: int = 10,
    min_slope: float = 0.0,
    exit_on_close_below_trend: bool = True,
):
    """
    SMA Pullback Continuation (long-only):

    Trend filter:
      - Close > SMA(trend_len)
      - SMA(trend_len) is rising over slope_lookback (optional)

    Pullback setup:
      - Yesterday close <= SMA(pullback_len) (i.e., a pullback to the fast MA)
      - Today close > SMA(pullback_len) (reclaim / bounce confirmation)

    Exit:
      - If exit_on_close_below_trend: SELL when close < SMA(trend_len)
      - Otherwise: strategy is entry-only and relies on another sleeve/exit logic (not recommended)

    Signal timing:
      - Signals computed on today's close; engine fills at next day open.

    Returns:
      - ("BUY", score, "pullback") or "SELL" or None
      - score ranks opportunities cross-sectionally.

    Notes:
      - This is NOT mean reversion; it aligns with trend and buys pullbacks.
      - Default params are conservative and robust.
    """

    if trend_len < 20:
        raise ValueError("trend_len should be >= 20")
    if pullback_len < 5:
        raise ValueError("pullback_len should be >= 5")
    if slope_lookback < 2:
        raise ValueError("slope_lookback should be >= 2")

    warmup = max(trend_len, pullback_len) + slope_lookback + 5

    def _strategy(ctx: Dict[str, Any]):
        hist: pd.DataFrame = ctx["history"]
        pos = int(ctx["position"])

        if len(hist) < warmup:
            return None

        close = hist["close"].astype(float)

        sma_trend = close.rolling(trend_len).mean()
        sma_fast = close.rolling(pullback_len).mean()

        c0 = float(close.iloc[-2])   # yesterday close
        c1 = float(close.iloc[-1])   # today close

        t0 = float(sma_trend.iloc[-2])
        t1 = float(sma_trend.iloc[-1])

        f0 = float(sma_fast.iloc[-2])
        f1 = float(sma_fast.iloc[-1])

        # Optional trend slope filter
        slope_ok = True
        if slope_lookback > 0:
            t_prev = float(sma_trend.iloc[-1 - slope_lookback])
            slope = (t1 / t_prev - 1.0) if t_prev != 0 else 0.0
            slope_ok = slope >= float(min_slope)

        in_trend = (c1 > t1) and slope_ok

        # Exit logic
        if pos != 0:
            if exit_on_close_below_trend and c1 < t1:
                return "SELL"
            return None

        # Entry logic (flat)
        if not in_trend:
            return None

        # Pullback + reclaim:
        # - yesterday was at/below fast SMA
        # - today reclaimed above fast SMA
        pulled_back = (c0 <= f0)
        reclaimed = (c1 > f1)

        if not (pulled_back and reclaimed):
            return None

        # Ranking score:
        # Prefer stronger trend + deeper pullback
        # 1) trend strength: distance above trend SMA
        trend_strength = (c1 / t1 - 1.0) if t1 != 0 else 0.0

        # 2) pullback depth: how far below fast SMA yesterday was (more negative => deeper)
        pullback_depth = (c0 / f0 - 1.0) if f0 != 0 else 0.0

        # Convert to a single score: more is better
        # - trend_strength positive helps
        # - deeper pullback means pullback_depth more negative, so subtract it
        score = float(trend_strength - pullback_depth)

        return ("BUY", score, "pullback")

    return _strategy
