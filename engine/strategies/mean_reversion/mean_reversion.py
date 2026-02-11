# strategies/mean_reversion_v1.py
from __future__ import annotations
import pandas as pd

import numpy as np
import pandas as pd

def _rsi_wilder(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()

    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    # IMPORTANT: use np.nan, not pd.NA
    rs = avg_gain / avg_loss.replace(0.0, np.nan)

    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Keep numeric dtype, fill initial NaNs to neutral value
    return rsi.fillna(50.0)


def mean_reversion(
    sma_trend: int = 200,
    sma_pullback: int = 20,
    rsi_period: int = 2,
    entry_rsi: float = 5.0,
    exit_rsi: float = 50.0,
):
    """
    Mean reversion (long-only) with trend filter:
      Entry:
        - close > SMA(sma_trend)
        - RSI(rsi_period) <= entry_rsi
        - close < SMA(sma_pullback)   (pullback)
      Exit:
        - RSI(rsi_period) >= exit_rsi OR close >= SMA(sma_pullback)

    Note: No time-stop here because your engine doesn't track entry dates.
    (If you want time-based exits, we can add position state next.)
    """

    warmup = max(sma_trend, sma_pullback, rsi_period) + 5

    def _strategy(ctx):
        hist: pd.DataFrame = ctx["history"]
        pos = int(ctx["position"])
        if len(hist) < warmup:
            return None

        close = hist["close"].astype(float)

        smaT = close.rolling(sma_trend).mean()
        smaP = close.rolling(sma_pullback).mean()
        rsi = _rsi_wilder(close, rsi_period)

        c = float(close.iloc[-1])
        t = float(smaT.iloc[-1])
        p = float(smaP.iloc[-1])
        r = float(rsi.iloc[-1])

        in_trend = (t > 0) and (c > t)
        pulled_back = (p > 0) and (c < p)

        if pos == 0:
            if in_trend and pulled_back and r <= entry_rsi:
                # optional score: deeper oversold gets higher priority
                score = (entry_rsi - r)  # bigger when r is smaller
                return ("BUY", float(score))
        else:
            if (r >= exit_rsi) or (p > 0 and c >= p):
                return "SELL"

        return None

    return _strategy
