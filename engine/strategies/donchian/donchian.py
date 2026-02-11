# strategies/donchian.py
from __future__ import annotations
import pandas as pd


def donchian_breakout(
    entry_lookback: int = 20,
    exit_lookback: int = 10,
    sma_period: int = 200,
):
    """
    Donchian breakout (long-only), ranked entries via hybrid breakout*trend strength:

      Entry:
        - BUY when today's close > prior N-day high (excluding today)
        - Score = ((close - prior_high) / prior_high) * ((close - SMA(sma_period)) / SMA(sma_period))

      Exit:
        - SELL when today's close < prior M-day low (excluding today)

    Signals are generated using today's close; fills happen next day open in the engine.
    """

    if entry_lookback < 2 or exit_lookback < 2 or sma_period < 20:
        raise ValueError("Lookbacks should be reasonable (entry/exit >=2, sma_period >=20)")

    warmup = max(entry_lookback, exit_lookback, sma_period) + 2

    def _strategy(ctx):
        hist: pd.DataFrame = ctx["history"]
        pos = int(ctx["position"])
        if len(hist) < warmup:
            return None

        close = hist["close"].astype(float)
        sma = close.rolling(sma_period).mean()

        # Prior-period channels (exclude today to avoid lookahead)
        prior_high = hist["high"].astype(float).shift(1).rolling(entry_lookback).max()
        prior_low  = hist["low"].astype(float).shift(1).rolling(exit_lookback).min()

        c = float(close.iloc[-1])
        hh = float(prior_high.iloc[-1])
        ll = float(prior_low.iloc[-1])
        s = float(sma.iloc[-1])

        if pos == 0:
            if c > hh and hh > 0 and s > 0:
                breakout_strength = (c - hh) / hh
                trend_strength = (c - s) / s
                score = breakout_strength * trend_strength
                return ("BUY", float(score))
        else:
            if c < ll:
                return "SELL"

        return None

    return _strategy
