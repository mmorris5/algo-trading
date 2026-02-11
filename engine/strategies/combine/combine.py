# strategies/combine.py
from __future__ import annotations
from typing import Any, Callable, Dict, Optional, Tuple, Union

StrategyFn = Callable[[Dict[str, Any]], Any]

# Returned by parser: (action, score, tag)
ParsedSignal = Tuple[Optional[str], float, Optional[str]]


def _parse_signal(sig) -> ParsedSignal:
    """
    Local parser (so we don't import backtest.py).

    Accepts:
      - None
      - "BUY" / "SELL"
      - ("BUY", score) or ("SELL", score)
      - ("BUY", score, tag)  <-- NEW
      - {"action": "BUY", "score": 1.23, "tag": "mr"}  <-- NEW (tag optional)

    Returns: (action: "BUY"/"SELL"/None, score: float, tag: Optional[str])
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

def combine_strategies(
    s1: StrategyFn,
    s2: StrategyFn,
    *,
    tag1: str = "donchian",
    tag2: str = "mean_reversion",
):
    """
    Combine two strategies with shared slots and DAY-LEVEL Donchian priority.

    Rules:
      - If ANY Donchian BUY occurs on a given day (across any symbol), then
        MR BUYs are ignored for the rest of that day.
      - If NO Donchian BUY occurs that day, MR is allowed to BUY.
      - Exits are controlled by entry_tag attribution (who entered controls exit).
    """

    state = {"dt": None, "seen_donchian_buy": False}

    def _strategy(ctx: Dict[str, Any]):
        # If engine asks for precheck, return Donchian (s1) action only
        if ctx.get("_precheck_donchian_only"):
            a1, sc1, _ = _parse_signal(s1(ctx))
            return ("BUY", float(sc1), tag1) if a1 == "BUY" else None
        
        dt = ctx.get("date")
        pos = int(ctx.get("position", 0))
        entry_tag = ctx.get("entry_tag")

        # reset each day
        if state["dt"] is None or dt != state["dt"]:
            state["dt"] = dt
            state["seen_donchian_buy"] = False

        a1, sc1, _ = _parse_signal(s1(ctx))
        a2, sc2, _ = _parse_signal(s2(ctx))

        if pos == 0:
            if a1 == "BUY":
                return ("BUY", float(sc1), tag1)
            if ctx.get("allow_mr_today", True) and a2 == "BUY":
                return ("BUY", float(sc2), tag2)
            return None

        # exits by attribution
        if entry_tag == tag1:
            return "SELL" if a1 == "SELL" else None
        if entry_tag == tag2:
            return "SELL" if a2 == "SELL" else None

        return "SELL" if (a1 == "SELL" or a2 == "SELL") else None

    return _strategy
