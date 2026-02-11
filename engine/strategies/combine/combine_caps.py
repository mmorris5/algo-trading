# strategies/combine_caps.py
from __future__ import annotations
from typing import Any, Callable, Dict, Optional, Tuple

StrategyFn = Callable[[Dict[str, Any]], Any]
ParsedSignal = Tuple[Optional[str], float, Optional[str]]


def _parse_signal(sig) -> ParsedSignal:
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
            return str(action).upper(), float(score), None
        if len(sig) == 3:
            action, score, tag = sig
            return str(action).upper(), float(score), (str(tag) if tag else None)
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


def combine_with_caps(
    s1: StrategyFn,
    s2: StrategyFn,
    *,
    tag1: str = "donchian",
    tag2: str = "pullback",
    max_s1_positions: int = 4,
    max_s2_positions: int = 6,
):
    """
    Combine 2 strategies into one:
      - Entries compete by score (ranked entries)
      - BUT each sleeve has a max concurrent open-position cap
      - Exits are attribution-based (who entered controls exit)

    Requires ctx:
      - entry_tag
      - open_by_tag (dict: tag -> count)
    """

    def _strategy(ctx: Dict[str, Any]):
        pos = int(ctx.get("position", 0))
        entry_tag = ctx.get("entry_tag")
        open_by_tag = ctx.get("open_by_tag") or {}

        # Evaluate both sleeves
        a1, sc1, _ = _parse_signal(s1(ctx))
        a2, sc2, _ = _parse_signal(s2(ctx))

        # -----------------
        # HOLDING → EXIT (attribution)
        # -----------------
        if pos != 0:
            if entry_tag == tag1:
                return "SELL" if a1 == "SELL" else None
            if entry_tag == tag2:
                return "SELL" if a2 == "SELL" else None
            # Safety fallback
            return "SELL" if (a1 == "SELL" or a2 == "SELL") else None

        # -----------------
        # FLAT → ENTRY (ranked + caps)
        # -----------------
        c1 = int(open_by_tag.get(tag1, 0))
        c2 = int(open_by_tag.get(tag2, 0))

        candidates = []

        if a1 == "BUY" and c1 < max_s1_positions:
            candidates.append((float(sc1), tag1))

        if a2 == "BUY" and c2 < max_s2_positions:
            candidates.append((float(sc2), tag2))

        if not candidates:
            return None

        score, chosen_tag = max(candidates, key=lambda x: x[0])
        return ("BUY", float(score), chosen_tag)

    return _strategy
