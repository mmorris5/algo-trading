# force_roundtrip_test.py
"""
Forced round-trip sanity test:
- BUY once (signal on first available day)
- SELL once (signal N trading days later)
- Fills happen on NEXT trading day OPEN (per your backtest.py model)
- Verifies:
    final_equity == initial_cash + qty*(sell_px - buy_px) - commissions
and prints all the details.

Run:
  python force_roundtrip_test.py
"""

from pathlib import Path
import pandas as pd

from backtest import BacktestConfig, Backtester


HOLD_DAYS = 10  # number of TRADING DAYS to hold after entry signal day


def force_roundtrip(ctx):
    """
    Signals:
    - BUY on first bar (history length == 1)
    - SELL when history length == 1 + HOLD_DAYS
      (i.e., after HOLD_DAYS bars have occurred since start)
    """
    hist = ctx["history"]
    pos = int(ctx["position"])
    if len(hist) == 1 and pos == 0:
        return "BUY"
    if len(hist) == 1 + HOLD_DAYS and pos != 0:
        return "SELL"
    return None


def main():
    # ---- Configure this to your environment ----
    # If you're running from a folder where ../historical_data/... is correct, keep it.
    BARS_ROOT = "../historical_data/bars_ohlcv_1d"
    SYMBOL = "AAPL"
    START = "2024-07-01"
    END = "2026-01-30"  # exclusive; make sure this window has > HOLD_DAYS+2 trading days
    INITIAL_CASH = 100_000.0

    # Turn frictions ON or OFF to test both
    COMMISSION = 1.00
    SLIPPAGE_BPS = 1.0

    cfg = BacktestConfig(
        bars_root=BARS_ROOT,
        symbols=[SYMBOL],
        start=START,
        end=END,
        initial_cash=INITIAL_CASH,
        commission_per_trade=COMMISSION,
        slippage_bps=SLIPPAGE_BPS,
        max_positions=1,
        target_position_pct=1.0,  # all-in to make math obvious
        allow_fractional=False,
    )

    outdir = Path("results")
    outdir.mkdir(parents=True, exist_ok=True)

    bt = Backtester(cfg, force_roundtrip)
    res = bt.run()

    trades_path = outdir / "roundtrip_trades.csv"
    equity_path = outdir / "roundtrip_equity.csv"
    res.trades.to_csv(trades_path, index=False)
    res.equity.to_csv(equity_path, index=False)

    trades = res.trades
    equity = res.equity

    print(f"Wrote {trades_path}")
    print(f"Wrote {equity_path}\n")

    if len(trades) != 2:
        print(trades)
        raise SystemExit(f"Expected exactly 2 trades (BUY, SELL). Got {len(trades)}.")

    buy = trades.iloc[0]
    sell = trades.iloc[1]
    if buy["side"] != "BUY" or sell["side"] != "SELL":
        raise SystemExit(f"Expected BUY then SELL. Got: {buy['side']} then {sell['side']}")

    qty = int(buy["qty"])
    buy_px = float(buy["price"])
    sell_px = float(sell["price"])

    # Total commissions paid (one per trade in this engine)
    comm_total = float(buy.get("commission", 0.0)) + float(sell.get("commission", 0.0))

    # Expected final equity:
    # initial_cash -> after buy: cash = initial_cash - qty*buy_px - buy_comm
    # after sell: cash = (initial_cash - qty*buy_px - buy_comm) + qty*sell_px - sell_comm
    expected_final = INITIAL_CASH + qty * (sell_px - buy_px) - comm_total

    reported_final = float(equity.iloc[-1]["equity"])
    reported_cash = float(equity.iloc[-1]["cash"])
    final_date = equity.iloc[-1]["date"]

    print("=== Round-trip details ===")
    print(f"Symbol: {SYMBOL}")
    print(f"Buy fill date:  {buy['date']} @ {buy_px:.4f}  qty={qty}")
    print(f"Sell fill date: {sell['date']} @ {sell_px:.4f}  qty={int(sell['qty'])}")
    print(f"Commission total: {comm_total:.2f}")
    print(f"Slippage bps: {SLIPPAGE_BPS}")
    print()
    print("=== Equity check ===")
    print(f"Expected final equity: {expected_final:.2f}")
    print(f"Reported final equity: {reported_final:.2f}")
    print(f"Reported final cash:   {reported_cash:.2f}")
    print(f"Final equity date:     {final_date}")

    diff = abs(expected_final - reported_final)
    if diff > 0.01:
        raise SystemExit(f"FAIL: expected vs reported differ by {diff:.4f}")
    print("PASS ✅ (expected final equity matches engine output)")


if __name__ == "__main__":
    main()
