import pandas as pd
from pathlib import Path

# Adjust these if needed
TRADES = Path("test_trades.csv")
EQUITY = Path("test_equity.csv")

trades = pd.read_csv(TRADES)
eq = pd.read_csv(EQUITY)

assert len(trades) == 1, f"Expected exactly 1 trade (BUY), got {len(trades)}"
t = trades.iloc[0]
assert t["side"] == "BUY", f"Expected BUY, got {t['side']}"

qty = int(t["qty"])
buy_px = float(t["price"])
commission = float(t.get("commission", 0.0))

final = eq.iloc[-1]
final_date = final["date"]
final_equity = float(final["equity"])
final_cash = float(final["cash"])

# This is what the last close *must* have been if the accounting is correct
implied_last_close = (final_equity - final_cash) / qty

print("Final date:", final_date)
print("Final equity (reported):", final_equity)
print("Final cash (reported):", final_cash)
print("Buy fill:", t["date"], "qty:", qty, "buy_px:", buy_px, "commission:", commission)
print("Implied last close:", implied_last_close)
