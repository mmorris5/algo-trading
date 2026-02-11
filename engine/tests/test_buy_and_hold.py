from backtest import BacktestConfig, Backtester

def buy_and_hold(ctx):
    # Buy on the very first bar, then do nothing
    hist = ctx["history"]
    if len(hist) == 1 and ctx["position"] == 0:
        return "BUY"
    return None


cfg = BacktestConfig(
    bars_root="../historical_data/bars_ohlcv_1d",
    symbols=["AAPL"],          # pick ONE liquid stock
    start="2024-07-01",
    end="2026-02-01",
    initial_cash=100_000,
    commission_per_trade=0.0,  # turn OFF frictions for this test
    slippage_bps=0.0,
    max_positions=1,
    target_position_pct=1.0,   # all-in
)

bt = Backtester(cfg, buy_and_hold)
res = bt.run()

res.trades.to_csv("test_trades.csv", index=False)
res.equity.to_csv("test_equity.csv", index=False)

print(res.trades)
print(res.equity.tail())
