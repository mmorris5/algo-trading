# engine/live/execution/ibkr.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import time

from ib_insync import IB, Stock, Order, Trade


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 4002     # Gateway paper by default
DEFAULT_CLIENT_ID = 7


@dataclass
class Fill:
    symbol: str
    side: str
    qty: int
    avg_fill_price: float


class IBKRExecutor:
    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, client_id: int = DEFAULT_CLIENT_ID):
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id
        self._conid_cache: Dict[str, int] = {}

    def connect(self) -> None:
        if self.ib.isConnected():
            return
        self.ib.connect(self.host, self.port, clientId=self.client_id)

    def __enter__(self) -> "IBKRExecutor":
        self.connect()
        return self

    def disconnect(self) -> None:
        if self.ib.isConnected():
            self.ib.disconnect()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disconnect()

    def get_cash(self, currency: str = "USD") -> float:
        """Return TotalCashValue for the account in the given currency (via accountSummary)."""
        total = 0.0
        for v in self.ib.accountSummary():
            # v has fields: account, tag, value, currency
            if v.tag == "TotalCashValue" and (currency is None or v.currency == currency):
                try:
                    total = float(v.value)
                except (TypeError, ValueError):
                    continue
        return total

    def qualify_stock(self, symbol: str) -> Stock:
        symbol = symbol.strip().upper()
        if symbol in self._conid_cache:
            c = Stock(symbol, "SMART", "USD")
            c.conId = self._conid_cache[symbol]
            return c

        c = Stock(symbol, "SMART", "USD")
        qs = self.ib.qualifyContracts(c)
        if not qs:
            raise RuntimeError(f"Unable to qualify contract for {symbol}")
        qc = qs[0]
        self._conid_cache[symbol] = int(qc.conId)
        return qc

    def get_positions(self) -> Dict[str, int]:
        # ib.positions() is sync; use it directly
        pos = {}
        for p in self.ib.positions():
            if p.position:
                pos[p.contract.symbol.upper()] = int(p.position)
        return pos

    def place_moo(self, symbol: str, side: str, qty: int) -> Trade:
        """
        Market-on-open order. In paper this is the closest analog to your backtest fill model.
        """
        if qty <= 0:
            raise ValueError("qty must be > 0")
        side = side.upper()
        if side not in ("BUY", "SELL"):
            raise ValueError("side must be BUY or SELL")

        contract = self.qualify_stock(symbol)
        order = Order(
            action=side,
            orderType="MKT",
            tif="DAY",              
            totalQuantity=qty,
        )
        trade = self.ib.placeOrder(contract, order)
        return trade

    def wait_for_fills(self, trades: List[Trade], timeout_sec: int = 120) -> List[Fill]:
        """
        Wait for fills (or partials) and return average fill prices.
        """
        fills: List[Fill] = []
        deadline = time.monotonic() + timeout_sec

        # poll until all are done or timeout
        pending: List[Trade] = list(trades)
        while pending and time.monotonic() < deadline:
            # Pump events and wait briefly for updates
            self.ib.waitOnUpdate(timeout=1.0)
            remaining: List[Trade] = []
            for tr in pending:
                status = tr.orderStatus.status
                if status in ("Filled", "Cancelled", "Inactive"):
                    continue
                remaining.append(tr)
            pending = remaining

        # collect fills
        for tr in trades:
            sym = tr.contract.symbol.upper()
            side = tr.order.action
            qty = int(tr.order.totalQuantity)
            avg = float(tr.orderStatus.avgFillPrice or 0.0)
            fills.append(Fill(symbol=sym, side=side, qty=qty, avg_fill_price=avg))
        return fills
def _cli_main() -> None:
    """Minimal CLI to place a single MOO order via IB paper.

    Usage example (from engine/ directory):
      python -m live.execution.ikbr --symbol AAPL --side BUY --qty 10
    """
    import argparse

    parser = argparse.ArgumentParser(description="Submit a single MOO order via IBKR paper account.")
    parser.add_argument("--symbol", required=True, help="Ticker symbol, e.g. AAPL")
    parser.add_argument("--side", required=True, choices=["BUY", "SELL", "buy", "sell"], help="BUY or SELL")
    parser.add_argument("--qty", required=True, type=int, help="Share quantity (positive integer)")
    parser.add_argument("--host", default=DEFAULT_HOST, help="IB host (default 127.0.0.1)")
    parser.add_argument("--port", default=DEFAULT_PORT, type=int, help="IB port (default 7497)")
    parser.add_argument("--client-id", default=DEFAULT_CLIENT_ID, type=int, help="IB client id")
    args = parser.parse_args()

    side = args.side.upper()

    with IBKRExecutor(host=args.host, port=args.port, client_id=args.client_id) as ex:
        trade = ex.place_moo(args.symbol, side, args.qty)

        # Optionally wait for fills but don't block forever; this is paper-only.
        fills = ex.wait_for_fills([trade], timeout_sec=120)

    for f in fills:
        print(f"Filled {f.side} {f.qty} {f.symbol} @ {f.avg_fill_price:.4f}")


if __name__ == "__main__":
    _cli_main()
