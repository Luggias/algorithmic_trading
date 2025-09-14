from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from common.types import Order, OrderType, Side, Bar

@dataclass(frozen=True)
class NextBarOpenExecution:
    """Deterministic: executes on the open of the next bar."""
    def simulate(self, pending: Order, next_bar: Bar) -> Optional[float]:
        o = pending
        px = float(next_bar.open)
        if o.order_type is OrderType.MARKET:
            return px
        if o.order_type is OrderType.LIMIT:
            if o.side is Side.BUY and px <= (o.limit_price or float("inf")):
                return px
            if o.side is Side.SELL and px >= (o.limit_price or float("-inf")):
                return px
            return None
        # TODO: STOP/STOP_LIMIT optional (implement later)
        return None