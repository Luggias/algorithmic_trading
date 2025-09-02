# Shared dataclasses (Bar, Order, Position)

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Literal

Price = Decimal  # more precise than float for money
Qty = Decimal

@dataclass(frozen=True) # immutabel (cannot be changed at execution time of the strategy)
class Bar:
    symbol: str
    timestamp: datetime       # tz-aware, UTC
    open: Price
    high: Price
    low: Price
    close: Price
    volume: Qty               # quantity – not an amount of money
    interval: Literal["1m","5m","1h","1d"] | str = "1d"

    def validate(self) -> None:
        assert self.low <= self.open <= self.high
        assert self.low <= self.close <= self.high
        assert self.volume >= 0
        assert self.timestamp.tzinfo is not None

from enum import Enum, auto
from dataclasses import field

class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class TIF(str, Enum):
    GTC = "GTC"   # Good-Till-Cancelled
    IOC = "IOC"   # Immediate-Or-Cancel
    FOK = "FOK"   # Fill-Or-Kill

@dataclass(frozen=True) # immutabel
class Order:
    symbol: str
    side: Side
    qty: Qty
    order_type: OrderType
    limit_price: Price | None = None
    stop_price: Price | None = None
    time_in_force: TIF = TIF.GTC
    client_order_id: str | None = None
    created_at: datetime | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        assert self.qty > 0
        if self.order_type == OrderType.MARKET:
            assert self.limit_price is None and self.stop_price is None
        if self.order_type == OrderType.LIMIT:
            assert self.limit_price is not None
        if self.order_type == OrderType.STOP:
            assert self.stop_price is not None
        if self.order_type == OrderType.STOP_LIMIT:
            assert self.limit_price is not None and self.stop_price is not None