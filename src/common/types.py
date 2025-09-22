# Shared dataclasses (Bar, Order, Position)

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Generic, Literal, Mapping, TypeVar
import math

# Type aliases for semantic clarity inside research/backtests
Money = float   # use Decimal only at the broker boundary
Price = float
Qty = float

Interval = Literal["1m", "5m", "1h", "1d"]

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ApiResponse(Generic[T]):
    """Canonical HTTP/IPC response envelope used across services."""

    success: bool
    data: T | None = None
    error: str | None = None
    metadata: Mapping[str, Any] | None = None

    def require_data(self) -> T:
        """Return payload when successful, raise otherwise."""
        if not self.success or self.data is None:
            raise ValueError("ApiResponse missing data; inspect error/metadata for details")
        return self.data

def _ensure_utc(ts: datetime) -> datetime:
    """Return ts as UTC-aware datetime."""
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)

class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"

    def sign(self) -> int:
        return 1 if self is Side.BUY else -1
    
class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class TIF(str, Enum):
    GTC = "GTC"   # Good-Till-Cancelled
    IOC = "IOC"   # Immediate-Or-Cancel
    FOK = "FOK"   # Fill-Or-Kill

@dataclass(frozen=True, slots=True) # immutable, memory-efficient
class Bar:
    symbol: str
    timestamp: datetime       # tz-aware, UTC
    open: Price
    high: Price
    low: Price
    close: Price
    volume: Qty               # quantity, not money
    interval: Interval = "1d"

    def __post_init__(self) -> None:
        ts = _ensure_utc(self.timestamp)
        object.__setattr__(self, "timestamp", ts)
        if self.timestamp.utcoffset() != timedelta(0):
            raise ValueError("timestamp must be UTC")
        if not (self.low <= self.open <= self.high):
            raise ValueError("open not within high/low")
        if not (self.low <= self.close <= self.high):
            raise ValueError("close not within high/low")
        if self.volume < 0:
            raise ValueError("volume must be non-negative")

@dataclass(frozen=True, slots=True) # immutabel
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

    def __post_init__(self) -> None:
        """Run invariant checks and normalize created_at to UTC."""
        # 1) Basic field sanity
        if not self.symbol or not self.symbol.strip():
            raise ValueError("symbol must be non-empty")

        if not (isinstance(self.qty, (int, float)) and math.isfinite(self.qty) and self.qty > 0):
            raise ValueError("qty must be a finite positive number")

        # 2) Price sanity (if provided)
        for name, px in (("limit_price", self.limit_price), ("stop_price", self.stop_price)):
            if px is not None:
                if not (isinstance(px, (int, float)) and math.isfinite(px) and px > 0):
                    raise ValueError(f"{name} must be a finite positive number")

        # 3) Type-specific constraints
        if self.order_type is OrderType.MARKET:
            if self.limit_price is not None or self.stop_price is not None:
                raise ValueError("market orders must not have prices")
        elif self.order_type is OrderType.LIMIT:
            if self.limit_price is None:
                raise ValueError("limit orders require limit_price")
        elif self.order_type is OrderType.STOP:
            if self.stop_price is None:
                raise ValueError("stop orders require stop_price")
        elif self.order_type is OrderType.STOP_LIMIT:
            if self.limit_price is None or self.stop_price is None:
                raise ValueError("stop_limit orders require both limit_price and stop_price")

        # 4) Normalize created_at to UTC (if provided)
        if self.created_at is not None:
            ct = _ensure_utc(self.created_at)
            object.__setattr__(self, "created_at", ct)
            if self.created_at.utcoffset() != timedelta(0):
                raise ValueError("created_at must be UTC")

        # 5) Normalize tags (already a tuple by type; ensure no None)
        if any(t is None for t in self.tags):
            raise ValueError("tags must not contain None")


@dataclass(frozen=True, slots=True)
class Fill:
    """Execution record for a single order fill."""

    order: Order
    price: Price
    qty: Qty
    timestamp: datetime

    def notional(self) -> Money:
        return self.price * self.qty


@dataclass(frozen=True, slots=True)
class Position:
    """Net position for a symbol with FIFO average price tracking."""

    symbol: str
    qty: Qty = 0.0
    avg_price: Price = 0.0

    def market_value(self, mark: Price) -> Money:
        return self.qty * mark

    def is_flat(self) -> bool:
        return math.isclose(self.qty, 0.0, abs_tol=1e-12)

    def apply_fill(self, fill: Fill) -> "Position":
        if fill.order.symbol != self.symbol:
            raise ValueError("Fill symbol mismatch")

        signed_qty = fill.qty * fill.order.side.sign()
        new_qty = self.qty + signed_qty

        if self.is_flat():
            new_avg = fill.price
        elif math.isclose(new_qty, 0.0, abs_tol=1e-12):
            new_avg = 0.0
        elif self.qty > 0 and new_qty > 0:
            total_cost = self.avg_price * self.qty + fill.price * signed_qty
            new_avg = total_cost / new_qty
        elif self.qty < 0 and new_qty < 0:
            total_cost = self.avg_price * abs(self.qty) + fill.price * abs(signed_qty)
            new_avg = total_cost / abs(new_qty)
        else:
            # Crossing through zero: realized PnL handled at portfolio level; carry new side at fill price
            new_avg = fill.price

        return Position(symbol=self.symbol, qty=new_qty, avg_price=new_avg)
