# Base Strategy class and context for trading strategies

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from src.common.types import Bar, Order, Position


@dataclass
class StrategyContext:
    """Context provided to strategy on each bar update."""

    current_bar: Bar
    history: Sequence[Bar]  # Recent bars (typically last 100-500 bars)
    positions: dict[str, Position]  # symbol -> current position
    cash: float  # Available cash
    equity: float  # Total portfolio equity (cash + positions)
    timestamp: datetime  # Current timestamp (UTC)

    def get_history(self, n_bars: int) -> Sequence[Bar]:
        """
        Get the last N bars from history.
        Returns at most what's available in history.
        """
        return self.history[-n_bars:] if len(self.history) >= n_bars else self.history


class Strategy(ABC):
    """
    Base class for trading strategies.

    Strategies are stateful and maintain internal state (models, indicators, buffers).
    State can be serialized for reproducibility in backtests.
    """

    @abstractmethod
    def initialize(self, initial_bars: Sequence[Bar]) -> None:
        """
        Initialize strategy with initial historical data.

        Called once before the first bar is processed.
        Use this to:
        - Fit initial models
        - Initialize indicators
        - Set up buffers

        Args:
            initial_bars: Historical bars for warmup/initialization
        """
        ...

    @abstractmethod
    def on_bar(self, context: StrategyContext) -> Sequence[Order] | None:
        """
        Process a new bar and generate trading orders.

        This is called for each new bar in chronological order.
        The strategy should:
        1. Update internal state (models, indicators)
        2. Generate trading signals
        3. Return orders to execute

        Args:
            context: Current market context with bar, history, positions, portfolio state

        Returns:
            Sequence of orders to execute, or None if no action
        """
        ...

    def cleanup(self) -> None:
        """
        Cleanup resources (optional).

        Called when strategy is no longer needed.
        Use this to save state, close resources, etc.
        """
        pass

    def get_state(self) -> dict:
        """
        Get strategy state for serialization (optional).

        Used for reproducibility in backtests.
        Should return a serializable dict with all state needed to restore the strategy.

        Returns:
            Dictionary containing strategy state
        """
        return {}

    def set_state(self, state: dict) -> None:
        """
        Restore strategy state (optional).

        Used to restore strategy from serialized state for reproducible backtests.

        Args:
            state: Dictionary containing strategy state (from get_state())
        """
        pass
