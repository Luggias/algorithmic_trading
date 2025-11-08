# Regression-based trading strategy with model selection and hyperparameter optimization

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Sequence

import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    TimeSeriesSplit,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.bot.strategy import Strategy, StrategyContext
from src.common.types import Bar, Order, OrderType, Position, Side

logger = logging.getLogger(__name__)


class RegressionType(str, Enum):
    """Supported regression model types."""

    LINEAR = "linear"
    RIDGE = "ridge"
    LASSO = "lasso"


@dataclass
class RegressionConfig:
    """Configuration for regression strategy."""

    symbol: str
    regression_type: RegressionType = RegressionType.LINEAR
    lookback: int = 20  # Number of lagged features
    min_train_bars: int = 100  # Minimum bars before trading
    retrain_every: int = 20  # Retrain model every N bars
    threshold_bps: float = 5.0  # Only trade if |prediction| > threshold (basis points)
    position_size_pct: float = 0.1  # Position size as % of equity
    use_cross_validation: bool = True  # Use CV for model evaluation
    cv_folds: int = 5  # Number of CV folds
    optimize_hyperparameters: bool = False  # Enable hyperparameter optimization
    hyperparameter_search: Literal["grid", "random"] = "grid"  # Search method
    n_iter_random: int = 50  # Iterations for random search

    # Ridge/Lasso hyperparameter ranges
    alpha_range: tuple[float, float] = (0.001, 100.0)  # Regularization strength range
    alpha_n_values: int = 20  # Number of alpha values to test

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.lookback < 1:
            raise ValueError("lookback must be >= 1")
        if self.min_train_bars < self.lookback:
            raise ValueError("min_train_bars must be >= lookback")
        if self.threshold_bps < 0:
            raise ValueError("threshold_bps must be >= 0")
        if not (0 < self.position_size_pct <= 1.0):
            raise ValueError("position_size_pct must be in (0, 1]")


class RegressionStrategy(Strategy):
    """
    Regression-based trading strategy with model selection and hyperparameter optimization.

    Supports:
    - Linear Regression
    - Ridge Regression (L2 regularization)
    - Lasso Regression (L1 regularization)
    - Cross-validation for model evaluation
    - Hyperparameter optimization (GridSearch/RandomSearch)
    - Walk-forward retraining

    The strategy:
    1. Extracts lagged return features from historical bars
    2. Predicts next-period return
    3. Trades when prediction exceeds threshold
    4. Retrains model periodically
    """

    def __init__(self, config: RegressionConfig) -> None:
        """
        Initialize regression strategy.

        Args:
            config: Strategy configuration
        """
        self.config = config
        self.model: LinearRegression | Ridge | Lasso | None = None
        self.scaler: StandardScaler | None = None
        self.pipeline: Pipeline | None = None
        self.bar_buffer: list[Bar] = []
        self.last_retrain_bar: int = -1
        self.cv_scores: list[float] = []
        self.best_params: dict | None = None

    def initialize(self, initial_bars: Sequence[Bar]) -> None:
        """
        Initialize strategy with historical data.

        Fits initial model if enough data is available.

        Args:
            initial_bars: Historical bars for initialization
        """
        self.bar_buffer = list(initial_bars)
        logger.info(
            f"Initializing {self.config.regression_type.value} strategy for {self.config.symbol} "
            f"with {len(initial_bars)} initial bars"
        )

        if len(self.bar_buffer) >= self.config.min_train_bars:
            self._fit_model()
            logger.info(
                f"Initial model fitted. CV score: {np.mean(self.cv_scores):.4f} "
                f"if available"
            )

    def on_bar(self, context: StrategyContext) -> Sequence[Order] | None:
        """
        Process new bar and generate trading signal.

        Args:
            context: Current market context

        Returns:
            List of orders to execute, or None
        """
        # Update buffer
        self.bar_buffer.append(context.current_bar)
        if len(self.bar_buffer) > self.config.min_train_bars * 2:
            # Keep buffer size reasonable
            self.bar_buffer.pop(0)

        # Need enough history to trade
        if len(self.bar_buffer) < self.config.min_train_bars:
            return None

        # Retrain if needed
        bars_since_retrain = len(self.bar_buffer) - self.last_retrain_bar
        if bars_since_retrain >= self.config.retrain_every:
            self._fit_model()

        # Extract features and predict
        features = self._extract_features(self.bar_buffer[-self.config.lookback :])
        if features is None:
            return None

        prediction = self._predict(features)
        if prediction is None:
            return None

        # Generate order if signal strong enough
        return self._signal_to_orders(prediction, context)

    def _extract_features(self, bars: Sequence[Bar]) -> np.ndarray | None:
        """
        Extract lagged return features from bars.

        Args:
            bars: Sequence of bars (should be at least lookback length)

        Returns:
            Feature vector (1D array) or None if insufficient data
        """
        if len(bars) < self.config.lookback:
            return None

        # Calculate log returns
        closes = np.array([bar.close for bar in bars])
        log_returns = np.diff(np.log(closes))

        if len(log_returns) < self.config.lookback:
            return None

        # Extract lagged features (t-1, t-2, ..., t-lookback)
        features = log_returns[-self.config.lookback :]
        return features.reshape(1, -1)

    def _prepare_training_data(self) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Prepare training data (features and targets) from bar buffer.

        Returns:
            (X, y) tuple where:
            - X: feature matrix (n_samples, n_features)
            - y: target vector (n_samples,) - next period returns
            Or None if insufficient data
        """
        if len(self.bar_buffer) < self.config.lookback + 1:
            return None

        closes = np.array([bar.close for bar in self.bar_buffer])
        log_returns = np.diff(np.log(closes))

        if len(log_returns) < self.config.lookback + 1:
            return None

        n_samples = len(log_returns) - self.config.lookback
        X = np.zeros((n_samples, self.config.lookback))
        y = np.zeros(n_samples)

        for i in range(n_samples):
            X[i] = log_returns[i : i + self.config.lookback]
            y[i] = log_returns[i + self.config.lookback]

        return X, y

    def _create_model(self) -> LinearRegression | Ridge | Lasso:
        """Create model instance based on regression type."""
        if self.config.regression_type == RegressionType.LINEAR:
            return LinearRegression()
        elif self.config.regression_type == RegressionType.RIDGE:
            # Default alpha, will be optimized if enabled
            return Ridge(alpha=1.0)
        elif self.config.regression_type == RegressionType.LASSO:
            return Lasso(alpha=1.0, max_iter=1000)
        else:
            raise ValueError(f"Unknown regression type: {self.config.regression_type}")

    def _get_hyperparameter_grid(self) -> dict:
        """Get hyperparameter grid for optimization."""
        if self.config.regression_type == RegressionType.LINEAR:
            return {}  # No hyperparameters for linear regression

        # Create alpha range
        alpha_min, alpha_max = self.config.alpha_range
        alphas = np.logspace(
            np.log10(alpha_min), np.log10(alpha_max), self.config.alpha_n_values
        )

        if self.config.regression_type == RegressionType.RIDGE:
            return {"regressor__alpha": alphas}
        elif self.config.regression_type == RegressionType.LASSO:
            return {"regressor__alpha": alphas}
        else:
            return {}

    def _fit_model(self) -> None:
        """Fit or retrain the regression model."""
        training_data = self._prepare_training_data()
        if training_data is None:
            logger.warning("Insufficient data for model training")
            return

        X, y = training_data
        logger.debug(f"Training model on {len(X)} samples")

        # Create pipeline with scaling and regression
        regressor = self._create_model()
        self.pipeline = Pipeline([("scaler", StandardScaler()), ("regressor", regressor)])

        # Hyperparameter optimization
        if self.config.optimize_hyperparameters:
            param_grid = self._get_hyperparameter_grid()
            if param_grid:
                # Use time series cross-validation (no shuffling)
                cv = TimeSeriesSplit(n_splits=self.config.cv_folds)

                if self.config.hyperparameter_search == "grid":
                    search = GridSearchCV(
                        self.pipeline,
                        param_grid,
                        cv=cv,
                        scoring="neg_mean_squared_error",
                        n_jobs=-1,
                    )
                else:  # random
                    search = RandomizedSearchCV(
                        self.pipeline,
                        param_grid,
                        cv=cv,
                        scoring="neg_mean_squared_error",
                        n_jobs=-1,
                        n_iter=self.config.n_iter_random,
                    )

                search.fit(X, y)
                self.pipeline = search.best_estimator_
                self.best_params = search.best_params_
                logger.info(
                    f"Hyperparameter optimization complete. Best params: {self.best_params}"
                )
            else:
                # No hyperparameters to optimize, just fit
                self.pipeline.fit(X, y)
        else:
            # No optimization, just fit
            self.pipeline.fit(X, y)

        # Cross-validation for model evaluation
        if self.config.use_cross_validation:
            cv = TimeSeriesSplit(n_splits=self.config.cv_folds)
            scores = cross_val_score(
                self.pipeline, X, y, cv=cv, scoring="neg_mean_squared_error"
            )
            self.cv_scores = -scores.tolist()  # Convert to positive MSE
            logger.info(
                f"CV MSE: {np.mean(self.cv_scores):.6f} (+/- {np.std(self.cv_scores):.6f})"
            )

        self.last_retrain_bar = len(self.bar_buffer)
        logger.debug("Model training complete")

    def _predict(self, features: np.ndarray) -> float | None:
        """
        Predict next period return.

        Args:
            features: Feature vector (1, n_features)

        Returns:
            Predicted return, or None if model not ready
        """
        if self.pipeline is None:
            return None

        try:
            prediction = self.pipeline.predict(features)[0]
            return float(prediction)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None

    def _signal_to_orders(
        self, prediction: float, context: StrategyContext
    ) -> Sequence[Order] | None:
        """
        Convert prediction signal to trading orders.

        Args:
            prediction: Predicted next-period return
            context: Current market context

        Returns:
            List of orders, or None
        """
        # Convert prediction to basis points
        prediction_bps = prediction * 10_000

        # Check threshold
        if abs(prediction_bps) < self.config.threshold_bps:
            return None

        # Get current position
        current_pos = context.positions.get(
            self.config.symbol,
            Position(symbol=self.config.symbol, qty=0.0, avg_price=0.0),
        )

        # Calculate desired position size
        position_value = context.equity * self.config.position_size_pct
        current_price = context.current_bar.close
        desired_qty = position_value / current_price

        # Determine side based on prediction
        if prediction_bps > 0:
            # Positive prediction -> buy
            desired_qty = abs(desired_qty)
            side = Side.BUY
        else:
            # Negative prediction -> sell (short)
            desired_qty = -abs(desired_qty)
            side = Side.SELL

        # Calculate order quantity (difference from current position)
        order_qty = desired_qty - current_pos.qty

        # Only trade if difference is significant
        if abs(order_qty) < 1e-9:
            return None

        # Create order
        order = Order(
            symbol=self.config.symbol,
            side=side,
            qty=abs(order_qty),
            order_type=OrderType.MARKET,
            created_at=context.timestamp,
            tags=("regression", self.config.regression_type.value),
        )

        return [order]

    def get_state(self) -> dict:
        """Get strategy state for serialization."""
        return {
            "config": {
                "symbol": self.config.symbol,
                "regression_type": self.config.regression_type.value,
                "lookback": self.config.lookback,
                "min_train_bars": self.config.min_train_bars,
                "retrain_every": self.config.retrain_every,
                "threshold_bps": self.config.threshold_bps,
                "position_size_pct": self.config.position_size_pct,
            },
            "last_retrain_bar": self.last_retrain_bar,
            "cv_scores": self.cv_scores,
            "best_params": self.best_params,
            # Note: Model weights would need to be serialized separately
            # (sklearn models have their own serialization methods)
        }

    def cleanup(self) -> None:
        """Cleanup resources."""
        self.bar_buffer.clear()
        self.model = None
        self.scaler = None
        self.pipeline = None

