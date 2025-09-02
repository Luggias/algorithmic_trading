import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load .env once – quietly, without prints/logs
load_dotenv()

def _get_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}

def _get_int(key: str, default: int) -> int:
    val = os.getenv(key)
    try:
        return int(val) if val is not None else default
    except ValueError:
        return default

@dataclass(frozen=True)
class Config:
    # Operating environment
    ENV: str = os.getenv("ENV", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Data/Network
    YF_TIMEOUT: int = _get_int("YF_TIMEOUT", 10)     # yfinance timeout (seconds)
    HTTP_TIMEOUT: int = _get_int("HTTP_TIMEOUT", 10)

    # Trading flags
    LIVE_TRADING: bool = _get_bool("LIVE_TRADING", False)

    # Broker (only required if LIVE_TRADING=true)
    BROKER_API_KEY: str = os.getenv("BROKER_API_KEY", "")
    BROKER_API_SECRET: str = os.getenv("BROKER_API_SECRET", "")

    # Storage
    DB_URI: str = os.getenv("DB_URI", "sqlite:///local.db")
    DATA_DIR: str = os.getenv("DATA_DIR", "./.data")

def _validate(cfg: Config) -> None:
    # Enforce hard requirements only if relevant.
    if cfg.LIVE_TRADING and (not cfg.BROKER_API_KEY or not cfg.BROKER_API_SECRET):
        raise RuntimeError("LIVE_TRADING=true, but BROKER_API_KEY/SECRET missing.")
    # Optional: whitelist LOG_LEVEL
    if cfg.LOG_LEVEL.upper() not in {"CRITICAL","ERROR","WARNING","INFO","DEBUG"}:
        raise RuntimeError(f"Invalid LOG_LEVEL: {cfg.LOG_LEVEL}")

# Global, immutable instance
CFG = Config()
_validate(CFG)