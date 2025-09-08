import os
from pathlib import Path
from functools import lru_cache
from dataclasses import dataclass, field
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

@lru_cache(maxsize=1)
def get_root() -> Path:
    """
    Resolve the project root with the following precedence:
    1) WORKDIR/PROJECT_ROOT env var (from .env or process env)
    2) Walk up from this file until a marker is found (pyproject.toml, .git, etc.)
    3) If inside a 'src' layout, use parent of 'src'
    4) Fallback to current working directory
    """
    # 1) explicit override
    env_root = os.getenv("WORKDIR") or os.getenv("PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    # 2) marker-based discovery from this file
    here = Path(__file__).resolve()
    markers = ("pyproject.toml", ".git")
    for candidate in (here, *here.parents):
        if any((candidate / m).exists() for m in markers):
            return candidate

    # 3) 'src' layout
    for parent in here.parents:
        if parent.name == "src":
            return parent.parent

    # 4) conservative fallback
    return Path.cwd().resolve()


def project_path(*parts: str) -> Path:
    """Convenience: build a path from the resolved project root."""
    return get_root().joinpath(*parts)


# --- update Config to use field(default_factory=...) for path-like defaults ---
@dataclass(frozen=True)
class Config:
    # Operating environment
    ENV: str = os.getenv("ENV", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Data/Network
    YF_TIMEOUT: int = _get_int("YF_TIMEOUT", 10)
    HTTP_TIMEOUT: int = _get_int("HTTP_TIMEOUT", 10)

    # Trading flags
    LIVE_TRADING: bool = _get_bool("LIVE_TRADING", False)

    # Broker
    BROKER_API_KEY: str = os.getenv("BROKER_API_KEY", "")
    BROKER_API_SECRET: str = os.getenv("BROKER_API_SECRET", "")

    # Paths (stringified for broad compatibility)
    ROOT_DIR: str = field(default_factory=lambda: str(get_root()))
    DATA_DIR: str = field(default_factory=lambda: str(get_root() / ".data"))
    DB_URI: str = os.getenv(
        "DB_URI",
        # keep default local sqlite in the data directory; always absolute & portable
        f"sqlite:///{(get_root() / '.data' / 'local.db').as_posix()}"
    )

def _validate(cfg: Config) -> None:
    if cfg.LIVE_TRADING and (not cfg.BROKER_API_KEY or not cfg.BROKER_API_SECRET):
        raise RuntimeError("LIVE_TRADING=true, but BROKER_API_KEY/SECRET missing.")
    if cfg.LOG_LEVEL.upper() not in {"CRITICAL","ERROR","WARNING","INFO","DEBUG"}:
        raise RuntimeError(f"Invalid LOG_LEVEL: {cfg.LOG_LEVEL}")

CFG = Config()
_validate(CFG)