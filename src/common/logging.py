# Centralized logging config with structured JSON output

import logging
from datetime import datetime, timezone
from typing import Any

# Fallback format for plain text logging
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

try:
    from pythonjsonlogger import jsonlogger
    _HAS_JSON_LOGGER = True
except ImportError:
    _HAS_JSON_LOGGER = False


if _HAS_JSON_LOGGER:
    class StructuredFormatter(jsonlogger.JsonFormatter):
        """JSON formatter that ensures UTC timestamps and structured output."""

        def add_fields(self, log_record: dict[str, Any], record: logging.LogRecord, message_dict: dict[str, Any]) -> None:
            super().add_fields(log_record, record, message_dict)
            # Ensure timestamp is ISO8601 UTC
            if "timestamp" not in log_record:
                log_record["timestamp"] = datetime.now(timezone.utc).isoformat()
            log_record["level"] = record.levelname
            log_record["logger"] = record.name
            # Never log secrets
            if "password" in log_record:
                log_record["password"] = "***"
            if "secret" in log_record:
                log_record["secret"] = "***"
            if "api_key" in log_record:
                log_record["api_key"] = "***"


def configure_logging(level: str = "INFO", use_json: bool = True) -> None:
    """
    Configure the root logger with structured JSON logging.

    Args:
        level: Log level (CRITICAL, ERROR, WARNING, INFO, DEBUG)
        use_json: If True, use JSON formatter (requires python-json-logger).
                  Falls back to plain text if package not available.

    Should be called once at application startup (e.g. in main.py).
    """
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    root.setLevel(level.upper())

    handler = logging.StreamHandler()

    if use_json and _HAS_JSON_LOGGER:
        formatter = StructuredFormatter(
            "%(timestamp)s %(level)s %(logger)s %(message)s",
            timestamp=True
        )
    else:
        if not _HAS_JSON_LOGGER:
            # Log warning about missing json logger, but only once
            if not hasattr(configure_logging, "_warned"):
                print("WARNING: python-json-logger not installed. Using plain text logging.")
                configure_logging._warned = True
        formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    handler.setFormatter(formatter)
    root.addHandler(handler)

def get_logger(name: str, **context: Any) -> logging.Logger:
    """
    Return a logger; optionally with static context.
    Context is passed as 'extra'; formatter can use these fields.
    """
    logger = logging.getLogger(name)
    if context:
        return logging.LoggerAdapter(logger, extra=context)  # type: ignore[return-value]
    return logger