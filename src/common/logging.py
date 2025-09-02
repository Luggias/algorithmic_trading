# Centralized logging config

import logging
from typing import Any

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def configure_logging(level: str = "INFO") -> None:
    """
    Configure the root logger:
    - remove existing handlers (prevents duplicate logs)
    - set the global log level
    - attach a StreamHandler with a consistent formatter
    Should be called once at application startup (e.g. in main.py).
    """
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    root.setLevel(level.upper())

    handler = logging.StreamHandler()
    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)
    handler.setFormatter(formatter)
    root.addHandler(handler)

def get_logger(name: str, **context: Any) -> logging.Logger:
    """
    Liefert einen Logger; optional mit statischem Kontext.
    Kontext wird als 'extra' übergeben; Formatter kann diese Felder nutzen.
    """
    logger = logging.getLogger(name)
    if context:
        return logging.LoggerAdapter(logger, extra=context)  # type: ignore[return-value]
    return logger