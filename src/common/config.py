import os
from dotenv import load_dotenv

# Load from .env file
load_dotenv()

# TODO: Example Code for class Config:

class Config:
    ENV: str = os.getenv("ENV", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Data sources
    YAHOO_TIMEOUT: int = int(os.getenv("YAHOO_TIMEOUT", 5))

    # Broker / Exchange
    BROKER_API_KEY: str = os.getenv("BROKER_API_KEY", "")
    BROKER_API_SECRET: str = os.getenv("BROKER_API_SECRET", "")

    # Database
    DB_URI: str = os.getenv("DB_URI", "sqlite:///local.db")

# TODO Implement usage everywhere: 
#   from common.config import Config
#   print(Config.BROKER_API_KEY)  # use in backend, bot, backtesting