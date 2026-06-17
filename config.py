import os

from dotenv import load_dotenv


load_dotenv()


TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

OPENROUTER_TOKEN = os.getenv("OPENROUTER_TOKEN")

PG_USER = os.getenv("PG_USER")
PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_HOST = os.getenv("PG_HOST")
PG_PORT = os.getenv("PG_PORT")
DB_NAME = os.getenv("DB_NAME")
LOG_LEVEL = os.getenv("LOG_LEVEL")

TEMPERATURE       = float(os.getenv("TEMPERATURE", 0.4))
MAX_TOKENS        = int(os.getenv("MAX_TOKENS", 10000))
TOP_P             = float(os.getenv("TOP_P", 0.9))
FREQUENCY_PENALTY = float(os.getenv("FREQUENCY_PENALTY", 0.1))
PRESENCE_PENALTY  = float(os.getenv("PRESENCE_PENALTY", 0.1))
N_RESULTS         = int(os.getenv("N_RESULTS", 5))
