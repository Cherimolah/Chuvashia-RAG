from aiogram import Bot, Dispatcher
from openai import OpenAI
import chromadb

from config import TELEGRAM_TOKEN, OPENROUTER_TOKEN
from middleware import UserMiddleware
from logger import get_logger

log = get_logger(__name__)

# --- Telegram ---
log.debug('Инициализация Telegram-бота…')
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()
dp.message.outer_middleware(UserMiddleware())
log.info('Telegram-бот и диспетчер созданы, middleware подключён')

# --- OpenRouter ---
log.debug('Инициализация клиента OpenRouter…')
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_TOKEN,
)
log.info('OpenRouter клиент готов (base_url=https://openrouter.ai/api/v1)')

# --- ChromaDB ---
log.debug('Инициализация ChromaDB (path=chroma_db)…')
chroma_client = chromadb.PersistentClient(path='chroma_db')
_collections = chroma_client.list_collections()

if not _collections:
    log.error('В ChromaDB не найдено ни одной коллекции! '
              'Сначала загрузи данные в chroma_db.')
    raise RuntimeError('ChromaDB пуст: нет коллекций')

collection = _collections[0]
try:
    _count = collection.count()
except Exception as exc:
    log.warning(f'Не удалось получить .count() коллекции: {exc}')
    _count = '?'

log.info(f'ChromaDB подключён. Коллекция: "{collection.name}", документов: {_count}')

if len(_collections) > 1:
    others = [c.name for c in _collections]
    log.warning(
        f'В ChromaDB найдено {len(_collections)} коллекций: {others}. '
        f'Используется только первая ("{collection.name}")'
    )
