from aiogram import Bot, Dispatcher
from openai import OpenAI
import chromadb

from config import TELEGRAM_TOKEN, OPENROUTER_TOKEN
from middleware import UserMiddleware


bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()
dp.message.outer_middleware(UserMiddleware())
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=OPENROUTER_TOKEN,
)

chroma_client = chromadb.PersistentClient(path='chroma_db')
collection = chroma_client.list_collections()[0]
