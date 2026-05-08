import asyncio

from logger import get_logger
from loader import bot, dp
from database import db
import handlers  # noqa: F401 — нужен для регистрации хендлеров

log = get_logger(__name__)


async def main():
    log.info('🚀 Запуск Chuvashia-RAG бота')
    log.info('Создание таблиц БД (если ещё не созданы)…')
    await db.create_tables()
    log.info('Таблицы готовы. Старт polling…')
    try:
        await dp.start_polling(bot)
    finally:
        log.info('🛑 Бот остановлен')


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        log.info('Получен сигнал остановки')
