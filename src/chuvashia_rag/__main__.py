import asyncio

from chuvashia_rag import handlers  # noqa: F401 — нужен для регистрации хендлеров
from chuvashia_rag.database import db
from chuvashia_rag.loader import bot, dp
from chuvashia_rag.logger import get_logger

log = get_logger(__name__)


async def main():
    log.info("Запуск Chuvashia-RAG бота")
    log.info("Создание таблиц БД (если ещё не созданы)…")
    await db.create_tables()
    log.info("Таблицы готовы. Старт polling…")
    try:
        await dp.start_polling(bot)
    finally:
        log.info("Бот остановлен")


def run():
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        log.info("Получен сигнал остановки")


if __name__ == "__main__":
    run()
