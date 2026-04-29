import asyncio

from loader import bot, dp
from database import db

import handlers


async def main():
    await db.create_tables()
    await dp.start_polling(bot)


if __name__ == '__main__':
    asyncio.run(main())

