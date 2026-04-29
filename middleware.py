from typing import Callable, Dict, Any, Awaitable

from aiogram import BaseMiddleware
from aiogram.types import TelegramObject, User

from database import db


class UserMiddleware(BaseMiddleware):
    async def __call__(
            self,
            handler: Callable[[TelegramObject, Dict[str, Any]], Awaitable[Any]],
            event: TelegramObject,
            data: Dict[str, Any]
    ) -> Any:
        # Определяем пользователя
        user: User = data.get('event_from_user')
        exist = await db.get_user_by_id(user.id)
        if not exist:
            await db.create_user(user.id, user.username, user.full_name)
        else:
            await db.update_user_by_id(user.id, user.username, user.full_name)
        response = await handler(event, data)
        return response

