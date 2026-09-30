from typing import Callable, Dict, Any, Awaitable

from aiogram import BaseMiddleware
from aiogram.types import TelegramObject, User

from database import db
from logger import get_logger

log = get_logger(__name__)


class UserMiddleware(BaseMiddleware):
    async def __call__(
            self,
            handler: Callable[[TelegramObject, Dict[str, Any]], Awaitable[Any]],
            event: TelegramObject,
            data: Dict[str, Any]
    ) -> Any:
        user: User = data.get('event_from_user')
        if user is None:
            log.warning('Middleware: event_from_user отсутствует, пропускаю')
            return await handler(event, data)

        exist = await db.get_user_by_id(user.id)
        if not exist:
            log.info(
                f'👤 Новый пользователь: id={user.id}, '
                f'username=@{user.username}, name={user.full_name!r}'
            )
            await db.create_user(user.id, user.username, user.full_name)
        else:
            log.debug(
                f'👤 Пользователь уже есть: id={user.id}, обновляю профиль '
                f'(@{user.username}, {user.full_name!r})'
            )
            await db.update_user_by_id(user.id, user.username, user.full_name)

        return await handler(event, data)
