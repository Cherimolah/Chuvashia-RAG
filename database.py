from sqlalchemy.orm import DeclarativeBase, Mapped, relationship
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy import (
    Column, BigInteger, String, Integer, Boolean,
    select, update, ForeignKey, delete,
)

from config import PG_USER, PG_PASSWORD, PG_HOST, PG_PORT, DB_NAME
from logger import get_logger, preview

log = get_logger(__name__)

DATABASE_URL = (
    f'postgresql+asyncpg://{PG_USER}:{PG_PASSWORD}'
    f'@{PG_HOST}:{PG_PORT}/{DB_NAME}'
)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    user_id: Mapped[int] = Column(BigInteger, primary_key=True)
    username: Mapped[str] = Column(String)
    full_name: Mapped[str] = Column(String)
    messages = relationship(
        "Message", back_populates="user", cascade="all, delete-orphan"
    )


class Message(Base):
    __tablename__ = "messages"
    id: Mapped[int] = Column(Integer, primary_key=True)
    text: Mapped[str] = Column(String)
    is_from_user: Mapped[bool] = Column(Boolean)
    user_id: Mapped[int] = Column(BigInteger, ForeignKey('users.user_id'))
    user = relationship("User", back_populates="messages")


class Database:
    def __init__(self, url: str):
        # В лог идёт только безопасная часть — без пароля
        safe = f'{PG_HOST}:{PG_PORT}/{DB_NAME} (user={PG_USER})'
        log.info(f'Инициализация Database, цель: {safe}')
        self.engine = create_async_engine(url, echo=False)
        self.session = async_sessionmaker(self.engine, expire_on_commit=False)

    async def create_tables(self):
        log.debug('create_tables: создаю таблицы (если ещё нет)…')
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        log.info('create_tables: ✅ готово')

    async def create_user(self, user_id, username: str, full_name: str):
        log.info(
            f'DB → create_user(user_id={user_id}, '
            f'username={username!r}, full_name={full_name!r})'
        )
        async with self.session() as session:
            user = User(user_id=user_id, username=username, full_name=full_name)
            session.add(user)
            await session.commit()
        log.debug(f'DB ← create_user: пользователь {user_id} сохранён')

    async def get_user_by_id(self, user_id: int) -> User:
        log.debug(f'DB → get_user_by_id(user_id={user_id})')
        async with self.session() as session:
            response = await session.scalar(
                select(User).where(User.user_id == user_id)
            )
        log.debug(
            f'DB ← get_user_by_id({user_id}): '
            f'{"найден" if response else "не найден"}'
        )
        return response

    async def update_user_by_id(self, user_id: int, username: str, full_name: str):
        log.debug(
            f'DB → update_user_by_id(user_id={user_id}, '
            f'username={username!r}, full_name={full_name!r})'
        )
        async with self.session() as session:
            await session.execute(
                update(User)
                .where(User.user_id == user_id)
                .values(full_name=full_name, username=username)
            )
            await session.commit()
        log.debug(f'DB ← update_user_by_id: данные {user_id} обновлены')

    async def create_message(self, user_id: int, text: str, is_from_user: bool):
        role = 'user' if is_from_user else 'bot'
        log.info(
            f'DB → create_message(user_id={user_id}, role={role}, '
            f'len={len(text)}): {preview(text, 80)}'
        )
        async with self.session() as session:
            message = Message(
                user_id=user_id, text=text, is_from_user=is_from_user
            )
            session.add(message)
            await session.commit()
            msg_id = message.id
        log.debug(f'DB ← create_message: id сообщения {msg_id}')

    async def get_context(self, user_id: int) -> list[Message]:
        log.debug(f'DB → get_context(user_id={user_id})')
        async with self.session() as session:
            response = await session.scalars(
                select(Message).where(Message.user_id == user_id)
            )
            messages = list(response.all())
        log.info(
            f'DB ← get_context: {len(messages)} сообщений в контексте '
            f'user={user_id}'
        )
        return messages

    async def delete_context(self, user_id: int):
        log.info(f'DB → delete_context(user_id={user_id})')
        async with self.session() as session:
            result = await session.execute(
                delete(Message).where(Message.user_id == user_id)
            )
            await session.commit()
        deleted = getattr(result, 'rowcount', '?')
        log.info(
            f'DB ← delete_context: удалено {deleted} сообщений '
            f'у user={user_id}'
        )


db = Database(DATABASE_URL)
