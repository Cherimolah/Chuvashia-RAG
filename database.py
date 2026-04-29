from sqlalchemy.orm import DeclarativeBase, Mapped, relationship
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy import Column, BigInteger, String, Integer, Boolean, select, update, ForeignKey, delete

from config import PG_USER, PG_PASSWORD, PG_HOST, PG_PORT, DB_NAME


DATABASE_URL = f'postgresql+asyncpg://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{DB_NAME}'


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    user_id: Mapped[int] = Column(BigInteger, primary_key=True)
    username: Mapped[str] = Column(String)
    full_name: Mapped[str] = Column(String)

    messages = relationship("Message", back_populates="user", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = Column(Integer, primary_key=True)
    text: Mapped[str] = Column(String)
    is_from_user: Mapped[bool] = Column(Boolean)
    user_id: Mapped[int] = Column(BigInteger, ForeignKey('users.user_id'))

    user = relationship("User", back_populates="messages")


class Database:

    def __init__(self, url: str):
        self.engine = create_async_engine(url, echo=False)
        self.session = async_sessionmaker(self.engine, expire_on_commit=False)

    async def create_tables(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def create_user(self, user_id: str, username: str, full_name: str):
        async with self.session() as session:
            user = User(user_id=user_id, username=username, full_name=full_name)
            session.add(user)
            await session.commit()

    async def get_user_by_id(self, user_id: int) -> User:
        async with self.session() as session:
            response = await session.scalar(select(User).where(User.user_id == user_id))
            return response

    async def update_user_by_id(self, user_id: int, username: str, full_name: str):
        async with self.session() as session:
            await session.execute(update(User).where(User.user_id == user_id).values(full_name=full_name, username=username))
            await session.commit()

    async def create_message(self, user_id: int, text: str, is_from_user: bool):
        async with self.session() as session:
            message = Message(user_id=user_id, text=text, is_from_user=is_from_user)
            session.add(message)
            await session.commit()

    async def get_context(self, user_id: int) -> list[Message]:
        async with self.session() as session:
            response = await session.scalars(select(Message).where(Message.user_id == user_id))
            return list(response.all())

    async def delete_context(self, user_id: int):
        async with self.session() as session:
            await session.execute(delete(Message).where(Message.user_id == user_id))
            await session.commit()


db = Database(DATABASE_URL)
