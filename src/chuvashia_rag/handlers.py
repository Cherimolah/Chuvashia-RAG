import contextlib

from aiogram import F
from aiogram.filters.command import CommandStart
from aiogram.types import KeyboardButton, Message, ReplyKeyboardMarkup

from chuvashia_rag.config import N_RESULTS
from chuvashia_rag.database import db
from chuvashia_rag.llm import extract_period, get_embedding, get_response, rag_prompt
from chuvashia_rag.loader import collection, dp
from chuvashia_rag.logger import get_logger, preview, timed

log = get_logger(__name__)


@dp.message(CommandStart())
async def hello(m: Message):
    if m.from_user is None:  # сообщение от имени канала / анонимного админа
        return
    log.info(f"/start от user_id={m.from_user.id} (@{m.from_user.username})")
    keyboard = ReplyKeyboardMarkup(
        resize_keyboard=True, keyboard=[[KeyboardButton(text="🗑 Сбросить контекст")]]
    )
    await m.answer(
        "Сывлӑх сунатӑп! Эпӗ — чӑваш культурин ӑста-ботӗ. "
        "Чӑваш халӑхӗн пуян еткерӗ, йӑли-йӗрки, чӗлхи, кӗвви-ҫемми, "
        "мифологийӗ тата историне тӗпчессипе савӑнӑҫлӑ пулӑшатӑп. "
        "Чӑваш тӗнчине хушӑр! Мӗнле ыйтусем пур сирӗн?",
        reply_markup=keyboard,
    )


@dp.message(F.text == "🗑 Сбросить контекст")
async def clear_context(m: Message):
    if m.from_user is None:
        return
    log.info(f"🗑  Сброс контекста для user_id={m.from_user.id}")
    await db.delete_context(m.from_user.id)
    await m.answer("✅ Контекста пӑрахӑҫланӑ")


@dp.message(F.text)
async def echo(m: Message):
    if m.from_user is None or m.text is None:  # text гарантирован фильтром F.text
        return
    user_id = m.from_user.id
    log.info(
        f"📨 Сообщение от user_id={user_id} (@{m.from_user.username}), "
        f"len={len(m.text)}: {preview(m.text, 100)}"
    )

    waiting_message = await m.answer("⏳ Хурав кӗтӗр")

    try:
        # 0. Если пользователь запрашивает период работаем с ним
        period_response = await extract_period(m.text)
        if period_response.need_period:
            period = period_response.period
            log.debug(
                "🤖 Агент решил выделить поиск по дате: "
                f"Начало {period.start_turn.strftime('%d.%m.%Y %H:%M')} "
                f"конец {period.end_turn.strftime('%d.%m.%Y %H:%M')}"
            )
            where_filter = {
                "$and": [
                    {"date": {"$gte": period_response.period.start_turn.timestamp()}},
                    {"date": {"$lte": period_response.period.end_turn.timestamp()}},
                ]
            }
        else:
            where_filter = None

        # 1. Загружаем историю диалога из БД
        history = await db.get_context(user_id)
        messages = [
            {"role": "user" if x.is_from_user else "assistant", "content": x.text} for x in history
        ]
        log.debug(f"История диалога подготовлена: {len(messages)} сообщений")
        messages.append({"role": "user", "content": m.text})

        # 2. Эмбеддинг (история + новый запрос)
        with timed(log, "эмбеддинг диалога"):
            embeddings = await get_embedding(messages)

        # 3. Поиск похожих фрагментов в ChromaDB
        with timed(log, "поиск в ChromaDB"):
            response = collection.query(
                query_embeddings=embeddings, n_results=N_RESULTS, where=where_filter
            )

        docs = response.get("documents", [[]])[0]
        distances = (response.get("distances") or [[]])[0]
        ids = (response.get("ids") or [[]])[0]
        log.info(f"🔎 ChromaDB вернул {len(docs)} документов")
        for i, doc in enumerate(docs):
            dist = f" dist={distances[i]:.4f}" if i < len(distances) else ""
            doc_id = f" id={ids[i]}" if i < len(ids) else ""
            log.debug(f"  [{i}]{doc_id}{dist}: {preview(doc, 150)}")

        context = ""
        for article in docs:
            context += f"{article}\n\n"

        log.debug(f"Склеенный контекст для RAG: {len(context)} симв.")

        # 4. Заменяем последний user-message на промпт с контекстом
        messages.pop(-1)
        final_prompt = rag_prompt.format(question=m.text, context=context)
        messages.append({"role": "user", "content": final_prompt})
        log.debug(
            f"Финальный промпт собран ({len(final_prompt)} симв.): {preview(final_prompt, 200)}"
        )

        # 5. Сохраняем запрос пользователя в БД
        await db.create_message(user_id, m.text, is_from_user=True)

        # 6. Запрос к LLM
        with timed(log, "генерация ответа LLM"):
            answer = await get_response(messages)
        log.info(f"🤖 Ответ LLM получен, len={len(answer)}: {preview(answer, 120)}")

        await waiting_message.delete()
        await db.create_message(user_id, answer, is_from_user=False)
        await m.answer(answer)
        log.info(f"✅ Ответ отправлен user_id={user_id}")

    except Exception:
        log.exception(f"❌ Ошибка в обработке сообщения user_id={user_id}")
        with contextlib.suppress(Exception):
            await waiting_message.delete()
        await m.answer("⚠️ Каҫарӑр, тем йӑнӑш пулчӗ. Тепре тӑрӑшӑр.")
