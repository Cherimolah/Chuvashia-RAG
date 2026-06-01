import random

from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton
from aiogram.filters.command import CommandStart
from aiogram import F

from loader import dp, collection, chroma_client
from llm import get_embedding, rag_prompt, get_response, extract_period
from database import db
from config import N_RESULTS
from logger import get_logger, preview, timed

log = get_logger(__name__)


@dp.message(CommandStart())
async def hello(m: Message):
    log.info(
        f'/start от user_id={m.from_user.id} (@{m.from_user.username})'
    )
    keyboard = ReplyKeyboardMarkup(resize_keyboard=True, keyboard=[[
        KeyboardButton(text='🗑 Сбросить контекст')
    ]])
    await m.answer(
        'Сывлӑх сунатӑп! Эпӗ — чӑваш культурин ӑста-ботӗ. '
        'Чӑваш халӑхӗн пуян еткерӗ, йӑли-йӗрки, чӗлхи, кӗвви-ҫемми, '
        'мифологийӗ тата историне тӗпчессипе савӑнӑҫлӑ пулӑшатӑп. '
        'Чӑваш тӗнчине хушӑр! Мӗнле ыйтусем пур сирӗн?',
        reply_markup=keyboard
    )


@dp.message(F.text == '🗑 Сбросить контекст')
async def clear_context(m: Message):
    log.info(f'🗑  Сброс контекста для user_id={m.from_user.id}')
    await db.delete_context(m.from_user.id)
    await m.answer('✅ Контекста пӑрахӑҫланӑ')


@dp.message(F.text)
async def echo(m: Message):
    user_id = m.from_user.id
    log.info(
        f'📨 Сообщение от user_id={user_id} (@{m.from_user.username}), '
        f'len={len(m.text)}: {preview(m.text, 100)}'
    )

    waiting_message = await m.answer('⏳ Хурав кӗтӗр')

    try:
        # 0. Если пользователь запрашивает период работаем с ним
        period_response = await extract_period(m.text)
        if period_response.need_period:
            log.debug(
                f'🤖 Агент решил выделить поиск по дате: Начало {period_response.period.start_turn.strftime("%d.%m.%Y %H:%M")} '
                f'конец {period_response.period.end_turn.strftime("%d.%m.%Y %H:%M")}')
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
            {'role': 'user' if x.is_from_user else 'assistant', 'content': x.text}
            for x in history
        ]
        log.debug(f'История диалога подготовлена: {len(messages)} сообщений')
        messages.append({'role': 'user', 'content': m.text})

        # 2. Эмбеддинг (история + новый запрос)
        with timed(log, 'эмбеддинг диалога'):
            embeddings = await get_embedding(messages)

        # 3. Поиск похожих фрагментов в ChromaDB
        with timed(log, 'поиск в ChromaDB'):
            response = collection.query(
                query_embeddings=embeddings, n_results=N_RESULTS,
                where=where_filter
            )

        docs = response.get('documents', [[]])[0]
        distances = (response.get('distances') or [[]])[0]
        ids = (response.get('ids') or [[]])[0]
        log.info(f'🔎 ChromaDB вернул {len(docs)} документов')
        for i, doc in enumerate(docs):
            dist = f' dist={distances[i]:.4f}' if i < len(distances) else ''
            doc_id = f' id={ids[i]}' if i < len(ids) else ''
            log.debug(f'  [{i}]{doc_id}{dist}: {preview(doc, 150)}')

        unique_urls = set()
        for metadata_list in response["metadatas"]:
            for meta in metadata_list:
                if meta and "url" in meta:
                    unique_urls.add(meta["url"])

        # 3. Для каждого уникального URL получаем все его чанки
        full_articles = {}
        for url in unique_urls:
            # Важно: limit должен быть достаточно большим, чтобы вместить все абзацы статьи.
            # Если статей очень длинные, можно увеличить число или сделать пагинацию.
            chunks = collection.get(
                where={"url": url},
                limit=10_000,  # максимальное число абзацев в одной статье
                include=["documents", "metadatas"]
            )

            # 4. Сортируем чанки по индексу абзаца из поля id (url#номер)
            paragraphs = []
            for doc, meta, doc_id in zip(chunks["documents"], chunks["metadatas"], chunks["ids"]):
                # Извлекаем номер абзаца, предполагая формат "http://...#N"
                try:
                    index = int(doc_id.split("#")[-1])
                except (ValueError, IndexError):
                    index = 0  # fallback
                paragraphs.append((index, doc))

            paragraphs.sort(key=lambda x: x[0])
            full_text = "\n".join(text for _, text in paragraphs)
            full_articles[url] = {'text': full_text, 'title': meta['title']}

        context = ''
        for article in full_articles.values():
            context += f'{article["title"]}\n\n{article["text"]}\n\n\n'

        log.debug(f'Склеенный контекст для RAG: {len(context)} симв.')

        # 4. Заменяем последний user-message на промпт с контекстом
        messages.pop(-1)
        final_prompt = rag_prompt.format(question=m.text, context=context)
        messages.append({'role': 'user', 'content': final_prompt})
        log.debug(
            f'Финальный промпт собран ({len(final_prompt)} симв.): '
            f'{preview(final_prompt, 200)}'
        )

        # 5. Сохраняем запрос пользователя в БД
        await db.create_message(user_id, m.text, is_from_user=True)

        # 6. Запрос к LLM
        with timed(log, 'генерация ответа LLM'):
            answer = await get_response(messages)
        log.info(
            f'🤖 Ответ LLM получен, len={len(answer)}: {preview(answer, 120)}'
        )

        await waiting_message.delete()
        await db.create_message(user_id, answer, is_from_user=False)
        await m.answer(answer)
        log.info(f'✅ Ответ отправлен user_id={user_id}')

    except Exception:
        log.exception(
            f'❌ Ошибка в обработке сообщения user_id={user_id}'
        )
        try:
            await waiting_message.delete()
        except Exception:
            pass
        await m.answer('⚠️ Каҫарӑр, тем йӑнӑш пулчӗ. Тепре тӑрӑшӑр.')
