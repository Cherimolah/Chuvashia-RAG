from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton
from aiogram.filters.command import CommandStart
from aiogram import F

from loader import dp, collection
from llm import get_embedding, rag_prompt, get_response
from database import db


@dp.message(CommandStart())
async def hello(m: Message):
    keyboard = ReplyKeyboardMarkup(resize_keyboard=True, keyboard=[[
        KeyboardButton(text='🗑 Сбросить контекст')
    ]])
    await m.answer('Сывлӑх сунатӑп! Эпӗ — чӑваш культурин ӑста-ботӗ. Чӑваш халӑхӗн пуян еткерӗ, йӑли-йӗрки, чӗлхи, кӗвви-ҫемми, мифологийӗ тата историне тӗпчессипе савӑнӑҫлӑ пулӑшатӑп. Чӑваш тӗнчине хушӑр! Мӗнле ыйтусем пур сирӗн?', reply_markup=keyboard)


@dp.message(F.text == '🗑 Сбросить контекст')
async def clear_context(m: Message):
    await db.delete_context(m.from_user.id)
    await m.answer('✅ Контекста пӑрахӑҫланӑ')


@dp.message(F.text)
async def echo(m: Message):
    waiting_message = await m.answer('⏳ Хурав кӗтӗр')
    messages = await db.get_context(m.from_user.id)
    messages = [{'role': 'user' if x.is_from_user else 'assistant',
                 'content': x.text} for x in messages]
    messages.append({'role': 'user', 'content': m.text})
    embeddings = await get_embedding(messages)
    response = collection.query(query_embeddings=embeddings,  n_results=5)
    context = '\n'.join(response['documents'][0])
    messages.pop(-1)
    messages.append({'role': 'assistant', 'content': rag_prompt.format(question=m.text, context=context)})
    await db.create_message(m.from_user.id, m.text, is_from_user=True)
    response = await get_response(messages)
    await waiting_message.delete()
    await db.create_message(m.from_user.id, response, is_from_user=False)
    await m.answer(response)

