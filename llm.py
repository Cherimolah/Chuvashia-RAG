import datetime

import numpy as np
from loader import client
from logger import get_logger, preview, timed
from config import TEMPERATURE, MAX_TOKENS, TOP_P, FREQUENCY_PENALTY, PRESENCE_PENALTY

from pydantic import BaseModel
import outlines

log = get_logger(__name__)

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
COMPLETION_MODEL = "deepseek/deepseek-v4-flash"

system_prompt = """
Ты — агент по культуре Чувашской Республики, носитель и хранитель чувашских традиций, языка и истории.
Ты отвечаешь на чувашском языке, опираясь ТОЛЬКО на предоставленные фрагменты.
Правила ответа:
- Если вопрос требует имени — дай только имя.
- Если вопрос требует даты — дай только дату.
- Если вопрос требует факта — дай только факт.
- Ответ не длиннее 2–3 предложений. Не сочиняй рассказ. Не добавляй лишних деталей.
- Если факта нет в приведённых текстах — ответь: «Эта информация отсутствует в базе данных.»
"""

rag_prompt = """
Запрос пользователя: {question}

Похожие тексты:
---
{context}
---

Краткий ответ на чувашском языке (только факт из текста выше):"""


agent_prompt = ('Тебе необходимо решить спрашивает ли пользователях о каких-то событиях '
                'за конкретный промежуток времени или нет. Если пользователь задают общий вопрос передавай'
                'need_period=false и period=null. Если пользователь спрашивает события за какой-то период то тебе'
                ' нужно написать начало и конец периода с которого запрашивает пользователь\n\n'
                'Текст пользователя: {0}')

model = outlines.from_openai(client, "x-ai/grok-4.20")


class PeriodModel(BaseModel):
    start_turn: datetime.datetime
    end_turn: datetime.datetime


class PeriodResponse(BaseModel):
    need_period: bool
    period: PeriodModel | None


def chunk_dialogue(messages, chunk_size=3, overlap=1):
    chunks = []
    if len(messages) < chunk_size:
        for i in range(0, len(messages)):
            text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            chunks.append({
                "text": text,
                "start_turn": i,
                "end_turn": i + len(messages) - 1
            })
        log.debug(
            f'chunk_dialogue: messages={len(messages)} < chunk_size={chunk_size}, '
            f'создал {len(chunks)} коротких чанков'
        )
        return chunks

    for i in range(0, len(messages), chunk_size - overlap):
        window = messages[i:i + chunk_size]
        if len(window) < 2:  # пропускаем слишком маленькие хвосты
            continue
        text = "\n".join(f"{m['role']}: {m['content']}" for m in window)
        chunks.append({
            "text": text,
            "start_turn": i,
            "end_turn": i + len(window) - 1
        })
    log.debug(
        f'chunk_dialogue: messages={len(messages)}, '
        f'chunk_size={chunk_size}, overlap={overlap} → {len(chunks)} чанков'
    )
    return chunks


async def get_embedding(messages: list[dict[str, str]]) -> list[float]:
    log.info(f'🧬 Эмбеддинг: на входе {len(messages)} сообщений диалога')

    chunks = chunk_dialogue(messages, chunk_size=3, overlap=1)
    # multilingual-e5-large требует префикс "query: " для поисковых запросов
    texts_only = ["query: " + chunk["text"] for chunk in chunks]
    log.debug(f'Получено {len(texts_only)} чанков для эмбеддинга:')
    for i, t in enumerate(texts_only):
        log.debug(f'  чанк[{i}] ({len(t)} симв.): {preview(t, 150)}')

    with timed(log, f'API запрос эмбеддингов ({EMBEDDING_MODEL})'):
        embedding = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts_only,
            encoding_format="float"
        )

    vectors = [d.embedding for d in embedding.data]
    vec = np.mean(vectors, axis=0).tolist()
    log.debug(f'Mean pooling: усреднено {len(vectors)} векторов')
    usage = getattr(embedding, 'usage', None)
    extra = ''
    if usage is not None:
        extra = f', tokens prompt={getattr(usage, "prompt_tokens", "?")}'
    log.info(
        f'🧬 Эмбеддинг готов: модель={EMBEDDING_MODEL}, dim={len(vec)}, '
        f'превью={[round(x, 4) for x in vec[:3]]}…{extra}'
    )
    return vec


async def get_response(messages: list[dict[str, str]]) -> str:
    messages.insert(0, {
        'role': 'system',
        'content': system_prompt
    })

    total_chars = sum(len(m['content']) for m in messages)
    log.info(
        f'🤖 LLM запрос: модель={COMPLETION_MODEL}, '
        f'сообщений={len(messages)} (включая system), '
        f'общая длина={total_chars} симв.'
    )
    for i, m in enumerate(messages):
        log.debug(
            f'  [{i}] role={m["role"]} ({len(m["content"])} симв.): '
            f'{preview(m["content"], 200)}'
        )

    with timed(log, f'API запрос LLM ({COMPLETION_MODEL})'):
        completion = client.chat.completions.create(
            model=COMPLETION_MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            top_p=TOP_P,
            frequency_penalty=FREQUENCY_PENALTY,
            presence_penalty=PRESENCE_PENALTY,
        )

    answer = completion.choices[0].message.content
    usage = getattr(completion, 'usage', None)
    if usage is not None:
        log.info(
            f'🤖 LLM usage: prompt={getattr(usage, "prompt_tokens", "?")}, '
            f'completion={getattr(usage, "completion_tokens", "?")}, '
            f'total={getattr(usage, "total_tokens", "?")} токенов'
        )
    log.debug(f'LLM raw answer ({len(answer)} симв.): {preview(answer, 300)}')
    return answer


async def extract_period(message: str) -> PeriodResponse:
    response = model.generate(agent_prompt.format(message), PeriodResponse)
    return PeriodResponse.model_validate_json(response)

