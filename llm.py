from loader import client
from logger import get_logger, preview, timed

log = get_logger(__name__)

EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"
COMPLETION_MODEL = "deepseek/deepseek-v3.2"

system_prompt = """
Ты — агент по культуре Чувашской Республики, носитель и хранитель чувашских традиций, языка и истории.
Ты отвечаешь на чувашском языке, уважительно и точно, опираясь только на предоставленные фрагменты.
Ниже приведены тексты, похожие на запрос пользователя. Они могут содержать русские или чувашские фрагменты.
Твоя задача — на их основе составить связный, информативный ответ на литературном чувашском языке.
Объедини ключевые идеи из всех фрагментов
"""

rag_prompt = """
Запрос пользователя: {question}

Похожие тексты:
---
{context}
---

Ответ на чувашском языке (начни с главной мысли, затем раскрой детали):"""


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
    texts_only = [chunk["text"] for chunk in chunks]
    log.debug(f'Получено {len(texts_only)} чанков для эмбеддинга:')
    for i, t in enumerate(texts_only):
        log.debug(f'  чанк[{i}] ({len(t)} симв.): {preview(t, 150)}')

    if len(texts_only) > 1:
        log.warning(
            f'⚠️  Эмбеддятся {len(texts_only)} чанков, '
            f'но в Chroma уйдёт только эмбеддинг ПЕРВОГО '
            f'(см. embedding.data[0].embedding в get_embedding). '
            f'Возможно, стоит усреднить векторы или использовать последний чанк '
            f'(он соответствует последнему запросу пользователя).'
        )

    with timed(log, f'API запрос эмбеддингов ({EMBEDDING_MODEL})'):
        embedding = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts_only,
            encoding_format="float"
        )

    vec = embedding.data[0].embedding
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
