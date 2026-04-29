from loader import client

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
            return chunks
    for i in range(0, len(messages), chunk_size - overlap):
        window = messages[i:i+chunk_size]
        if len(window) < 2:  # пропускаем слишком маленькие хвосты
            continue
        text = "\n".join(f"{m['role']}: {m['content']}" for m in window)
        chunks.append({
            "text": text,
            "start_turn": i,
            "end_turn": i + len(window) - 1
        })
    return chunks


async def get_embedding(messages: list[dict[str, str]]) -> list[float]:
    chunks = chunk_dialogue(messages, chunk_size=3, overlap=1)
    texts_only = [chunk["text"] for chunk in chunks]
    embedding = client.embeddings.create(
        model="qwen/qwen3-embedding-8b",
        input=texts_only,
        encoding_format="float"
    )
    return embedding.data[0].embedding


async def get_response(messages: list[dict[str, str]]) -> str:
    messages.insert(0, {
        'role': 'system',
        'content': system_prompt
    })
    completion = client.chat.completions.create(
        model="deepseek/deepseek-v3.2",
        messages=messages,

    )
    return completion.choices[0].message.content
