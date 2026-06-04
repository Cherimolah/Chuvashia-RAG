"""
Наполняет chroma_db_e5 из chuvash_news_final.json,
используя intfloat/multilingual-e5-large через OpenRouter.

Запуск из папки Chuvashia-RAG/:
    python build_chroma_e5.py

Поддерживает возобновление: повторный запуск пропускает уже добавленные документы.
"""
import asyncio
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv(Path(__file__).parent / ".env")

OPENROUTER_TOKEN = os.getenv("OPENROUTER_TOKEN", "")
if not OPENROUTER_TOKEN:
    print("ОШИБКА: OPENROUTER_TOKEN не задан в .env")
    sys.exit(1)

JSON_PATH = Path(__file__).parent / "chuvash_news_final.json"
CHROMA_PATH = Path(__file__).parent / "chroma_db_e5"
COLLECTION_NAME = "chuvash_news"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
BATCH_SIZE = 256      # текстов в одном API-запросе
CONCURRENCY = 6       # параллельных API-запросов одновременно
LOG_EVERY = 1000
MAX_RETRIES = 3

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_TOKEN,
)


def parse_date(date_str: str) -> float:
    try:
        return datetime.strptime(date_str, "%d.%m.%Y %H:%M").timestamp()
    except ValueError:
        return 0.0


async def embed_batch(texts: list[str], sem: asyncio.Semaphore) -> list[list[float]]:
    async with sem:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = await client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=texts,
                    encoding_format="float",
                )
                return [d.embedding for d in resp.data]
            except Exception as e:
                wait = 2 ** attempt
                print(f"  [попытка {attempt}/{MAX_RETRIES}] ошибка: {e} — жду {wait}с")
                if attempt == MAX_RETRIES:
                    raise
                await asyncio.sleep(wait)


def build_documents(articles: list[dict]) -> tuple[list, list, list, list]:
    ids, docs, metas, texts = [], [], [], []
    for article in articles:
        url = article.get("url", "").strip()
        title = article.get("title", "").strip()
        date_ts = parse_date(article.get("date", ""))
        raw_text = article.get("text", "").strip()
        if not raw_text or not url:
            continue
        paragraphs = [p.strip() for p in raw_text.split("\n\n") if len(p.strip()) >= 10]
        for i, para in enumerate(paragraphs):
            ids.append(f"{url}#{i}")
            docs.append(para)
            metas.append({"url": url, "title": title, "date": date_ts})
            # multilingual-e5-large требует префикс "passage: " для индексируемых текстов
            texts.append(f"passage: {para}")
    return ids, docs, metas, texts


async def main():
    print(f"Загружаю {JSON_PATH} …")
    with open(JSON_PATH, encoding="utf-8") as f:
        articles = json.load(f)
    print(f"Статей загружено: {len(articles)}")

    print("Разбиваю на параграфы …")
    all_ids, all_docs, all_metas, all_texts = build_documents(articles)
    print(f"Итого параграфов: {len(all_ids)}")

    print("Подключаюсь к ChromaDB серверу на localhost:8001 …")
    chroma_client = chromadb.HttpClient(host="localhost", port=8001)
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Получаем уже добавленные ID для возобновления
    try:
        existing_count = collection.count()
    except Exception as e:
        print(f"\nОШИБКА: HNSW-индекс повреждён: {e}")
        print(f"Удалите папку вручную и запустите скрипт заново:")
        print(f"  Remove-Item -Recurse -Force \"{CHROMA_PATH}\"")
        sys.exit(1)
    existing_ids: set[str] = set()
    if existing_count > 0:
        print(f"В коллекции уже {existing_count} документов, получаю существующие ID …")
        offset, page = 0, 5000
        while True:
            result = collection.get(limit=page, offset=offset, include=[])
            batch_ids = result.get("ids", [])
            if not batch_ids:
                break
            existing_ids.update(batch_ids)
            offset += len(batch_ids)
            if len(batch_ids) < page:
                break
        print(f"Пропускаю {len(existing_ids)} уже добавленных документов.")

    # Фильтруем уже добавленные
    new_ids, new_docs, new_metas, new_texts = [], [], [], []
    for doc_id, doc, meta, text in zip(all_ids, all_docs, all_metas, all_texts):
        if doc_id not in existing_ids:
            new_ids.append(doc_id)
            new_docs.append(doc)
            new_metas.append(meta)
            new_texts.append(text)

    if not new_ids:
        print("Все документы уже добавлены. Ничего не делаю.")
        return

    total = len(new_ids)
    print(f"Добавляю {total} документов (батч={BATCH_SIZE}, параллельно={CONCURRENCY}) …")

    # Нарезаем батчи
    batches = []
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batches.append((start, end))

    sem = asyncio.Semaphore(CONCURRENCY)
    added = 0
    t_start = time.perf_counter()

    # Обрабатываем батчи группами размером CONCURRENCY*2, чтобы не копить
    # все результаты в памяти сразу
    group_size = CONCURRENCY * 4
    for g_start in range(0, len(batches), group_size):
        group = batches[g_start: g_start + group_size]

        tasks = [
            embed_batch(new_texts[s:e], sem)
            for s, e in group
        ]
        results = await asyncio.gather(*tasks)

        for (s, e), embeddings in zip(group, results):
            collection.add(
                ids=new_ids[s:e],
                documents=new_docs[s:e],
                embeddings=embeddings,
                metadatas=new_metas[s:e],
            )
            added += e - s

        elapsed = time.perf_counter() - t_start
        rate = added / elapsed if elapsed > 0 else 0
        eta = (total - added) / rate if rate > 0 else 0
        print(f"  [{added}/{total}] {added/total*100:.1f}%  {rate:.0f} doc/s  ETA {eta/60:.1f} мин")

    final_count = collection.count()
    print(f"\nГотово! Итого в коллекции: {final_count} документов.")


if __name__ == "__main__":
    asyncio.run(main())
