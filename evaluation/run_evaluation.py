"""
evaluation/run_evaluation.py

Полный пайплайн оценки качества RAG-системы.

Запуск:
    python -m evaluation.run_evaluation --test-size 50 --output results.json

Или напрямую:
    python evaluation/run_evaluation.py --help
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# -------------------------------------------------------
# Путь к корню проекта (для импортов llm.py, loader.py и т.д.)
# -------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

from logger import get_logger

log = get_logger(__name__)

# -------------------------------------------------------
# Директория для логов оценки
# -------------------------------------------------------
EVAL_LOGS_DIR = PROJECT_ROOT / "evaluation_logs"
EVAL_LOGS_DIR.mkdir(exist_ok=True)

TEST_DATA_PATH = PROJECT_ROOT / "test_data" / "questions_chuvash.json"


# =========================================================
# Standalone клиенты (не импортируем loader.py, чтобы
# не инициализировать Telegram-бота)
# =========================================================

def _init_chroma_collection():
    """Инициализирует ChromaDB коллекцию независимо от loader.py."""
    import chromadb
    path = str(PROJECT_ROOT / "chroma_db")
    log.info(f"Подключение к ChromaDB: path={path}")
    client = chromadb.PersistentClient(path=path)
    collections = client.list_collections()
    if not collections:
        raise RuntimeError(
            "ChromaDB пуст. Сначала загрузи данные в chroma_db/.\n"
            "Документы о чувашской культуре должны быть проиндексированы."
        )
    coll = collections[0]
    log.info(f"ChromaDB: коллекция '{coll.name}', документов: {coll.count()}")
    return coll


def _init_openai_client():
    """Инициализирует AsyncOpenAI клиент для OpenRouter."""
    from openai import AsyncOpenAI
    token = os.getenv("OPENROUTER_TOKEN", "")
    if not token:
        raise ValueError("OPENROUTER_TOKEN не задан в .env")
    return AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=token,
    )


# =========================================================
# Async RAG pipeline для сбора данных
# =========================================================

async def run_rag_for_question(
    question: str,
    collection,
    openai_client,
    n_results: int = 5,
) -> dict:
    """
    Выполняет полный RAG-цикл для одного вопроса:
    эмбеддинг → ChromaDB retrieval → RAG-промпт → LLM-ответ.

    Returns:
        {"question": str, "contexts": list[str], "answer": str}
    """
    from llm import (
        EMBEDDING_MODEL,
        COMPLETION_MODEL,
        system_prompt,
        rag_prompt,
        chunk_dialogue,
    )

    # 1. Эмбеддинг вопроса (один тёрн)
    messages = [{"role": "user", "content": question}]
    chunks = chunk_dialogue(messages, chunk_size=3, overlap=1)
    texts = [c["text"] for c in chunks]

    embedding_resp = await openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
        encoding_format="float",
    )
    query_vec = embedding_resp.data[0].embedding

    # 2. Retrieval из ChromaDB
    chroma_resp = collection.query(
        query_embeddings=[query_vec],
        n_results=n_results,
    )
    docs = chroma_resp.get("documents", [[]])[0]
    log.debug(f"  Retrieval: {len(docs)} документов для '{question[:60]}…'")

    # 3. Сборка RAG-промпта
    context = "\n".join(docs)
    prompt = rag_prompt.format(question=question, context=context)
    llm_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": prompt},
    ]

    # 4. Ответ LLM
    completion = await openai_client.chat.completions.create(
        model=COMPLETION_MODEL,
        messages=llm_messages,
    )
    answer = completion.choices[0].message.content

    return {
        "question": question,
        "contexts": docs,
        "answer": answer,
    }


async def collect_rag_responses(
    questions: list[dict],
    collection,
    openai_client,
    concurrency: int = 3,
) -> list[dict]:
    """
    Асинхронно собирает RAG-ответы для всех вопросов.
    Использует семафор для ограничения параллельных запросов.

    Returns:
        Список словарей: question, contexts, answer, ground_truth.
    """
    semaphore = asyncio.Semaphore(concurrency)
    results = []
    total = len(questions)

    async def _process_one(i: int, q_item: dict) -> dict | None:
        async with semaphore:
            q_text = q_item.get("question") or q_item.get("question_ru", "")
            gt = q_item.get("ground_truth") or q_item.get("ground_truth_ru", "")
            log.info(f"[{i+1}/{total}] Обрабатываю: {q_text[:70]}…")
            try:
                rag_result = await run_rag_for_question(
                    q_text, collection, openai_client
                )
                return {
                    **rag_result,
                    "ground_truth": gt,
                    "category": q_item.get("category", ""),
                    "difficulty": q_item.get("difficulty", ""),
                    "id": q_item.get("id", i),
                }
            except Exception as e:
                log.error(f"Ошибка для вопроса id={q_item.get('id', i)}: {e}")
                return None

    tasks = [_process_one(i, q) for i, q in enumerate(questions)]
    raw = await asyncio.gather(*tasks)
    results = [r for r in raw if r is not None]
    log.info(f"collect_rag_responses: {len(results)}/{total} успешно")
    return results


# =========================================================
# Генерация HTML отчёта
# =========================================================

def _render_html_report(report: dict, output_path: Path) -> None:
    """Генерирует HTML-отчёт с таблицами и визуализациями."""
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
        has_plotly = True
    except ImportError:
        has_plotly = False
        log.warning("plotly не установлен, HTML будет без графиков")

    summary = report.get("summary", {})
    noise = report.get("noise_sensitivity", {})

    # Radar chart для метрик (plotly)
    chart_html = ""
    if has_plotly and summary:
        cats = list(summary.keys())
        vals = [summary[m]["mean"] for m in cats]
        fig = go.Figure(data=go.Scatterpolar(
            r=vals + [vals[0]],
            theta=cats + [cats[0]],
            fill="toself",
            line_color="#4F6EF7",
            fillcolor="rgba(79, 110, 247, 0.2)",
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Сводка метрик (radar chart)",
            height=400,
        )
        chart_html = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")

    # Noise degradation bar chart
    noise_chart_html = ""
    if has_plotly and noise:
        noise_levels = sorted([
            k for k in noise if k.startswith("noise_") and "pct" in k
        ])
        for metric_name in ["faithfulness", "answer_relevancy"]:
            vals_n = []
            labels_n = []
            for lk in noise_levels:
                entry = noise.get(lk, {})
                m = entry.get("metrics", {}).get(metric_name, {})
                if "mean" in m:
                    vals_n.append(m["mean"])
                    labels_n.append(lk.replace("noise_", "").replace("pct", "%"))
            if vals_n:
                fig2 = go.Figure(go.Bar(
                    x=labels_n, y=vals_n,
                    marker_color="#F76B4F",
                    name=metric_name,
                ))
                fig2.update_layout(
                    title=f"Шумоустойчивость: {metric_name}",
                    yaxis=dict(range=[0, 1]),
                    height=300,
                )
                noise_chart_html += pio.to_html(fig2, full_html=False, include_plotlyjs="cdn")

    # Таблица с weak examples
    weak_html = ""
    weak = report.get("weak_examples", [])
    if weak:
        rows = "".join(
            f"<tr><td>{w.get('id','')}</td>"
            f"<td>{w.get('question','')[:80]}</td>"
            f"<td>{w.get('faithfulness','N/A')}</td>"
            f"<td>{w.get('context_precision','N/A')}</td>"
            f"<td>{w.get('context_recall','N/A')}</td>"
            f"<td>{w.get('answer_relevancy','N/A')}</td></tr>"
            for w in weak[:20]
        )
        weak_html = f"""
        <h2>⚠️ Слабые примеры (метрика < 0.5)</h2>
        <table border="1" cellpadding="6" style="border-collapse:collapse;width:100%">
            <tr><th>ID</th><th>Вопрос</th><th>Faithfulness</th>
                <th>Ctx Precision</th><th>Ctx Recall</th><th>Ans Relevancy</th></tr>
            {rows}
        </table>"""

    # Рекомендации
    recs_html = ""
    recs = report.get("recommendations", [])
    if recs:
        items = "".join(f"<li>{r}</li>" for r in recs)
        recs_html = f"<h2>📋 Рекомендации</h2><ul>{items}</ul>"

    # Сводная таблица
    summary_rows = "".join(
        f"<tr><td><b>{m}</b></td>"
        f"<td>{v['mean']:.4f}</td>"
        f"<td>{v['std']:.4f}</td>"
        f"<td>{v['min']:.4f}</td>"
        f"<td>{v['max']:.4f}</td>"
        f"<td>{v['count']}</td></tr>"
        for m, v in summary.items()
    )

    ts = report.get("timestamp", "")
    n = report.get("total_questions_evaluated", 0)

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>RAG Evaluation Report — Чувашский бот</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 30px; background: #f9f9f9; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #4F6EF7; }}
        table {{ background: white; }}
        th {{ background: #4F6EF7; color: white; padding: 8px; }}
        td {{ padding: 6px; }}
        .meta {{ background: white; padding: 15px; border-radius: 8px; margin: 10px 0; }}
    </style>
</head>
<body>
<h1>📊 RAG Evaluation Report — Чувашский бот</h1>
<div class="meta">
    <b>Дата:</b> {ts} &nbsp;|&nbsp;
    <b>Вопросов:</b> {n} &nbsp;|&nbsp;
    <b>Модель LLM:</b> {report.get("llm_model","?")} &nbsp;|&nbsp;
    <b>Embedding:</b> {report.get("embedding_model","?")}
</div>

<h2>📈 Сводная таблица метрик</h2>
<table border="1" cellpadding="6" style="border-collapse:collapse">
    <tr><th>Метрика</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th><th>Count</th></tr>
    {summary_rows}
</table>

{chart_html}

<h2>🔊 Шумоустойчивость (Noise Sensitivity)</h2>
{noise_chart_html if noise_chart_html else "<p>Нет данных (тест не запускался)</p>"}

{weak_html}
{recs_html}

</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    log.info(f"HTML отчёт сохранён: {output_path}")


# =========================================================
# Генерация рекомендаций
# =========================================================

def generate_recommendations(summary: dict, noise_report: dict) -> list[str]:
    """Автоматически генерирует рекомендации по результатам оценки."""
    recs = []

    faith = summary.get("faithfulness", {}).get("mean", 1.0)
    ctx_prec = summary.get("context_precision", {}).get("mean", 1.0)
    ctx_rec = summary.get("context_recall", {}).get("mean", 1.0)
    ans_rel = summary.get("answer_relevancy", {}).get("mean", 1.0)

    if faith < 0.7:
        recs.append(
            "🔴 Faithfulness низкий ({:.2f}): LLM добавляет информацию "
            "не из контекста. Ужесточи system_prompt — укажи явно "
            '"отвечай ТОЛЬКО на основе предоставленного контекста".'.format(faith)
        )
    if ctx_prec < 0.7:
        recs.append(
            "🔴 Context Precision низкий ({:.2f}): ChromaDB возвращает "
            "нерелевантные документы. Попробуй уменьшить n_results (сейчас 5) "
            "или применить re-ranking (Cross-Encoder).".format(ctx_prec)
        )
    if ctx_rec < 0.6:
        recs.append(
            "🟡 Context Recall низкий ({:.2f}): система пропускает "
            "нужные документы. Проверь чанкинг данных в ChromaDB — "
            "возможно chunk_size слишком велик и информация размыта.".format(ctx_rec)
        )
    if ans_rel < 0.7:
        recs.append(
            "🟡 Answer Relevancy низкий ({:.2f}): ответы отклоняются "
            "от темы. Добавь в system_prompt инструкцию отвечать кратко "
            "и по существу вопроса.".format(ans_rel)
        )

    # Специфичные рекомендации для чувашского языка
    recs.append(
        "📌 Чувашский язык: embedding-модель qwen3-embedding-8b "
        "обучена преимущественно на китайском/английском. "
        "Рекомендуется оценить multilingual e5-large или LaBSE как альтернативу."
    )
    recs.append(
        "📌 Чанкинг диалога: текущий chunk_size=3, overlap=1 при коротком "
        "диалоге создаёт дублирующиеся чанки. "
        "Рассмотри использование только последнего сообщения для эмбеддинга."
    )
    recs.append(
        "📌 В ChromaDB используется только эмбеддинг первого чанка диалога "
        "(см. предупреждение в llm.py). Попробуй mean-pooling векторов "
        "всех чанков для более точного представления контекста."
    )
    recs.append(
        "📌 Добавь метаданные в ChromaDB (источник, дата, категория) "
        "и используй metadata filtering для повышения Context Precision."
    )
    recs.append(
        "📌 Для чувашскоязычных вопросов рассмотри Hypothetical Document "
        "Embeddings (HyDE): сначала генерируй гипотетический ответ, "
        "затем ищи по нему в ChromaDB."
    )

    # Из noise-анализа
    noise_recs = noise_report.get("analysis", {}).get("recommendations", [])
    recs.extend(noise_recs)

    return recs


# =========================================================
# Главный пайплайн
# =========================================================

async def run_evaluation_pipeline(
    test_size: int,
    output_path: Path,
    skip_noise: bool = False,
    concurrency: int = 3,
    categories: list[str] | None = None,
) -> dict:
    """
    Полный пайплайн оценки:
        1. Загрузка тестового датасета
        2. Async сбор RAG-ответов (retrieval + LLM)
        3. Вычисление 4 метрик RAGAS
        4. Тест шумоустойчивости
        5. Сохранение JSON + HTML отчётов
    """
    log.info("=" * 60)
    log.info("🚀 Запуск evaluation pipeline")
    log.info(f"   test_size={test_size}, output={output_path}")
    log.info("=" * 60)
    pipeline_start = time.perf_counter()

    # ── Шаг 1: Загрузка датасета ─────────────────────────────
    log.info("Шаг 1/5: Загрузка тестового датасета…")
    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Тестовый датасет не найден: {TEST_DATA_PATH}\n"
            f"Убедись, что файл test_data/questions_chuvash.json существует."
        )
    with open(TEST_DATA_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    all_questions = raw["questions"]

    # Фильтрация по категории
    if categories:
        all_questions = [q for q in all_questions if q.get("category") in categories]
        log.info(f"Фильтр по категориям {categories}: осталось {len(all_questions)}")

    # Ограничение размера
    questions = all_questions[:test_size]
    log.info(f"Выбрано {len(questions)} вопросов для оценки")

    # ── Шаг 2: Инициализация клиентов ────────────────────────
    log.info("Шаг 2/5: Инициализация ChromaDB и OpenAI клиентов…")
    collection = _init_chroma_collection()
    openai_client = _init_openai_client()

    # ── Шаг 3: Сбор RAG-ответов ──────────────────────────────
    log.info(f"Шаг 3/5: Async сбор RAG-ответов (concurrency={concurrency})…")
    rag_data_list = await collect_rag_responses(
        questions, collection, openai_client,
        concurrency=concurrency,
    )

    if not rag_data_list:
        raise RuntimeError("Не удалось получить ни одного RAG-ответа. Проверь подключение.")

    # Приводим к формату RAGAS
    data_dict = {
        "question":     [r["question"]    for r in rag_data_list],
        "answer":       [r["answer"]      for r in rag_data_list],
        "contexts":     [r["contexts"]    for r in rag_data_list],
        "ground_truth": [r["ground_truth"] for r in rag_data_list],
    }

    # Сохраняем raw данные для отладки
    raw_path = EVAL_LOGS_DIR / f"raw_responses_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(rag_data_list, f, ensure_ascii=False, indent=2)
    log.info(f"Raw ответы сохранены: {raw_path}")

    # ── Шаг 4: RAGAS метрики ─────────────────────────────────
    log.info("Шаг 4/5: Вычисление RAGAS метрик…")
    from evaluation.ragas_evaluation import RAGEvaluator, NoiseTestSuite

    evaluator = RAGEvaluator()
    results_df = evaluator.compute_metrics(data_dict)
    summary = evaluator.summarize(results_df)
    weak_df = evaluator.find_weak_examples(results_df, threshold=0.5)

    # ── Шаг 4b: Тест шумоустойчивости ────────────────────────
    noise_report: dict = {}
    if not skip_noise:
        log.info("Шаг 4b: Тест шумоустойчивости…")
        noise_suite = NoiseTestSuite(evaluator, noise_levels=[0.0, 0.1, 0.3, 0.5])
        noise_report = noise_suite.run_noise_test(data_dict)

    # ── Шаг 5: Генерация отчётов ──────────────────────────────
    log.info("Шаг 5/5: Генерация отчётов…")
    from llm import EMBEDDING_MODEL, COMPLETION_MODEL

    recommendations = generate_recommendations(summary, noise_report)

    report = {
        "timestamp":                 datetime.now().isoformat(),
        "total_questions_evaluated": len(rag_data_list),
        "llm_model":                 COMPLETION_MODEL,
        "embedding_model":           EMBEDDING_MODEL,
        "chunking_config": {
            "chunk_size": 3,
            "overlap": 1,
        },
        "summary": summary,
        "per_question": results_df.to_dict(orient="records"),
        "weak_examples": weak_df.to_dict(orient="records"),
        "noise_sensitivity": noise_report,
        "recommendations": recommendations,
    }

    # JSON отчёт
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log.info(f"✅ JSON отчёт: {output_path}")

    # HTML отчёт
    html_path = output_path.with_suffix(".html")
    _render_html_report(report, html_path)
    log.info(f"✅ HTML отчёт: {html_path}")

    # Копия в evaluation_logs/
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_copy = EVAL_LOGS_DIR / f"report_{ts_str}.json"
    with open(log_copy, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    elapsed = time.perf_counter() - pipeline_start
    log.info(f"🏁 Pipeline завершён за {elapsed:.1f} сек")

    # Печатаем итоги в консоль
    print("\n" + "=" * 50)
    print("📊 ИТОГИ ОЦЕНКИ")
    print("=" * 50)
    for metric, vals in summary.items():
        bar = "█" * int(vals["mean"] * 20)
        print(f"  {metric:<22} {vals['mean']:.4f}  [{bar:<20}]")
    print(f"\n  Вопросов оценено: {len(rag_data_list)}")
    print(f"  Слабых примеров: {len(weak_df)}")
    print(f"  Отчёт (JSON): {output_path}")
    print(f"  Отчёт (HTML): {html_path}")
    print("=" * 50)

    return report


# =========================================================
# CLI
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Оценка качества RAG-бота на чувашском языке (RAGAS)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  # Базовый запуск (50 вопросов)
  python -m evaluation.run_evaluation --test-size 50 --output results.json

  # Только определённые категории
  python -m evaluation.run_evaluation --test-size 20 --categories culture mythology

  # Без теста шума (быстрее)
  python -m evaluation.run_evaluation --test-size 50 --skip-noise

  # Больше параллельных запросов
  python -m evaluation.run_evaluation --test-size 50 --concurrency 5
        """,
    )
    parser.add_argument(
        "--test-size", type=int, default=50,
        help="Количество вопросов для оценки (по умолчанию: 50)",
    )
    parser.add_argument(
        "--output", type=str, default="results.json",
        help="Путь для JSON-отчёта (по умолчанию: results.json)",
    )
    parser.add_argument(
        "--skip-noise", action="store_true",
        help="Пропустить тест шумоустойчивости (быстрее)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=3,
        help="Параллельные API-запросы (по умолчанию: 3)",
    )
    parser.add_argument(
        "--categories", nargs="*",
        choices=["culture", "traditions", "language", "history",
                 "mythology", "music", "geography", "hard_synthesis",
                 "trick_questions", "noise_sensitive"],
        help="Фильтр по категориям вопросов",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Уровень логирования",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Настройка уровня логирования
    logging.getLogger("chuvashia").setLevel(
        getattr(logging, args.log_level, logging.INFO)
    )

    output_path = Path(args.output).resolve()

    try:
        report = asyncio.run(
            run_evaluation_pipeline(
                test_size=args.test_size,
                output_path=output_path,
                skip_noise=args.skip_noise,
                concurrency=args.concurrency,
                categories=args.categories,
            )
        )
        sys.exit(0)
    except KeyboardInterrupt:
        log.info("Прервано пользователем")
        sys.exit(1)
    except Exception as e:
        log.exception(f"Критическая ошибка pipeline: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
