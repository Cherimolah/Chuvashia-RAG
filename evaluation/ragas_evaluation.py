"""
evaluation/ragas_evaluation.py

Модуль оценки качества RAG-системы на основе RAGAS.
Содержит классы RAGEvaluator и NoiseTestSuite.

Использование:
    from evaluation.ragas_evaluation import RAGEvaluator, NoiseTestSuite

    evaluator = RAGEvaluator()
    results = await evaluator.evaluate_dataset(data)
"""

import asyncio
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

# Путь к корню проекта (папка выше evaluation/)
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from logger import get_logger

# --- RAGAS imports ---
try:
    from ragas import evaluate
    from ragas.metrics import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
except ImportError as e:
    raise ImportError(
        f"RAGAS не установлен: {e}\n"
        "Установи: pip install -r requirements_ragas.txt"
    ) from e

# --- LangChain imports ---
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except ImportError as e:
    raise ImportError(
        f"langchain-openai не установлен: {e}\n"
        "Установи: pip install langchain-openai"
    ) from e

load_dotenv()

log = get_logger(__name__)

# =========================================================
# Константы
# =========================================================

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_TOKEN = os.getenv("OPENROUTER_TOKEN", "")

# Модели (должны совпадать с llm.py)
EVAL_LLM_MODEL = "deepseek/deepseek-v3.2"
EVAL_EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"

# Шум: фрагменты для теста шума (нерелевантные тексты)
NOISE_DOCUMENTS = [
    "Московский метрополитен открылся в 1935 году и является одним из крупнейших в мире.",
    "В Японии живут около 125 миллионов человек, столица — Токио.",
    "Квантовая физика изучает поведение частиц на субатомном уровне.",
    "Роман «Война и мир» Льва Толстого написан в 1865–1869 годах.",
    "Пирамиды Гизы построены около 2560 года до нашей эры.",
    "Атмосферное давление на уровне моря составляет 101 325 Па.",
    "Браузер Chrome разработан компанией Google и выпущен в 2008 году.",
    "Температура кипения воды при нормальном давлении равна 100°C.",
    "Большой Барьерный риф расположен у берегов Австралии.",
    "Первая программа для компьютера была написана Адой Лавлейс.",
    "Протокол HTTP был разработан Тимом Бернерс-Ли в 1989 году.",
    "Нейронные сети вдохновлены биологическими нейронами мозга.",
    "Парфенон — древнегреческий храм на Акрополе Афин.",
    "Самая высокая гора в мире — Эверест, 8848 метров.",
    "Скорость света в вакууме — около 299 792 км/с.",
]


# =========================================================
# Вспомогательные функции
# =========================================================

def _make_langchain_llm() -> ChatOpenAI:
    """
    Создаёт LangChain-совместимый LLM через OpenRouter.
    RAGAS использует его для LLM-assisted метрик.
    """
    if not OPENROUTER_TOKEN:
        raise ValueError(
            "OPENROUTER_TOKEN не задан в .env. "
            "RAGAS не сможет вызвать LLM для оценки."
        )
    log.debug(f"Создаю LangChain ChatOpenAI → {EVAL_LLM_MODEL}")
    return ChatOpenAI(
        model=EVAL_LLM_MODEL,
        openai_api_key=OPENROUTER_TOKEN,
        openai_api_base=OPENROUTER_BASE_URL,
        temperature=0.0,
        max_tokens=1024,
    )


def _make_langchain_embeddings() -> OpenAIEmbeddings:
    """
    Создаёт LangChain-совместимые эмбеддинги через OpenRouter.
    """
    log.debug(f"Создаю LangChain Embeddings → {EVAL_EMBEDDING_MODEL}")
    return OpenAIEmbeddings(
        model=EVAL_EMBEDDING_MODEL,
        openai_api_key=OPENROUTER_TOKEN,
        openai_api_base=OPENROUTER_BASE_URL,
    )


def inject_noise(
    contexts: list[str],
    noise_ratio: float,
    noise_docs: list[str] | None = None,
    seed: int | None = None,
) -> list[str]:
    """
    Добавляет нерелевантные фрагменты в список контекстов.

    Args:
        contexts:    Исходные релевантные контексты.
        noise_ratio: Доля шумовых документов (0.1 = 10 %).
        noise_docs:  Список нерелевантных текстов (по умолчанию NOISE_DOCUMENTS).
        seed:        Фиксировать random seed для воспроизводимости.

    Returns:
        Новый список контекстов с добавленным шумом.

    Пример:
        clean = ["Акатуй — летний праздник чувашей"]
        noisy = inject_noise(clean, noise_ratio=0.5)
        # → ["Акатуй — летний праздник чувашей", "<нерелевантный текст>"]
    """
    if seed is not None:
        random.seed(seed)

    pool = noise_docs if noise_docs is not None else NOISE_DOCUMENTS

    if not pool:
        log.warning("inject_noise: пул шумовых документов пуст, возвращаю оригинал")
        return contexts.copy()

    n_total = len(contexts)
    # Сколько шумовых добавить: пропорционально noise_ratio к общему числу
    n_noise = max(1, round(n_total * noise_ratio / (1 - noise_ratio + 1e-9)))
    n_noise = min(n_noise, len(pool))  # не больше пула

    selected = random.sample(pool, n_noise)
    result = contexts.copy()
    for doc in selected:
        pos = random.randint(0, len(result))
        result.insert(pos, doc)

    log.debug(
        f"inject_noise: было={n_total} контекстов, "
        f"добавлено={n_noise} шумовых (ratio={noise_ratio:.0%}), "
        f"итого={len(result)}"
    )
    return result


# =========================================================
# RAGEvaluator
# =========================================================

class RAGEvaluator:
    """
    Класс для оценки RAG-системы с помощью RAGAS.

    Поддерживаемые метрики:
        1. Faithfulness        — верность ответа контексту
        2. ContextPrecision    — точность извлечённых документов
        3. ContextRecall       — полнота извлечённого контекста
        4. AnswerRelevancy     — релевантность ответа вопросу
        (5. NoiseSensitivity   — через NoiseTestSuite)

    Пример:
        evaluator = RAGEvaluator()
        result_df = evaluator.compute_metrics(data_dict)
    """

    def __init__(
        self,
        llm: ChatOpenAI | None = None,
        embeddings: OpenAIEmbeddings | None = None,
    ):
        log.info("Инициализация RAGEvaluator…")
        self._llm = llm or _make_langchain_llm()
        self._embeddings = embeddings or _make_langchain_embeddings()

        # Обёртки RAGAS
        self._ragas_llm = LangchainLLMWrapper(self._llm)
        self._ragas_embeddings = LangchainEmbeddingsWrapper(self._embeddings)

        # Метрики с кастомными промптами (русский язык для оценщика)
        self.faithfulness = Faithfulness(llm=self._ragas_llm)
        self.context_precision = ContextPrecision(llm=self._ragas_llm)
        self.context_recall = ContextRecall(llm=self._ragas_llm)
        self.answer_relevancy = AnswerRelevancy(
            llm=self._ragas_llm,
            embeddings=self._ragas_embeddings,
        )

        self.metrics = [
            self.faithfulness,
            self.context_precision,
            self.context_recall,
            self.answer_relevancy,
        ]
        log.info("RAGEvaluator готов, метрики: faithfulness, context_precision, "
                 "context_recall, answer_relevancy")

    def _validate_data(self, data: dict[str, list]) -> None:
        """Проверяет наличие обязательных полей в датасете."""
        required = {"question", "answer", "contexts", "ground_truth"}
        missing = required - set(data.keys())
        if missing:
            raise ValueError(
                f"В data отсутствуют обязательные поля: {missing}\n"
                f"Ожидаются: question, answer, contexts, ground_truth"
            )
        lengths = {k: len(v) for k, v in data.items() if k in required}
        if len(set(lengths.values())) > 1:
            raise ValueError(
                f"Длины списков не совпадают: {lengths}"
            )
        log.debug(f"Валидация датасета: {list(lengths.items())} — OK")

    def _to_hf_dataset(self, data: dict[str, list]) -> Dataset:
        """Конвертирует словарь в HuggingFace Dataset для RAGAS."""
        self._validate_data(data)
        # RAGAS ожидает contexts как List[List[str]]
        formatted_data = {
            "question": data["question"],
            "answer": data["answer"],
            "contexts": [
                ctx if isinstance(ctx, list) else [ctx]
                for ctx in data["contexts"]
            ],
            "ground_truth": data["ground_truth"],
        }
        ds = Dataset.from_dict(formatted_data)
        log.debug(f"HuggingFace Dataset создан: {len(ds)} строк")
        return ds

    def compute_metrics(
        self,
        data: dict[str, list],
        batch_size: int = 5,
    ) -> pd.DataFrame:
        """
        Вычисляет все 4 метрики RAGAS для переданного датасета.

        Args:
            data: Словарь с ключами question, answer, contexts, ground_truth.
                  Каждый ключ — список значений одинаковой длины.
            batch_size: Размер батча для API-запросов.

        Returns:
            DataFrame с колонками: question, faithfulness, context_precision,
            context_recall, answer_relevancy.
        """
        log.info(f"compute_metrics: запуск на {len(data['question'])} примерах…")
        t0 = time.perf_counter()

        hf_ds = self._to_hf_dataset(data)

        try:
            result = evaluate(
                dataset=hf_ds,
                metrics=self.metrics,
                llm=self._ragas_llm,
                embeddings=self._ragas_embeddings,
                raise_exceptions=False,
            )
        except Exception as e:
            log.exception(f"Ошибка RAGAS evaluate(): {e}")
            raise

        elapsed = time.perf_counter() - t0
        log.info(f"compute_metrics: завершено за {elapsed:.1f} сек")

        df = result.to_pandas()

        # Добавляем исходные вопросы если нет
        if "question" not in df.columns:
            df.insert(0, "question", data["question"])

        log.info(
            f"Результаты (mean): "
            + ", ".join(
                f"{col}={df[col].mean():.3f}"
                for col in ["faithfulness", "context_precision",
                            "context_recall", "answer_relevancy"]
                if col in df.columns
            )
        )
        return df

    def summarize(self, df: pd.DataFrame) -> dict[str, dict[str, float]]:
        """
        Считает сводную статистику (mean, std, min, max) по каждой метрике.

        Returns:
            {"faithfulness": {"mean": 0.8, "std": 0.1, "min": 0.5, "max": 1.0}, ...}
        """
        metric_cols = [c for c in df.columns if c in {
            "faithfulness", "context_precision", "context_recall", "answer_relevancy"
        }]
        summary = {}
        for col in metric_cols:
            series = df[col].dropna()
            summary[col] = {
                "mean": round(float(series.mean()), 4),
                "std":  round(float(series.std()), 4),
                "min":  round(float(series.min()), 4),
                "max":  round(float(series.max()), 4),
                "count": int(series.count()),
            }
        log.info("Сводка метрик:\n" + json.dumps(summary, ensure_ascii=False, indent=2))
        return summary

    def find_weak_examples(
        self,
        df: pd.DataFrame,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """
        Возвращает строки, где хотя бы одна метрика ниже порога.
        Помогает найти слабые места системы.
        """
        metric_cols = [c for c in df.columns if c in {
            "faithfulness", "context_precision", "context_recall", "answer_relevancy"
        }]
        mask = (df[metric_cols] < threshold).any(axis=1)
        weak = df[mask].copy()
        log.info(
            f"find_weak_examples (threshold={threshold}): "
            f"{len(weak)}/{len(df)} примеров ниже порога"
        )
        return weak


# =========================================================
# NoiseTestSuite
# =========================================================

class NoiseTestSuite:
    """
    Тесты на шумоустойчивость RAG-системы.

    Измеряет, как изменяются метрики faithfulness и answer_relevancy
    при добавлении 10 %, 30 %, 50 % нерелевантных фрагментов в контекст.

    Пример:
        evaluator = RAGEvaluator()
        noise_suite = NoiseTestSuite(evaluator)
        noise_report = noise_suite.run_noise_test(base_data)
    """

    DEFAULT_NOISE_LEVELS = [0.0, 0.10, 0.30, 0.50]

    def __init__(
        self,
        evaluator: RAGEvaluator,
        noise_levels: list[float] | None = None,
        noise_docs: list[str] | None = None,
        seed: int = 42,
    ):
        self.evaluator = evaluator
        self.noise_levels = noise_levels or self.DEFAULT_NOISE_LEVELS
        self.noise_docs = noise_docs or NOISE_DOCUMENTS
        self.seed = seed
        log.info(
            f"NoiseTestSuite: уровни шума={self.noise_levels}, "
            f"размер пула шумов={len(self.noise_docs)}"
        )

    def run_noise_test(
        self,
        base_data: dict[str, list],
    ) -> dict[str, Any]:
        """
        Запускает тест шумоустойчивости.

        Для каждого noise_level создаёт версию датасета с добавленным шумом,
        вычисляет метрики и сравнивает с базовым уровнем.

        Args:
            base_data: Чистый датасет (question, answer, contexts, ground_truth).

        Returns:
            Словарь с результатами по каждому уровню шума.
        """
        log.info(
            f"NoiseTestSuite.run_noise_test: "
            f"{len(base_data['question'])} примеров × {len(self.noise_levels)} уровней шума"
        )
        results = {}

        for noise_level in self.noise_levels:
            log.info(f"  → Шум {noise_level:.0%}…")

            # Создаём загрязнённые контексты
            noisy_contexts = [
                inject_noise(
                    contexts=ctx if isinstance(ctx, list) else [ctx],
                    noise_ratio=noise_level,
                    noise_docs=self.noise_docs,
                    seed=self.seed,
                )
                for ctx in base_data["contexts"]
            ]

            noisy_data = {**base_data, "contexts": noisy_contexts}

            # Вычисляем только нужные метрики
            try:
                df = self.evaluator.compute_metrics(noisy_data)
                metric_cols = [c for c in df.columns if c in {
                    "faithfulness", "answer_relevancy"
                }]
                results[f"noise_{int(noise_level * 100)}pct"] = {
                    "noise_level": noise_level,
                    "n_samples": len(df),
                    "metrics": {
                        col: {
                            "mean": round(float(df[col].mean()), 4),
                            "std":  round(float(df[col].std()),  4),
                        }
                        for col in metric_cols
                    }
                }
            except Exception as e:
                log.error(f"Ошибка при noise_level={noise_level}: {e}")
                results[f"noise_{int(noise_level * 100)}pct"] = {"error": str(e)}

        analysis = self.analyze_noise_impact(results)
        results["analysis"] = analysis

        log.info("NoiseTestSuite: тест завершён")
        return results

    def analyze_noise_impact(
        self,
        noise_results: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Анализирует деградацию метрик при росте шума.

        Returns:
            Словарь с анализом: относительное падение, устойчивость.
        """
        levels = [k for k in noise_results if k.startswith("noise_") and "pct" in k]
        if not levels:
            return {"error": "Нет данных для анализа"}

        base_key = "noise_0pct"
        if base_key not in noise_results:
            base_key = levels[0]

        base = noise_results.get(base_key, {})
        base_metrics = base.get("metrics", {})

        analysis = {
            "baseline_noise": base.get("noise_level", 0.0),
            "metric_degradation": {},
            "recommendations": [],
        }

        for metric in ["faithfulness", "answer_relevancy"]:
            if metric not in base_metrics:
                continue
            base_val = base_metrics[metric]["mean"]
            degradation = {}
            for key in levels:
                if key == base_key:
                    continue
                entry = noise_results.get(key, {})
                m = entry.get("metrics", {}).get(metric, {})
                val = m.get("mean", None)
                if val is not None and base_val > 0:
                    drop_pct = (base_val - val) / base_val * 100
                    degradation[key] = {
                        "absolute": round(val, 4),
                        "drop_percent": round(drop_pct, 1),
                    }
            analysis["metric_degradation"][metric] = degradation

        # Автоматические рекомендации
        for metric, deg in analysis["metric_degradation"].items():
            for key, vals in deg.items():
                if vals["drop_percent"] > 20:
                    analysis["recommendations"].append(
                        f"⚠️  {metric} падает на {vals['drop_percent']:.0f}% "
                        f"при {key} — рекомендуется улучшить re-ranking или "
                        f"фильтрацию нерелевантных документов."
                    )

        log.info(
            "Анализ шума:\n"
            + json.dumps(analysis, ensure_ascii=False, indent=2)
        )
        return analysis
