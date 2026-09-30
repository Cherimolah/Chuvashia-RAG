# Chuvashia-RAG

**Telegram-бот о культуре Чувашии**, отвечающий на чувашском языке. Использует RAG-архитектуру: извлекает релевантные фрагменты из векторной базы знаний и формирует ответ через LLM.

---

## О проекте

Бот выступает «агентом чувашской культуры» — хранителем традиций, языка и истории Чувашской Республики. Пользователь задаёт вопрос на русском или чувашском, бот находит релевантные документы в ChromaDB и отвечает на литературном чувашском языке.

**Ключевые возможности:**
- Ответы строго на основе базы знаний
- Диалоговый контекст: история переписки хранится в PostgreSQL
- Интеллектуальная фильтрация по датам: агент-классификатор автоматически определяет, запрашивает ли пользователь события за конкретный период, и применяет фильтр по дате в ChromaDB
- Команда сброса контекста (`🗑 Сбросить контекст`)
- Встроенный пайплайн оценки качества на базе RAGAS

---

## Архитектура

```
Пользователь (Telegram)
        │
        ▼
  [handlers.py]  ← aiogram 3
        │
        ├─ История диалога ──► [database.py] (PostgreSQL + SQLAlchemy async)
        │
        ├─ Классификация ───► [llm.py] → OpenRouter (grok-4.20 + outlines)
        │   вопроса              «нужен ли фильтр по дате?»
        │
        ├─ Embed запроса ────► [llm.py] → OpenRouter (qwen3-embedding-8b)
        │                                        │
        ├─ Векторный поиск ──► [ChromaDB] ◄──────┘
        │       (top-N документов, опционально с фильтром по дате)
        │
        └─ Генерация ответа ─► [llm.py] → OpenRouter (deepseek-v4-flash)
                                        │
                                        ▼
                               Ответ на чувашском языке
```

**Поток обработки сообщения:**

1. Сообщение пользователя поступает в `handlers.py`
2. Агент-классификатор (`grok-4.20` + `outlines`) определяет, нужен ли фильтр по временному периоду
3. Из PostgreSQL загружается история диалога
4. Диалог чанкуется и преобразуется в эмбеддинг через `qwen/qwen3-embedding-8b` (mean pooling по чанкам)
5. ChromaDB возвращает топ-N похожих документов из базы знаний (с фильтром по дате, если агент определил период)
6. Вопрос + контекст передаётся в `deepseek/deepseek-v4-flash` с системным промптом
7. Ответ отправляется пользователю, диалог сохраняется в БД

---

## Структура репозитория

```
chuvashia-rag/
├── main.py                  # Точка входа: инициализация таблиц, запуск polling
├── loader.py                # Инициализация Bot, Dispatcher, OpenAI client, ChromaDB
├── handlers.py              # Обработчики Telegram-сообщений
├── llm.py                   # Эмбеддинги, RAG-промпт, агент-классификатор, вызов LLM
├── database.py              # Модели SQLAlchemy (User, Message) + async CRUD
├── middleware.py             # Middleware авторегистрации пользователей
├── config.py                # Загрузка переменных окружения
├── logger.py                # Настройка логирования (консоль + файл + ротация)
├── .env.example             # Шаблон переменных окружения
├── requirements.txt         # Зависимости Python (основной бот)
├── requirements_ragas.txt   # Зависимости для оценки качества (RAGAS)
├── merged.jsonl             # Данные для базы знаний
├── evaluation/              # Пайплайн оценки качества
│   ├── __init__.py
│   ├── ragas_evaluation.py  # Обёртки над RAGAS-метриками + тест шумоустойчивости
│   └── run_evaluation.py    # CLI для запуска оценки с генерацией отчётов
├── test_data/               # Тестовые данные
│   ├── questions_chuvash.json  # Датасет вопросов для оценки
│   └── example_report.json     # Пример отчёта
├── evaluation_logs/         # Логи и отчёты оценок (генерируются автоматически)
└── chroma_db/               # Персистентная векторная база (создаётся локально)
```

---

## Стек технологий

| Компонент          | Технология                                            |
| ------------------ | ----------------------------------------------------- |
| Telegram-фреймворк | aiogram 3.x                                          |
| LLM (генерация)    | `deepseek/deepseek-v4-flash` via OpenRouter           |
| Агент-классификатор| `x-ai/grok-4.20` via OpenRouter + outlines (structured output) |
| Эмбеддинги         | `qwen/qwen3-embedding-8b` via OpenRouter              |
| Векторная БД       | ChromaDB (персистентный клиент)                       |
| Реляционная БД     | PostgreSQL + SQLAlchemy (asyncpg)                     |
| Оценка качества    | RAGAS (faithfulness, context precision/recall, answer relevancy) |
| Язык               | Python 3.11+ (async/await)                            |

---

## Быстрый старт

### 1. Клонирование и установка зависимостей

```bash
git clone https://gitverse.ru/Vsemivladeu/Chuvashia-RAG.git
cd Chuvashia-RAG
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Настройка переменных окружения

```bash
cp .env.example .env
```

Заполните `.env`:

```env
TELEGRAM_TOKEN=<токен вашего бота из @BotFather>
OPENROUTER_TOKEN=<ключ API с openrouter.ai>

PG_USER=postgres
PG_PASSWORD=your_password
PG_HOST=localhost
PG_PORT=5432
DB_NAME=chuvashia

LOG_LEVEL=INFO
```

### 3. Подготовка базы данных

Создайте базу данных PostgreSQL:

```sql
CREATE DATABASE chuvashia;
```

Таблицы создаются автоматически при первом запуске.

### 4. Подготовка векторной базы знаний

> Для работы RAG необходима предварительно заполненная коллекция ChromaDB в директории `chroma_db/`.
> Загрузите свои документы о культуре Чувашии и создайте коллекцию (см. раздел ниже).

Пример наполнения ChromaDB:

```python
import chromadb
from openai import OpenAI

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key="<OPENROUTER_TOKEN>")
chroma = chromadb.PersistentClient(path="chroma_db")
collection = chroma.create_collection("chuvashia")

documents = ["Ваш текст 1", "Ваш текст 2", ...]
embeddings = client.embeddings.create(
    model="qwen/qwen3-embedding-8b",
    input=documents
).data

collection.add(
    documents=documents,
    embeddings=[e.embedding for e in embeddings],
    ids=[str(i) for i in range(len(documents))]
)
```

### 5. Запуск

```bash
python main.py
```

---

## Переменные окружения

| Переменная | Описание | По умолчанию |
|---|---|---|
| `TELEGRAM_TOKEN` | Токен бота (получить у [@BotFather](https://t.me/BotFather)) | — |
| `OPENROUTER_TOKEN` | API-ключ [OpenRouter](https://openrouter.ai/) | — |
| `PG_USER` | Пользователь PostgreSQL | — |
| `PG_PASSWORD` | Пароль PostgreSQL | — |
| `PG_HOST` | Хост PostgreSQL | `localhost` |
| `PG_PORT` | Порт PostgreSQL | `5432` |
| `DB_NAME` | Имя базы данных | — |
| `LOG_LEVEL` | Уровень логирования (`DEBUG`, `INFO`, `WARNING`, `ERROR`) | `INFO` |
| `TEMPERATURE` | Температура генерации LLM | `0.4` |
| `MAX_TOKENS` | Максимум токенов в ответе LLM | `10000` |
| `TOP_P` | Top-p (nucleus sampling) | `0.9` |
| `FREQUENCY_PENALTY` | Штраф за повторение токенов | `0.1` |
| `PRESENCE_PENALTY` | Штраф за присутствие токенов | `0.1` |
| `N_RESULTS` | Количество документов из ChromaDB | `5` |

---

## Оценка качества (Evaluation)

В проекте встроен пайплайн оценки качества RAG-системы на базе [RAGAS](https://docs.ragas.io/).

### Установка зависимостей для оценки

```bash
pip install -r requirements_ragas.txt
```

### Запуск оценки

```bash
# Базовый запуск (50 вопросов)
python -m evaluation.run_evaluation --test-size 50 --output results.json

# Только определённые категории
python -m evaluation.run_evaluation --test-size 20 --categories culture mythology

# Без теста шумоустойчивости (быстрее)
python -m evaluation.run_evaluation --test-size 50 --skip-noise

# Только две метрики (меньше токенов)
python -m evaluation.run_evaluation --test-size 50 --metrics faithfulness answer_relevancy
```

### Метрики

| Метрика | Описание |
|---|---|
| **Faithfulness** | Насколько ответ основан на предоставленном контексте |
| **Context Precision** | Точность извлечённых документов |
| **Context Recall** | Полнота извлечённых документов |
| **Answer Relevancy** | Релевантность ответа вопросу |

Результаты сохраняются в JSON и HTML форматах с визуализацией (radar chart, таблицы слабых примеров, рекомендации).

---

## Команды бота

| Команда / кнопка | Действие |
|---|---|
| `/start` | Приветствие на чувашском, меню |
| `🗑 Сбросить контекст` | Очистить историю диалога |
| Любое текстовое сообщение | RAG-запрос к базе знаний |

---

## Логирование

Бот пишет логи в три направления:
- **Консоль** — цветной вывод, уровень задаётся через `LOG_LEVEL`
- **`logs/app.log`** — полный лог (DEBUG), ротация по 5 МБ
- **`logs/errors.log`** — только ошибки, ротация по 5 МБ
