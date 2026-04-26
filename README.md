# 📂 KOIB RAG v4.1 — Система поиска по документации КОИБ

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

RAG (Retrieval-Augmented Generation) система для автоматической обработки и интеллектуального поиска по технической документации КОИБ (Комплексы Обработки Избирательных Бюллетеней).

---

## ✨ Особенности

| Возможность | Описание |
|---|---|
| 📄 Форматы | PDF (в т.ч. отсканированные), DOCX |
| 🔍 OCR | pytesseract + EasyOCR (fallback) |
| 🤖 Эмбеддинги | `intfloat/multilingual-e5-large` |
| 🗄️ Векторный поиск | FAISS с поддержкой фильтрации по модели КОИБ |
| 💬 LLM | GigaChat (Сбербанк) |
| 🏷️ Модели КОИБ | КОИБ-2010, КОИБ-2017А, КОИБ-2017Б |

---

## 🐛 Исправленные ошибки (v4.1)

| # | Файл | Ошибка | Исправление |
|---|------|--------|-------------|
| 1 | `preprocessing.py` | `AttributeError: Rect has no attribute 'expand'` в PyMuPDF ≥ 1.23 | Заменено на `_expand_rect()` — явное вычисление координат |
| 2 | `preprocessing.py` | Поле `"file"` не совпадало с `"source"` в `index_building.py` — **данные не индексировались** | Поле унифицировано: `"source"` (с алиасом `"file"`) |
| 3 | `preprocessing.py` | Поле `"model"` не добавлялось в `text_blocks` | Теперь каждый блок содержит `"model"` |
| 4 | `preprocessing.py` | `FIGURES_DIR` — глобальная переменная использовалась до инициализации пайплайна | Передаётся явным параметром `figures_dir` |
| 5 | `index_building.py` | `block_type`, `headings` не существовали в блоках → `KeyError` | Все поля читаются через `.get()` с дефолтами |
| 6 | `index_building.py` | Все блоки группировались как `("unknown", "", page)` | Фикс алиаса `"source"/"file"`, `"model"` из блоков |

---

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

**Tesseract OCR (обязательно для сканов):**
```bash
# Ubuntu/Debian
sudo apt-get install -y tesseract-ocr tesseract-ocr-rus

# macOS
brew install tesseract tesseract-lang

# Windows — скачать с https://github.com/UB-Mannheim/tesseract/wiki
```

### 2. Добавьте документы

```
Koib/
└── data/
    └── docs/            ← положите сюда PDF/DOCX файлы КОИБ
        ├── koib2010_руководство.pdf
        ├── koib2017a_инструкция.pdf
        └── ...
```

### 3. Запуск

**Полный пайплайн (рекомендуется):**
```bash
python run_pipeline.py
```

**Пошагово:**
```bash
# Part 1: Предобработка документов (OCR, извлечение текста и рисунков)
python -m src.preprocessing

# Part 2: Построение векторного индекса FAISS
python -m src.index_building

# Part 3: Интерактивный поиск
python -m src.query_engine
```

**Один запрос:**
```bash
python run_pipeline.py --query "Как включить КОИБ-2010?"
python run_pipeline.py --query "Замена термоленты" --model koib2017b
```

---

## ⚙️ Настройка через переменные окружения

```bash
# Пути
export KOIB_DOCS_DIR=./data/docs
export KOIB_OUTPUT_DIR=./output

# Параметры OCR
export OCR_DPI=300
export OCR_MIN_TEXT_CHARS=50

# Параметры чанкирования
export CHUNK_SIZE=2000
export CHUNK_OVERLAP=320

# Эмбеддинги
export EMBEDDING_MODEL_NAME=intfloat/multilingual-e5-large

# GigaChat
export GIGACHAT_CREDENTIALS=<ваш_base64_token>
```

---

## 📁 Структура проекта

```
Koib/
├── run_pipeline.py         # Единая точка входа
├── gigachat_client.py      # Клиент GigaChat API
├── requirements.txt
├── data/
│   └── docs/               # Входные PDF/DOCX документы
├── output/                 # Генерируется автоматически
│   ├── metadata/
│   │   ├── text_blocks.json    # Извлечённые текстовые блоки
│   │   ├── figures_index.json  # Индекс рисунков
│   │   └── chunks.json         # Чанки для FAISS
│   ├── faiss_index/
│   │   ├── koib_index.faiss    # Векторный индекс
│   │   └── index_meta.json     # Метаданные индекса
│   ├── figures/                # Извлечённые изображения
│   ├── classified/             # CSV-отчёт классификации
│   └── logs/                   # Логи обработки
└── src/
    ├── preprocessing.py    # Part 1: OCR, извлечение
    ├── index_building.py   # Part 2: FAISS индекс
    ├── query_engine.py     # Part 3: RAG поиск
    └── utils.py            # Общие утилиты
```

---

## 🔌 Использование как библиотека

```python
from src.query_engine import KoibQueryEngine

engine = KoibQueryEngine()

# Простой поиск
context, docs, figures = engine.ask("Как заменить термоленту?")
for doc in docs:
    print(f"[{doc['model_display']}] {doc['source_name']}, стр. {doc['page']}")
    print(doc['text_preview'])

# Поиск с фильтром модели
context, docs, figures = engine.ask(
    "Порядок включения устройства",
    koib_model="koib2010"
)

# RAG + GigaChat
answer = engine.ask_with_gigachat(
    "Что делать при ошибке E05?",
    koib_model="koib2017a",
    gigachat_credentials="<ваш_токен>"
)
print(answer)
```

---

## 📊 Архитектура системы

```
PDF/DOCX
   │
   ▼
[preprocessing.py]
   ├── PyMuPDF: извлечение текста
   ├── OCR (pytesseract / EasyOCR): сканированные страницы
   ├── Определение модели КОИБ по имени файла / тексту
   └── text_blocks.json + figures_index.json
         │
         ▼
[index_building.py]
   ├── RecursiveCharacterTextSplitter → чанки
   ├── multilingual-e5-large: эмбеддинги
   └── FAISS индекс
         │
         ▼
[query_engine.py]
   ├── FAISS similarity search
   ├── Фильтрация по модели КОИБ
   ├── Сборка контекста + рисунки
   └── Промпт для GigaChat → ответ
```

---

## 📝 Лицензия

MIT License. Подробности в [LICENSE](LICENSE).

---

## 🔍 Оценка качества RAG-системы

Система включает модуль автоматической оценки качества генерации по 5 метрикам через **LLM-as-Judge** (GigaChat):

| Метрика | Описание | Шкала |
|---|---|---|
| **Faithfulness** | Верность ответа контексту (нет домыслов) | 0–1 |
| **Answer Relevancy** | Релевантность ответа вопросу | 0–1 |
| **Context Precision** | Доля полезных чанков среди найденных | 0–1 |
| **Context Recall** | Полнота покрытия фактов контекстом | 0–1 |
| **Token F1** | Токен-совпадение с эталонным ответом | 0–1 |

### Подготовка датасета

Создайте файл `eval_dataset.json` со списком тестовых вопросов:

```json
[
  {
    "id": "q001",
    "question": "Как включить КОИБ-2010?",
    "koib_model": "koib2010",
    "category": "Включение",
    "reference_answer": "Нажмите кнопку «Питание» на передней панели..."
  },
  {
    "id": "q002",
    "question": "Что делать при ошибке E05?",
    "koib_model": "koib2017a",
    "category": "Ошибки"
  }
]
```

**Поля:**
- `id` — уникальный идентификатор вопроса
- `question` — текст вопроса
- `koib_model` — фильтр по модели (опционально)
- `category` — категория для отчёта
- `reference_answer` — эталонный ответ (опционально, нужен для Token F1)

### Запуск оценки

```bash
# Базовый запуск
python evaluate_rag.py --credentials "YOUR_GIGACHAT_TOKEN" \
    --dataset eval_dataset.json \
    --output eval_report.json

# С указанием количества чанков
python evaluate_rag.py --credentials "YOUR_TOKEN" --top_k 3

# Оценка конкретных вопросов по ID
python evaluate_rag.py --credentials "YOUR_TOKEN" --ids q001 q003

# Выбор модели GigaChat (GigaChat / GigaChat-Pro / GigaChat-Max)
python evaluate_rag.py --credentials "YOUR_TOKEN" --gigachat_model GigaChat-Pro
```

**Где взять токен:**
1. Зайдите в [личный кабинет Sber GigaChat](https://developers.sber.ru/docs/ru/gigachat)
2. Создайте проект и получите `client_id` и `client_secret`
3. Объедините их в строку `client_id:client_secret` и закодируйте в base64
4. Передайте через `--credentials` или установите переменную окружения:
   ```bash
   export GIGACHAT_CREDENTIALS="your_base64_token"
   ```

### Просмотр отчёта

После оценки запустите визуализатор:

```bash
python eval_report_viewer.py --report eval_report.json
```

**Пример вывода:**
```
══════════════════════════════════════════════════════════════
  ОТЧЁТ КАЧЕСТВА RAG-СИСТЕМЫ  КОИБ v4.1
══════════════════════════════════════════════════════════════
  Вопросов: 10/10  |  Итоговый RAG Score: 0.782  — ★★★★☆ Хорошо

  Faithfulness      (верность контексту)  0.820  █████████████████████████░░░░░
  Answer Relevancy  (релевантность ответа) 0.790  ███████████████████████░░░░░░
  Context Precision (точность контекста)  0.750  ██████████████████████░░░░░░░
  Context Recall    (полнота контекста)   0.770  ███████████████████████░░░░░░

  ⚠ Слабейшая метрика: Context Precision = 0.750
     • Много лишних чанков — попробуйте уменьшить top_k (напр. 3 вместо 5).
     • Также можно снизить CHUNK_SIZE для более точечных фрагментов.
```

### Интерпретация результатов

| RAG Score | Оценка | Рекомендации |
|---|---|---|
| ≥ 0.85 | ★★★★★ Отлично | Система готова к продакшену |
| 0.70–0.84 | ★★★★☆ Хорошо | Требует небольшой доработки |
| 0.55–0.69 | ★★★☆☆ Удовл. | Нужна настройка промптов и параметров чанкирования |
| < 0.55 | ★★☆☆☆ Слабо | Критические проблемы: проверьте качество документов и промпты |

**Рекомендации по улучшению метрик:**

| Метрика | Проблема | Решение |
|---|---|---|
| **Faithfulness** ↓ | Ответы содержат домыслы | Ужесточить системный промпт: «Отвечай ТОЛЬКО на основе контекста» |
| **Answer Relevancy** ↓ | Ответы не по теме | Проверить шаблон промпта в `query_engine.py` |
| **Context Precision** ↓ | Много лишних чанков | Уменьшить `top_k` или `CHUNK_SIZE` |
| **Context Recall** ↓ | Контекст неполный | Увеличить `top_k` или `CHUNK_OVERLAP` |

