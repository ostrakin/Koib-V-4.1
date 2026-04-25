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
