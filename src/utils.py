# -*- coding: utf-8 -*-
"""
KOIB RAG - Common Utilities

Общие утилиты для системы RAG документации КОИБ.
"""

import os
import re
import hashlib
from pathlib import Path
from typing import Tuple, Dict, List, Any

# Паттерны моделей КОИБ
KNOWN_MODELS = {"koib2010", "koib2017a", "koib2017b"}

KOIB_MODEL_PATTERNS = {
    "koib2010": [
        r"КОИБ[-\s]?2010", r"КОИБ\s*2010", r"0912054",
        r"PRINT_KOIB2010", r"2010.*руководство", r"руководство.*2010",
        r"модель\s*17404049\.438900\.001",
    ],
    "koib2017a": [
        r"КОИБ[-\s]?2017\s*[АA]", r"КОИБ[-\s]?2017А",
        r"модель\s*17404049\.5013009\.008-01", r"17404049\.5013009",
        r"PRINT_KOIB2017[АA]", r"2017[АA].*руководство",
    ],
    "koib2017b": [
        r"КОИБ[-\s]?2017\s*[БB]", r"КОИБ[-\s]?2017Б",
        r"БАВУ\.201119", r"0912053", r"PRINT_KOIB2017[БB]",
        r"2017[БB].*руководство",
    ],
}

MODEL_DISPLAY_NAMES = {
    "koib2010": "КОИБ-2010",
    "koib2017a": "КОИБ-2017А",
    "koib2017b": "КОИБ-2017Б",
    "unknown": "Неизвестная модель",
}

FIGURE_CAPTION_PATTERNS = [
    re.compile(r"(Рис(?:ун(?:ок|ке))[\s.]?\s*[\d.]+[^\n]*)", re.IGNORECASE),
    re.compile(r"(Рис\.?\s*[\d.]+[^\n]*)", re.IGNORECASE),
    re.compile(r"(Фиг\.?\s*[\d.]+[^\n]*)", re.IGNORECASE),
    re.compile(r"(Рисунок\s+\d+[\.][^\n]*)", re.IGNORECASE),
]


def get_base_dir() -> Path:
    """Получить базовую директорию проекта."""
    return Path(__file__).parent.parent


def get_docs_dir() -> Path:
    """Получить директорию с документами из переменной окружения или использовать путь по умолчанию."""
    docs_dir = os.getenv("KOIB_DOCS_DIR")
    if docs_dir:
        return Path(docs_dir)
    # Путь по умолчанию для Colab
    colab_path = Path("/content/drive/MyDrive/Koib/docs")
    if colab_path.exists():
        return colab_path
    # Локальный путь по умолчанию
    return get_base_dir() / "data" / "docs"


def get_output_dir() -> Path:
    """Получить выходную директорию из переменной окружения или использовать путь по умолчанию."""
    output_dir = os.getenv("KOIB_OUTPUT_DIR")
    if output_dir:
        return Path(output_dir)
    # Путь по умолчанию для Colab
    colab_path = Path("/content/drive/MyDrive/Koib/koib_rag_GLM1")
    if colab_path.exists():
        return colab_path
    # Локальный путь по умолчанию
    return get_base_dir() / "output"


def clean_text(text: str) -> str:
    """
    Очистить текст от лишних пробелов и специальных символов.
    
    Args:
        text: Исходный текст
        
    Returns:
        Очищенный текст
    """
    if not text:
        return ""
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[^\x20-\x7E\u0400-\u04FF\u2116\n\r\t]', '', text)
    lines = [line.strip() for line in text.split('\n')]
    return '\n'.join(lines).strip()


def text_hash(text: str) -> str:
    """
    Вычислить хеш текста (первые 12 символов MD5).
    
    Args:
        text: Текст для хеширования
        
    Returns:
        Хеш текста
    """
    return hashlib.md5(text.encode('utf-8', errors='ignore')).hexdigest()[:12]


def normalize_model_key(key: str) -> str:
    """
    Нормализовать ключ модели.
    
    Args:
        key: Ключ модели
        
    Returns:
        Нормализованный ключ или 'unknown'
    """
    key = str(key).strip().lower()
    return key if key in KNOWN_MODELS else "unknown"



def detect_model_in_text(text: str) -> Tuple[str, float]:
    """
    Определить модель КОИБ по тексту документа.
    
    Args:
        text: Текст документа
        
    Returns:
        Кортеж (ключ_модели, уверенность)
    """
    if not text or len(text.strip()) < 5:
        return ("unknown", 0.0)
    
    scores = {}
    for model_key, patterns in KOIB_MODEL_PATTERNS.items():
        match_count = 0
        total_matches = 0
        for pat in patterns:
            matches = re.findall(pat, text, re.IGNORECASE)
            if matches:
                match_count += 1
                total_matches += len(matches)
        if match_count > 0:
            scores[model_key] = match_count * 10 + total_matches
    
    if not scores:
        return ("unknown", 0.0)
    
    best_model = max(scores, key=scores.get)
    best_score = scores[best_model]
    confidence = min(best_score / 30.0, 1.0)
    return (best_model, round(confidence, 3))


def detect_model_from_filename(filename: str) -> str:
    """
    Определить модель КОИБ по имени файла.
    
    Args:
        filename: Имя файла
        
    Returns:
        Ключ модели или 'unknown'
    """
    fn = filename.lower()
    for model_key, patterns in KOIB_MODEL_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, fn, re.IGNORECASE):
                return model_key
    return "unknown"


def find_figure_caption(text: str, max_distance: int = 300) -> str:
    """
    Найти подпись к рисунку в тексте.
    
    Args:
        text: Текст для поиска
        max_distance: Максимальное расстояние от позиции поиска
        
    Returns:
        Найденная подпись или пустая строка
    """
    if not text:
        return ""
    for pat in FIGURE_CAPTION_PATTERNS:
        match = pat.search(text)
        if match:
            caption = match.group(1).strip()
            if len(caption) > 3:
                return caption
    return ""


def ensure_dirs(*dirs: Path) -> None:
    """
    Создать директории, если они не существуют.
    
    Args:
        dirs: Список путей к директориям
    """
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
