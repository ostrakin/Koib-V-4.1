# -*- coding: utf-8 -*-
"""
KOIB RAG - Setup Module для Google Colab

Модуль инициализации системы RAG для VK-чат-бота по документации КОИБ.
Выполняет:
- Монтирование Google Drive
- Чтение секретов из Colab Secrets
- Загрузку или построение FAISS-индекса
- Кэширование модели эмбеддингов
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Tuple, Optional

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Пути для Google Colab
GOOGLE_DRIVE_PATH = "/content/drive/MyDrive"
KOIB_BASE_PATH = Path("/content/drive/MyDrive/Koib")
KOIB_MODELS_PATH = KOIB_BASE_PATH / "models"
KOIB_OUTPUT_PATH = KOIB_BASE_PATH / "koib_rag_GLM1"
KOIB_METADATA_PATH = KOIB_OUTPUT_PATH / "metadata"
KOIB_FAISS_PATH = KOIB_OUTPUT_PATH / "faiss_index"
PROMPT_PATH = KOIB_BASE_PATH / "prompt.txt"


def mount_google_drive() -> None:
    """Смонтировать Google Drive в Google Colab."""
    from google.colab import drive
    
    if not Path(GOOGLE_DRIVE_PATH).exists():
        logger.info("Монтирование Google Drive...")
        drive.mount(GOOGLE_DRIVE_PATH, force_remount=False)
        logger.info("✅ Google Drive смонтирован")
    else:
        logger.info("✅ Google Drive уже смонтирован")


def get_secrets() -> Tuple[str, str]:
    """
    Получить секреты из Colab Secrets.
    
    Returns:
        Кортеж (GIGACHAT_CREDENTIALS, VK_GROUP_TOKEN)
    """
    from google.colab import userdata
    
    try:
        gigachat_credentials = userdata.get('GIGACHAT_CREDENTIALS')
        logger.info("✅ GIGACHAT_CREDENTIALS получен из Colab Secrets")
    except userdata.SecretNotFoundError:
        logger.error("❌ Секрет 'GIGACHAT_CREDENTIALS' не найден в Colab Secrets")
        raise
    
    try:
        vk_token = userdata.get('VK_GROUP_TOKEN')
        logger.info("✅ VK_GROUP_TOKEN получен из Colab Secrets")
    except userdata.SecretNotFoundError:
        logger.error("❌ Секрет 'VK_GROUP_TOKEN' не найден в Colab Secrets")
        raise
    
    return gigachat_credentials, vk_token


def get_system_prompt() -> str:
    """
    Прочитать system prompt из файла на Google Drive.
    
    Returns:
        Текст системного промпта
    """
    if not PROMPT_PATH.exists():
        # Создаём дефолтный промпт если файл не существует
        default_prompt = """Вы — ассистент для технической поддержки комплексов обработки избирательных бюллетеней (КОИБ).
Отвечайте ТОЛЬКО по документации КОИБ. Если информации нет в документации — сообщите об этом.
---
Здравствуйте! Я виртуальный помощник по комплексам обработки избирательных бюллетеней (КОИБ).

Я могу отвечать на вопросы по следующим моделям:
• КОИБ-2010
• КОИБ-2017А  
• КОИБ-2017Б

Выберите модель из меню ниже или задайте общий вопрос."""
        
        PROMPT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PROMPT_PATH, 'w', encoding='utf-8') as f:
            f.write(default_prompt)
        logger.info(f"📝 Создан файл prompt.txt по умолчанию: {PROMPT_PATH}")
    
    with open(PROMPT_PATH, 'r', encoding='utf-8') as f:
        prompt = f.read().strip()
    
    logger.info(f"✅ System prompt загружен из {PROMPT_PATH}")
    return prompt


def check_faiss_index_exists() -> bool:
    """
    Проверить наличие FAISS-индекса на Google Drive.
    
    Returns:
        True если индекс существует
    """
    index_path = KOIB_FAISS_PATH / "koib_index"
    meta_path = KOIB_FAISS_PATH / "index_meta.json"
    
    exists = index_path.exists() and meta_path.exists()
    if exists:
        logger.info(f"✅ FAISS-индекс найден: {index_path}")
    else:
        logger.info("⚠️ FAISS-индекс не найден на Google Drive")
    
    return exists


def check_text_blocks_exists() -> bool:
    """
    Проверить наличие text_blocks.json на Google Drive.
    
    Returns:
        True если файл существует
    """
    blocks_path = KOIB_METADATA_PATH / "text_blocks.json"
    exists = blocks_path.exists()
    if exists:
        logger.info(f"✅ text_blocks.json найден: {blocks_path}")
    else:
        logger.info("⚠️ text_blocks.json не найден на Google Drive")
    return exists


def create_embedding_model() -> 'HuggingFaceEmbeddings':
    """
    Создать модель эмбеддингов с кэшированием на Google Drive.
    
    Returns:
        Экземпляр HuggingFaceEmbeddings
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    
    # Создаём директорию для кэша моделей
    KOIB_MODELS_PATH.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Загрузка модели эмбеддингов intfloat/multilingual-e5-large...")
    logger.info(f"Кэш моделей: {KOIB_MODELS_PATH}")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        cache_folder=str(KOIB_MODELS_PATH),
        encode_kwargs={"normalize_embeddings": True},
        model_kwargs={"device": "cpu"}  # В Colab Free CPU достаточно
    )
    
    logger.info("✅ Модель эмбеддингов загружена и закэширована")
    return embeddings


def run_preprocessing() -> None:
    """Запустить Part 1: предобработку документов."""
    logger.info("🚀 Запуск Part 1: Предобработка документов...")
    
    # Добавляем src в path
    src_path = Path("/workspace/src")
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    from preprocessing import KoibPreprocessingPipeline
    
    pipeline = KoibPreprocessingPipeline(
        docs_dir=KOIB_BASE_PATH / "docs",
        output_dir=KOIB_OUTPUT_PATH
    )
    
    pipeline.process_all()
    pipeline.save_artifacts()
    
    logger.info("✅ Part 1 завершён")


def run_index_building() -> None:
    """Запустить Part 2: построение FAISS-индекса."""
    logger.info("🚀 Запуск Part 2: Построение FAISS-индекса...")
    
    # Добавляем src в path
    src_path = Path("/workspace/src")
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    from index_building import KoibIndexBuilder
    
    builder = KoibIndexBuilder(
        metadata_dir=KOIB_METADATA_PATH,
        figures_index_path=KOIB_METADATA_PATH / "figures_index.json",
        output_dir=KOIB_OUTPUT_PATH
    )
    
    builder.load_text_blocks()
    builder.build_chunks()
    builder.save_chunks()
    builder.build_faiss_index()
    
    logger.info("✅ Part 2 завершён")


def load_query_engine() -> 'KoibQueryEngine':
    """
    Загрузить Query Engine с существующим индексом.
    
    Returns:
        Экземпляр KoibQueryEngine
    """
    # Добавляем src в path
    src_path = Path("/workspace/src")
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    from query_engine import KoibQueryEngine
    
    engine = KoibQueryEngine(
        faiss_index_path=KOIB_FAISS_PATH / "koib_index",
        figures_index_path=KOIB_METADATA_PATH / "figures_index.json",
        chunks_path=KOIB_METADATA_PATH / "chunks.json",
        output_dir=KOIB_OUTPUT_PATH
    )
    
    logger.info("✅ Query Engine загружен")
    return engine


def initialize() -> Tuple['KoibQueryEngine', str]:
    """
    Главная функция инициализации системы.
    
    Выполняет:
    1. Монтирование Google Drive
    2. Чтение секретов
    3. Чтение system prompt
    4. Проверку/создание FAISS-индекса
    5. Загрузку Query Engine
    
    Returns:
        Кортеж (engine, system_prompt)
    """
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("ИНИЦИАЛИЗАЦИЯ KOIB RAG СИСТЕМЫ")
    logger.info("=" * 60)
    
    # 1. Монтируем Google Drive
    mount_google_drive()
    
    # 2. Создаём необходимые директории
    KOIB_BASE_PATH.mkdir(parents=True, exist_ok=True)
    KOIB_MODELS_PATH.mkdir(parents=True, exist_ok=True)
    KOIB_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    # 3. Получаем секреты (проверка наличия)
    try:
        gigachat_creds, vk_token = get_secrets()
        # Сохраняем в переменные окружения для использования в других модулях
        os.environ['GIGACHAT_CREDENTIALS'] = gigachat_creds
        os.environ['VK_GROUP_TOKEN'] = vk_token
    except Exception as e:
        logger.warning(f"⚠️ Секреты не найдены: {e}")
        logger.warning("Убедитесь, что вы добавили GIGACHAT_CREDENTIALS и VK_GROUP_TOKEN в Colab Secrets")
    
    # 4. Читаем system prompt
    system_prompt = get_system_prompt()
    
    # 5. Проверяем наличие индекса и строим при необходимости
    if check_faiss_index_exists():
        # Индекс есть - просто загружаем
        logger.info("Загрузка существующего FAISS-индекса...")
        engine = load_query_engine()
    else:
        # Индекса нет - запускаем полный пайплайн
        logger.info("FAISS-индекс не найден. Запуск полного пайплайна...")
        
        if not check_text_blocks_exists():
            # Нет text_blocks - запускаем Part 1
            run_preprocessing()
        
        # Запускаем Part 2
        run_index_building()
        
        # Загружаем engine
        engine = load_query_engine()
    
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"✅ ИНИЦИАЛИЗАЦИЯ ЗАВЕРШЕНА ЗА {elapsed:.1f} сек")
    logger.info("=" * 60)
    
    return engine, system_prompt


if __name__ == "__main__":
    # Тестовый запуск
    engine, prompt = initialize()
    print("\n✅ Система готова к работе!")
    print(f"System prompt (первые 100 символов): {prompt[:100]}...")
