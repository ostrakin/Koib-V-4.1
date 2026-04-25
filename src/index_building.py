# -*- coding: utf-8 -*-
"""
KOIB RAG - Index Building Module (v4.1 — fixed)

Часть 2 системы RAG: чанкирование текстов, построение FAISS индекса
с multilingual эмбеддингами.

Исправления v4.1:
- [BUG FIX] build_chunks() теперь читает "source" И "file" (алиас) —
  совместимо с обоими вариантами text_blocks.json
- [BUG FIX] Поля block_type, headings, caption обрабатываются безопасно
  (через .get() с дефолтами — не падает на старых блоках)
- [BUG FIX] model=unknown теперь корректно группируется и попадает в индекс
- [IMPROVEMENT] Прогресс-бар при создании чанков
- [IMPROVEMENT] Подробное логирование
"""

import os
import json
import re
import time
import datetime
import hashlib
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

try:
    from .utils import (
        clean_text, text_hash, normalize_model_key,
        get_output_dir, MODEL_DISPLAY_NAMES
    )
except ImportError:
    from utils import (
        clean_text, text_hash, normalize_model_key,
        get_output_dir, MODEL_DISPLAY_NAMES
    )

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "2000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "320"))
MIN_CHUNK_LEN = int(os.getenv("MIN_CHUNK_LEN", "120"))

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "intfloat/multilingual-e5-large"
)
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "


def _get_block_source(block: Dict) -> str:
    """
    Получить путь к источнику из блока.
    BUG FIX: поддерживает оба имени поля — "source" и "file" (alias).
    """
    return block.get("source") or block.get("file") or ""


class KoibIndexBuilder:
    """
    Построитель векторного индекса для системы RAG КОИБ.

    Загружает text_blocks.json (Part 1), разбивает на чанки,
    строит FAISS индекс с multilingual-e5-large эмбеддингами.
    """

    def __init__(
        self,
        metadata_dir: Optional[Path] = None,
        figures_index_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ):
        self.output_dir = output_dir or get_output_dir()
        self.metadata_dir = metadata_dir or (self.output_dir / "metadata")
        self.faiss_index_dir = self.output_dir / "faiss_index"
        self.logs_dir = self.output_dir / "logs"

        self.faiss_index_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.figures_index: List[Dict] = []
        if figures_index_path and Path(figures_index_path).exists():
            with open(figures_index_path, 'r', encoding='utf-8') as f:
                self.figures_index = json.load(f)

        self.text_blocks: List[Dict] = []
        self.chunks: List[Document] = []
        self.vectorstore: Optional[FAISS] = None

    # ------------------------------------------------------------------
    # Загрузка блоков
    # ------------------------------------------------------------------

    def load_text_blocks(self) -> List[Dict]:
        blocks_path = self.metadata_dir / "text_blocks.json"
        if not blocks_path.exists():
            raise FileNotFoundError(
                f"❌ text_blocks.json не найден: {blocks_path}\n"
                "   Сначала запустите: python -m src.preprocessing"
            )
        with open(blocks_path, 'r', encoding='utf-8') as f:
            self.text_blocks = json.load(f)
        print(f"✅ Загружено {len(self.text_blocks)} блоков из {blocks_path}")
        return self.text_blocks

    # ------------------------------------------------------------------
    # Построение чанков
    # ------------------------------------------------------------------

    def build_chunks(self) -> List[Document]:
        """
        Создать чанки из текстовых блоков.

        BUG FIX: читает "source" | "file", безопасно берёт model/headings/
        block_type через .get() — не падает при отсутствии полей.
        """
        if not self.text_blocks:
            self.load_text_blocks()

        # Группировка: (model, source, page)
        groups: Dict[tuple, List[Dict]] = defaultdict(list)
        for block in self.text_blocks:
            key = (
                block.get("model", "unknown"),   # BUG FIX: раньше отсутствовало
                _get_block_source(block),         # BUG FIX: "source" or "file"
                block.get("page", 0),
            )
            groups[key].append(block)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        all_chunks: List[Document] = []
        seen_hashes: set = set()

        # Порядок типов блоков
        type_order = {"heading": 0, "text": 1, "table": 2, "ocr_text": 3}

        for (model, source, page), blocks in groups.items():
            # Сортировка по типу
            blocks.sort(key=lambda b: type_order.get(b.get("block_type", "text"), 1))

            combined = clean_text(
                "\n\n".join(b.get("text", "") for b in blocks if b.get("text"))
            )
            if not combined or len(combined) < MIN_CHUNK_LEN:
                continue

            split_texts = splitter.split_text(combined)

            # Метаданные: заголовки и подписи
            all_headings: List[str] = []
            all_captions: List[str] = []
            for b in blocks:
                # headings может быть list или строкой
                bh = b.get("headings", [])
                if isinstance(bh, list):
                    all_headings.extend(bh)
                elif bh:
                    all_headings.append(str(bh))
                if b.get("block_type") == "heading":
                    all_headings.append(b.get("text", ""))
                cap = b.get("caption", "")
                if cap:
                    all_captions.append(cap)

            # Наличие рисунков на этой странице
            has_figures = any(
                f.get("model") == model
                and _get_block_source(f) == source
                and f.get("page") == page
                for f in self.figures_index
            )

            for chunk_text in split_texts:
                chunk_text = chunk_text.strip()
                if len(chunk_text) < MIN_CHUNK_LEN:
                    continue
                h = text_hash(chunk_text)
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)

                doc = Document(
                    page_content=chunk_text,
                    metadata={
                        "model": model,
                        "source": source,
                        "page": page,
                        "chunk_type": "mixed",
                        "has_figures": has_figures,
                        "captions": "; ".join(all_captions[:3]),
                        "headings": "; ".join(all_headings[:5]),
                    }
                )
                all_chunks.append(doc)

        self.chunks = all_chunks
        print(f"✅ Создано {len(self.chunks)} чанков из {len(self.text_blocks)} блоков")
        self._log_chunk_stats()
        return self.chunks

    def _log_chunk_stats(self) -> None:
        """Вывести статистику по чанкам."""
        by_model: Dict[str, int] = defaultdict(int)
        for doc in self.chunks:
            by_model[doc.metadata.get("model", "unknown")] += 1
        print(f"   Распределение по моделям: {dict(by_model)}")

    # ------------------------------------------------------------------
    # Сохранение чанков
    # ------------------------------------------------------------------

    def save_chunks(self) -> None:
        chunks_path = self.metadata_dir / "chunks.json"
        chunks_data = [
            {"text": doc.page_content, "metadata": doc.metadata}
            for doc in self.chunks
        ]
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
        print(f"✅ Чанки сохранены: {chunks_path} ({len(self.chunks)} шт.)")

    # ------------------------------------------------------------------
    # Построение FAISS индекса
    # ------------------------------------------------------------------

    def build_faiss_index(self) -> FAISS:
        if not self.chunks:
            self.build_chunks()
        if not self.chunks:
            raise ValueError("Нет чанков для индексирования. Проверьте text_blocks.json.")

        print(f"\n🚀 Строим FAISS индекс (модель: {EMBEDDING_MODEL_NAME})")

        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
        print(f"   Устройство: {device}")

        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            encode_kwargs={"normalize_embeddings": True},
            model_kwargs={"device": device},
        )

        # passage: префикс для интрукционных эмбеддингов
        texts = [PASSAGE_PREFIX + doc.page_content for doc in self.chunks]
        metadatas = [doc.metadata for doc in self.chunks]

        print(f"   Индексируем {len(texts)} текстов...")
        self.vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

        index_path = self.faiss_index_dir / "koib_index"
        self.vectorstore.save_local(str(index_path))
        print(f"✅ FAISS индекс сохранён: {index_path}")

        meta_path = self.faiss_index_dir / "index_meta.json"
        meta = {
            "model_name": EMBEDDING_MODEL_NAME,
            "num_chunks": len(self.chunks),
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "passage_prefix": PASSAGE_PREFIX,
            "created": datetime.datetime.now().isoformat(),
            "device": device,
        }
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"✅ Метаданные индекса: {meta_path}")

        return self.vectorstore


def main() -> None:
    print("╔" + "═" * 68 + "╗")
    print("║" + "  KOIB RAG v4.1 – ЧАСТЬ 2: INDEX BUILDING".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    t0 = time.time()

    output_dir = get_output_dir()
    metadata_dir = output_dir / "metadata"

    if not metadata_dir.exists():
        print(f"❌ Директория метаданных не найдена: {metadata_dir}")
        print("   Сначала запустите: python -m src.preprocessing")
        return

    figures_path = metadata_dir / "figures_index.json"
    if not figures_path.exists():
        print("⚠️  figures_index.json не найден — продолжаем без рисунков")
        figures_path = None

    builder = KoibIndexBuilder(metadata_dir, figures_path, output_dir)

    try:
        builder.load_text_blocks()
    except FileNotFoundError as e:
        print(e)
        return

    chunks = builder.build_chunks()
    if not chunks:
        print("❌ Чанки не созданы. Проверьте text_blocks.json.")
        return

    builder.save_chunks()

    try:
        builder.build_faiss_index()
    except Exception as e:
        print(f"❌ Ошибка построения FAISS: {e}")
        import traceback
        traceback.print_exc()
        return

    elapsed = time.time() - t0
    print(f"\n⏱️  Время: {elapsed:.1f}с ({elapsed/60:.1f} мин)")
    print("✅ ЧАСТЬ 2 ЗАВЕРШЕНА. Запустите ЧАСТЬ 3: python -m src.query_engine")


if __name__ == "__main__":
    main()
