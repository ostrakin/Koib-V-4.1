# -*- coding: utf-8 -*-
"""
KOIB RAG - Query Engine Module (v4.1)

Часть 3 системы RAG: поисковый движок, интерактивные запросы,
интеграция с GigaChat LLM.

Изменения v4.1:
- [IMPROVEMENT] Улучшена обработка ошибок при загрузке индекса
- [IMPROVEMENT] Добавлен интерактивный CLI с историей запросов
- [IMPROVEMENT] Поддержка фильтрации по модели КОИБ
- [IMPROVEMENT] ask_with_gigachat() для end-to-end RAG
"""

import os
import json
import re
import time
import datetime
import hashlib
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

try:
    from .utils import (
        clean_text, text_hash, normalize_model_key,
        get_output_dir, MODEL_DISPLAY_NAMES, KOIB_MODEL_PATTERNS
    )
except ImportError:
    from utils import (
        clean_text, text_hash, normalize_model_key,
        get_output_dir, MODEL_DISPLAY_NAMES, KOIB_MODEL_PATTERNS
    )

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "intfloat/multilingual-e5-large"
)
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "
FAISS_SEARCH_K = int(os.getenv("FAISS_SEARCH_K", "5"))


class KoibQueryEngine:
    """
    Поисковый движок для системы RAG КОИБ.

    Использование:
        engine = KoibQueryEngine()
        context, docs, figures = engine.ask("Как включить КОИБ-2010?")
    """

    def __init__(
        self,
        faiss_index_path: Optional[Path] = None,
        figures_index_path: Optional[Path] = None,
        chunks_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ):
        self.output_dir = output_dir or get_output_dir()
        self.metadata_dir = self.output_dir / "metadata"
        self.faiss_index_dir = self.output_dir / "faiss_index"
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.vectorstore: Optional[FAISS] = None
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self.figures_index: List[Dict] = []
        self.chunks_data: List[Dict] = []

        if faiss_index_path is None:
            faiss_index_path = self.faiss_index_dir / "koib_index"
        if figures_index_path is None:
            figures_index_path = self.metadata_dir / "figures_index.json"
        if chunks_path is None:
            chunks_path = self.metadata_dir / "chunks.json"

        # Загрузка эмбеддинг-модели и FAISS
        try:
            device = "cuda" if self._has_cuda() else "cpu"
            print(f"Загружаем модель {EMBEDDING_MODEL_NAME} на {device}...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                encode_kwargs={"normalize_embeddings": True},
                model_kwargs={"device": device},
            )

            if Path(faiss_index_path).exists():
                self.vectorstore = FAISS.load_local(
                    str(faiss_index_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                print(f"✅ FAISS индекс загружен: {faiss_index_path}")
            else:
                print(f"⚠️  FAISS индекс не найден: {faiss_index_path}")
                print("   Запустите: python -m src.index_building")
        except Exception as exc:
            print(f"❌ Ошибка загрузки FAISS: {exc}")

        # Загрузка индекса рисунков
        try:
            if Path(figures_index_path).exists():
                with open(figures_index_path, 'r', encoding='utf-8') as f:
                    self.figures_index = json.load(f)
                print(f"✅ Рисунков в индексе: {len(self.figures_index)}")
        except Exception as exc:
            print(f"⚠️  figures_index: {exc}")

        # Загрузка чанков (для статистики/отладки)
        try:
            if Path(chunks_path).exists():
                with open(chunks_path, 'r', encoding='utf-8') as f:
                    self.chunks_data = json.load(f)
                print(f"✅ Чанков загружено: {len(self.chunks_data)}")
        except Exception:
            pass

    def _has_cuda(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    # ------------------------------------------------------------------

    def _search(
        self,
        query: str,
        model_filter: str = "",
        k: int = FAISS_SEARCH_K,
    ) -> List[Tuple[Document, float]]:
        """Поиск в векторном индексе с опциональным фильтром по модели."""
        if self.vectorstore is None:
            print("❌ FAISS индекс не загружен!")
            return []

        query_text = QUERY_PREFIX + query
        results = self.vectorstore.similarity_search_with_score(query_text, k=k * 3)

        if model_filter:
            mf = normalize_model_key(model_filter)
            results = [
                (doc, score) for doc, score in results
                if doc.metadata.get("model") == mf
            ][:k]
        else:
            results = results[:k]

        return results

    def ask(
        self,
        query: str,
        koib_model: str = "",
        k: int = FAISS_SEARCH_K,
    ) -> Tuple[str, List[Dict], List[Dict]]:
        """
        Выполнить поиск и вернуть (context_text, docs, figures).

        Args:
            query:      Поисковый запрос
            koib_model: Фильтр модели ("koib2010" / "koib2017a" / "koib2017b")
            k:          Количество результатов

        Returns:
            (context_text, relevant_docs, relevant_figures)
        """
        results = self._search(query, koib_model, k)
        if not results:
            return "", [], []

        context_parts: List[str] = []
        relevant_docs: List[Dict] = []
        sources_used: set = set()

        for doc, score in results:
            meta = doc.metadata
            source = meta.get("source", "unknown")
            page = meta.get("page", "?")
            model = meta.get("model", "unknown")
            has_figs = meta.get("has_figures", False)
            headings = meta.get("headings", "")
            captions = meta.get("captions", "")

            source_key = f"{source}_p{page}"
            if source_key in sources_used:
                continue
            sources_used.add(source_key)

            model_display = MODEL_DISPLAY_NAMES.get(model, model)
            src_name = Path(source).name if source != "unknown" else source

            header = f"--- [{model_display}] {src_name}, стр. {page}"
            if headings:
                header += f" | {headings[:80]}"
            if has_figs:
                header += " [🖼️ есть рисунки]"
            if captions:
                header += f"\n    Подписи: {captions[:100]}"

            context_parts.append(f"{header}\n{doc.page_content}")
            relevant_docs.append({
                "source": source,
                "source_name": src_name,
                "page": page,
                "model": model,
                "model_display": model_display,
                "score": round(float(score), 4),
                "text_preview": (doc.page_content[:300] + "…")
                                if len(doc.page_content) > 300 else doc.page_content,
                "has_figures": has_figs,
            })

        relevant_figures = self._find_figures(query, koib_model, sources_used)
        context_text = "\n\n".join(context_parts)
        return context_text, relevant_docs, relevant_figures

    def _find_figures(
        self,
        query: str,
        koib_model: str,
        sources_used: set,
    ) -> List[Dict]:
        if not self.figures_index:
            return []

        model_filter = normalize_model_key(koib_model) if koib_model else ""
        stop_words = {
            'как', 'что', 'где', 'когда', 'почему', 'для', 'это',
            'на', 'в', 'с', 'и', 'или', 'не', 'по', 'к', 'от', 'до', 'при', 'о'
        }
        query_words = set(re.findall(r'\w+', query.lower())) - stop_words

        scored: List[Tuple[Dict, float]] = []
        for fig in self.figures_index:
            if model_filter and fig.get("model") != model_filter:
                continue
            fig_source = fig.get("source") or fig.get("file", "")
            fig_key = f"{fig_source}_p{fig.get('page', 0)}"
            if sources_used and fig_key not in sources_used:
                continue

            combined = (
                (fig.get("caption") or "") + " " +
                (fig.get("surrounding_text") or "")
            ).lower()
            fig_words = set(re.findall(r'\w+', combined))
            overlap = query_words & fig_words
            score = len(overlap) / max(len(query_words), 1)
            if score > 0.1 or not query_words:
                scored.append((fig, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [f for f, _ in scored[:5]]

    def ask_with_llm_context(
        self,
        query: str,
        koib_model: str = "",
        k: int = FAISS_SEARCH_K,
    ) -> str:
        """Сформировать готовый промпт для LLM с контекстом RAG."""
        context_text, docs, figures = self.ask(query, koib_model, k)
        if not context_text:
            return "Контекст не найден. Нет релевантных документов по данному запросу."

        model_display = (
            MODEL_DISPLAY_NAMES.get(normalize_model_key(koib_model), "Все модели")
            if koib_model else "Все модели"
        )

        parts = [
            f"КОНТЕКСТ ИЗ ТЕХНИЧЕСКОЙ ДОКУМЕНТАЦИИ КОИБ ({model_display}):",
            "",
            context_text,
        ]

        if figures:
            parts += ["", "РЕЛЕВАНТНЫЕ РИСУНКИ И СХЕМЫ:"]
            for fig in figures:
                fig_model = MODEL_DISPLAY_NAMES.get(
                    fig.get("model", ""), fig.get("model", "")
                )
                src_name = Path(fig.get("source") or fig.get("file", "")).name
                caption = fig.get("caption") or "без подписи"
                parts.append(f"  - [{fig_model}] {src_name}: {caption}")

        parts += [
            "",
            "ИНСТРУКЦИЯ:",
            "На основе контекста из технической документации КОИБ ответь на вопрос:",
            f"«{query}»",
            "Если ответ не содержится в контексте, явно укажи это.",
            "При ответе ссылайся на конкретный документ и страницу.",
        ]

        return "\n".join(parts)

    def ask_with_gigachat(
        self,
        query: str,
        koib_model: str = "",
        gigachat_credentials: str = "",
        k: int = FAISS_SEARCH_K,
    ) -> str:
        """
        Полный RAG-ответ: поиск + GigaChat генерация.

        Args:
            query:                 Вопрос пользователя
            koib_model:            Фильтр модели
            gigachat_credentials:  Токен GigaChat (или из GIGACHAT_CREDENTIALS)
            k:                     Кол-во чанков для контекста

        Returns:
            Текст ответа от GigaChat
        """
        try:
            import sys
            # Импортируем GigaChat клиент из корня проекта
            root = Path(__file__).parent.parent
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            from gigachat_client import GigaChatClient
        except ImportError:
            return "❌ gigachat_client.py не найден. Убедитесь что он в корне проекта."

        creds = gigachat_credentials or os.getenv("GIGACHAT_CREDENTIALS", "")
        if not creds:
            return "❌ Не задан GIGACHAT_CREDENTIALS (переменная окружения или параметр)."

        prompt = self.ask_with_llm_context(query, koib_model, k)
        client = GigaChatClient(creds)
        return client.chat(prompt)

    def print_results(self, docs: List[Dict], figures: List[Dict]) -> None:
        """Вывести результаты поиска в читаемом формате."""
        if not docs:
            print("  Ничего не найдено.")
            return
        print(f"\n  📄 Найдено {len(docs)} релевантных фрагментов:")
        for i, d in enumerate(docs, 1):
            score_str = f"score={d['score']:.4f}"
            print(f"  {i}. [{d['model_display']}] {d['source_name']}, стр. {d['page']} ({score_str})")
            print(f"     {d['text_preview'][:120]}…")
        if figures:
            print(f"\n  🖼️  Релевантных рисунков: {len(figures)}")
            for fig in figures:
                print(f"     - {Path(fig.get('source','?')).name}: {fig.get('caption','—')}")


def _run_interactive_cli(engine: "KoibQueryEngine") -> None:
    """Интерактивный CLI для запросов к RAG."""
    print("\n" + "=" * 70)
    print("  ИНТЕРАКТИВНЫЙ РЕЖИМ  |  введите 'q' для выхода")
    print("  Фильтр модели: введите 'model koib2010' (или 2017a / 2017b / all)")
    print("=" * 70)

    koib_model = ""
    history: List[str] = []

    while True:
        try:
            print()
            raw = input("❓ Вопрос: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nВыход.")
            break

        if not raw or raw.lower() in ('q', 'quit', 'exit', 'выход'):
            print("До свидания!")
            break

        if raw.lower().startswith("model "):
            arg = raw.split(None, 1)[1].strip().lower()
            if arg == "all":
                koib_model = ""
                print("  Фильтр модели снят")
            else:
                koib_model = arg
                from utils import normalize_model_key, MODEL_DISPLAY_NAMES
                display = MODEL_DISPLAY_NAMES.get(normalize_model_key(koib_model), koib_model)
                print(f"  Фильтр: {display}")
            continue

        history.append(raw)
        t0 = time.time()
        context, docs, figures = engine.ask(raw, koib_model)
        elapsed = time.time() - t0

        engine.print_results(docs, figures)
        print(f"  ⏱️  {elapsed:.2f}с")

    # Сохранение истории
    if history:
        log_path = engine.logs_dir / f"cli_history_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        print(f"\n📝 История запросов сохранена: {log_path}")


def main() -> None:
    print("╔" + "═" * 68 + "╗")
    print("║" + "  KOIB RAG v4.1 – ЧАСТЬ 3: QUERY ENGINE".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    output_dir = get_output_dir()
    engine = KoibQueryEngine(output_dir=output_dir)

    if engine.vectorstore is None:
        print("❌ Query engine не инициализирован.")
        print("   Убедитесь, что Parts 1 и 2 выполнены успешно.")
        return

    # Автотест
    print("\n🧪 Автотестовый запрос...")
    _, docs, _ = engine.ask("Как включить КОИБ?", k=3)
    if docs:
        print(f"✅ Тест пройден: {len(docs)} результатов")
    else:
        print("⚠️  Тестовый запрос не вернул результатов (это нормально без документов)")

    # Интерактивный режим
    _run_interactive_cli(engine)


if __name__ == "__main__":
    main()
