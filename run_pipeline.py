# -*- coding: utf-8 -*-
"""
KOIB RAG - Полный пайплайн (запуск всех трёх частей)

Использование:
    python run_pipeline.py            # всё с настройками по умолчанию
    python run_pipeline.py --only-preprocess
    python run_pipeline.py --only-index
    python run_pipeline.py --interactive  # только query engine
    python run_pipeline.py --query "Как включить КОИБ-2010?"
"""

import sys
import time
import argparse
import json
from pathlib import Path

# Добавляем корень проекта в sys.path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.utils import get_docs_dir, get_output_dir, ensure_dirs


def run_preprocessing(docs_dir: Path, output_dir: Path) -> bool:
    """Запустить Part 1: preprocessing."""
    print("\n" + "━" * 70)
    print("  ЧАСТЬ 1 / 3: PREPROCESSING")
    print("━" * 70)
    try:
        from src.preprocessing import KoibPreprocessingPipeline, generate_source_models
        metadata_dir = output_dir / "metadata"
        sm_path = metadata_dir / "source_models.json"
        if sm_path.exists():
            with open(sm_path, 'r', encoding='utf-8') as f:
                source_models = json.load(f)
        else:
            source_models = generate_source_models(docs_dir, metadata_dir)

        pipeline = KoibPreprocessingPipeline(docs_dir, output_dir, source_models)
        blocks = pipeline.process_all()
        pipeline.save_artifacts()
        pipeline.print_summary()

        if not blocks:
            print("\n⚠️  Предупреждение: текстовые блоки не созданы.")
            print("   Убедитесь, что в data/docs/ есть PDF или DOCX файлы.")
            return False
        return True
    except Exception as exc:
        print(f"❌ Ошибка preprocessing: {exc}")
        import traceback
        traceback.print_exc()
        return False


def run_index_building(output_dir: Path) -> bool:
    """Запустить Part 2: index building."""
    print("\n" + "━" * 70)
    print("  ЧАСТЬ 2 / 3: INDEX BUILDING")
    print("━" * 70)
    try:
        from src.index_building import KoibIndexBuilder
        metadata_dir = output_dir / "metadata"
        figures_path = metadata_dir / "figures_index.json"

        builder = KoibIndexBuilder(
            metadata_dir=metadata_dir,
            figures_index_path=figures_path if figures_path.exists() else None,
            output_dir=output_dir,
        )
        builder.load_text_blocks()
        chunks = builder.build_chunks()
        if not chunks:
            print("❌ Нет чанков для индексирования.")
            return False
        builder.save_chunks()
        builder.build_faiss_index()
        return True
    except FileNotFoundError as exc:
        print(exc)
        return False
    except Exception as exc:
        print(f"❌ Ошибка index building: {exc}")
        import traceback
        traceback.print_exc()
        return False


def run_query(output_dir: Path, query: str, koib_model: str = "") -> None:
    """Выполнить один запрос и вывести результат."""
    from src.query_engine import KoibQueryEngine
    engine = KoibQueryEngine(output_dir=output_dir)
    if engine.vectorstore is None:
        print("❌ Индекс не загружен. Запустите pipeline полностью.")
        return
    context, docs, figures = engine.ask(query, koib_model)
    engine.print_results(docs, figures)
    if context:
        print("\n─ Контекст для LLM ─────────────────────────────────")
        print(context[:1500])
        if len(context) > 1500:
            print(f"  … (всего {len(context)} символов)")


def run_interactive(output_dir: Path) -> None:
    """Запустить интерактивный CLI."""
    from src.query_engine import KoibQueryEngine, _run_interactive_cli
    engine = KoibQueryEngine(output_dir=output_dir)
    if engine.vectorstore is None:
        print("❌ Индекс не загружен. Сначала запустите пайплайн полностью.")
        return
    _run_interactive_cli(engine)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="KOIB RAG Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--only-preprocess", action="store_true",
                        help="Запустить только preprocessing (Part 1)")
    parser.add_argument("--only-index", action="store_true",
                        help="Запустить только index building (Part 2)")
    parser.add_argument("--interactive", action="store_true",
                        help="Интерактивный CLI (Part 3)")
    parser.add_argument("--query", type=str, default="",
                        help="Один запрос к RAG (Part 3)")
    parser.add_argument("--model", type=str, default="",
                        help="Фильтр модели: koib2010 / koib2017a / koib2017b")
    parser.add_argument("--docs-dir", type=str, default="",
                        help="Директория с документами (по умолчанию data/docs)")
    parser.add_argument("--output-dir", type=str, default="",
                        help="Выходная директория (по умолчанию output)")
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir) if args.docs_dir else get_docs_dir()
    output_dir = Path(args.output_dir) if args.output_dir else get_output_dir()
    ensure_dirs(docs_dir, output_dir)

    print("╔" + "═" * 68 + "╗")
    print("║" + "  KOIB RAG v4.1 — ПОЛНЫЙ ПАЙПЛАЙН".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print(f"  Документы:  {docs_dir}")
    print(f"  Выход:      {output_dir}")

    t_global = time.time()

    if args.only_preprocess:
        run_preprocessing(docs_dir, output_dir)

    elif args.only_index:
        run_index_building(output_dir)

    elif args.interactive:
        run_interactive(output_dir)

    elif args.query:
        run_query(output_dir, args.query, args.model)

    else:
        # Полный пайплайн
        ok1 = run_preprocessing(docs_dir, output_dir)
        if ok1:
            ok2 = run_index_building(output_dir)
            if ok2:
                print("\n✅ Пайплайн завершён успешно!")
                print("   Для запросов: python run_pipeline.py --interactive")
                print("   Или:          python run_pipeline.py --query 'Как включить?'")
        else:
            print("\n⚠️  Part 1 завершился с предупреждениями. Проверьте данные.")

    elapsed = time.time() - t_global
    print(f"\n⏱️  Общее время: {elapsed:.1f}с ({elapsed/60:.1f} мин)")


if __name__ == "__main__":
    main()
