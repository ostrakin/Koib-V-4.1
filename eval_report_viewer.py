# -*- coding: utf-8 -*-
"""
Визуализатор отчёта RAG-оценки
================================
Читает eval_report.json и печатает красивый текстовый дашборд.

Запуск:
  python eval_report_viewer.py
  python eval_report_viewer.py --report my_report.json
"""

import argparse
import json
from pathlib import Path


METRIC_NAMES = {
    "faithfulness":       "Faithfulness      (верность контексту) ",
    "answer_relevancy":   "Answer Relevancy  (релевантность ответа)",
    "context_precision":  "Context Precision (точность контекста) ",
    "context_recall":     "Context Recall    (полнота контекста)  ",
}

def bar(value: float, width: int = 30) -> str:
    """Текстовый progress-bar."""
    filled = int(round(value * width))
    empty  = width - filled
    color_code = (
        "\033[92m" if value >= 0.75 else   # зелёный
        "\033[93m" if value >= 0.50 else   # жёлтый
        "\033[91m"                          # красный
    )
    reset = "\033[0m"
    return f"{color_code}{'█' * filled}{'░' * empty}{reset}"

def rating(value: float) -> str:
    if value >= 0.85: return "★★★★★ Отлично"
    if value >= 0.70: return "★★★★☆ Хорошо"
    if value >= 0.55: return "★★★☆☆ Удовлетворительно"
    if value >= 0.40: return "★★☆☆☆ Слабо"
    return "★☆☆☆☆ Неудовлетворительно"


def main():
    parser = argparse.ArgumentParser(description="Просмотр отчёта RAG-оценки")
    parser.add_argument("--report", default="eval_report.json")
    args = parser.parse_args()

    path = Path(args.report)
    if not path.exists():
        print(f"[ERROR] Файл не найден: {path}")
        return

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    summary = data["summary"]
    results = data["results"]
    ok      = [r for r in results if not r.get("error")]

    W = 70
    print()
    print("═" * W)
    print("  ОТЧЁТ КАЧЕСТВА RAG-СИСТЕМЫ  КОИБ v4.1")
    print("═" * W)
    print(f"  Вопросов: {summary['successful']}/{summary['total']}  |  "
          f"Итоговый RAG Score: {summary.get('avg_rag_score', 0):.3f}  "
          f"— {rating(summary.get('avg_rag_score', 0))}")
    print()

    # Метрики со шкалой
    for key, label in METRIC_NAMES.items():
        avg_key = f"avg_{key}"
        val = summary.get(avg_key, 0.0)
        print(f"  {label}  {val:.3f}  {bar(val)}")

    print()
    print("─" * W)
    print("  ДЕТАЛИЗАЦИЯ ПО ВОПРОСАМ")
    print("─" * W)

    for r in results:
        qid      = r["question_id"]
        question = r["question"][:55] + ("…" if len(r["question"]) > 55 else "")
        cat      = r.get("category", "—")

        if r.get("error"):
            print(f"  [{qid}] ❌ ОШИБКА: {r['error'][:50]}")
            continue

        rag = (r["faithfulness"] + r["answer_relevancy"] +
               r["context_precision"] + r["context_recall"]) / 4

        color = (
            "\033[92m" if rag >= 0.75 else
            "\033[93m" if rag >= 0.50 else
            "\033[91m"
        )
        reset = "\033[0m"

        print(f"  {color}[{qid}]{reset} {question}")
        print(f"    RAG={rag:.3f}  F={r['faithfulness']:.2f}  "
              f"AR={r['answer_relevancy']:.2f}  "
              f"CP={r['context_precision']:.2f}  "
              f"CR={r['context_recall']:.2f}  "
              f"| {cat}  | {r.get('latency_sec', 0)}s")

        if r.get("has_reference") and r.get("token_f1", 0) > 0:
            print(f"    Token F1 (vs эталон): {r['token_f1']:.3f}")

        # Превью ответа
        ans = r.get("answer", "")
        if ans:
            preview = ans[:120].replace("\n", " ")
            print(f"    ↳ {preview}{'…' if len(ans) > 120 else ''}")
        print()

    # Рекомендации
    if ok:
        metrics = {
            "faithfulness":      summary.get("avg_faithfulness", 0),
            "answer_relevancy":  summary.get("avg_answer_relevancy", 0),
            "context_precision": summary.get("avg_context_precision", 0),
            "context_recall":    summary.get("avg_context_recall", 0),
        }
        sorted_metrics = sorted(metrics.items(), key=lambda x: x[1])

        print("─" * W)
        print("  РЕКОМЕНДАЦИИ ПО УЛУЧШЕНИЮ")
        print("─" * W)
        tips = {
            "faithfulness": [
                "Добавьте в системный промпт строгий запрет домыслов:",
                '«Отвечай ТОЛЬКО на основе предоставленного контекста. Если ответа нет — скажи об этом.»',
            ],
            "answer_relevancy": [
                "Ответы уходят в сторону — пересмотрите шаблон промпта в query_engine.py.",
                "Убедитесь, что вопрос передаётся в GigaChat в явном виде.",
            ],
            "context_precision": [
                "Много лишних чанков — попробуйте уменьшить top_k (напр. 3 вместо 5).",
                "Также можно снизить CHUNK_SIZE для более точечных фрагментов.",
            ],
            "context_recall": [
                "Контекст неполный — увеличьте top_k или CHUNK_OVERLAP.",
                "Проверьте правильность фильтрации по модели КОИБ.",
            ],
        }

        for metric, val in sorted_metrics[:2]:  # Топ-2 слабых места
            label = METRIC_NAMES[metric].strip()
            print(f"\n  ⚠  {label} = {val:.3f}")
            for tip in tips[metric]:
                print(f"     • {tip}")

    print()
    print("═" * W)


if __name__ == "__main__":
    main()
