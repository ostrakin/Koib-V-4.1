# -*- coding: utf-8 -*-
"""
RAG Evaluation Framework для КОИБ v4.1
=======================================
Оценивает качество RAG-системы по 4 метрикам (LLM-as-Judge через GigaChat):
  1. Faithfulness       — верность ответа контексту (0–1)
  2. Answer Relevancy   — релевантность ответа вопросу (0–1)
  3. Context Precision  — точность: доля полезных чанков (0–1)
  4. Context Recall     — полнота: покрытие нужных фактов контекстом (0–1)
  5. F1 Token Match     — токен-совпадение с эталоном (если задан)

Запуск:
  python evaluate_rag.py --credentials YOUR_GIGACHAT_TOKEN
  python evaluate_rag.py --credentials YOUR_TOKEN --dataset eval_dataset.json
  python evaluate_rag.py --credentials YOUR_TOKEN --top_k 5 --output report.json

КУДА ВСТАВИТЬ API:
  1. Получите credentials в личном кабинете Сбербанк GigaChat
     (client_id:client_secret в base64)
  2. Передайте через параметр --credentials или установите переменную окружения:
     export GIGACHAT_CREDENTIALS="your_base64_credentials_here"
  3. Пример запуска:
     python evaluate_rag.py --credentials "YOUR_BASE64_TOKEN_HERE" --dataset eval_dataset.json
"""

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, List, Dict, Any
import requests
import urllib3

# ── Подавляем SSL-предупреждения GigaChat ──────────────────────────────────
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ── Добавляем корень проекта в sys.path ────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.query_engine import KoibQueryEngine
except ImportError as e:
    print(f"[ERROR] Не удалось импортировать KoibQueryEngine: {e}")
    print("Убедитесь, что скрипт лежит в корне проекта рядом с папкой src/")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# GigaChat клиент (судья)
# ══════════════════════════════════════════════════════════════════════════════

class GigaChatJudge:
    """Вызывает GigaChat для получения оценок."""

    AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    API_URL  = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"

    def __init__(self, credentials: str, model: str = "GigaChat"):
        self.credentials = credentials
        self.model = model
        self._token: Optional[str] = None
        self._token_expires: float = 0.0

    def _refresh_token(self) -> None:
        resp = requests.post(
            self.AUTH_URL,
            headers={
                "Authorization": f"Basic {self.credentials}",
                "RqUID": "eval-judge-001",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={"scope": "GIGACHAT_API_PERS"},
            verify=False,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        self._token = data["access_token"]
        self._token_expires = time.time() + data.get("expires_in", 1800) - 60

    def _get_token(self) -> str:
        if not self._token or time.time() > self._token_expires:
            self._refresh_token()
        return self._token

    def ask(self, prompt: str, max_tokens: int = 200) -> str:
        token = self._get_token()
        resp = requests.post(
            self.API_URL,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.1,
            },
            verify=False,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def score(self, prompt: str) -> float:
        """Извлекает число 0–10 из ответа и нормирует до 0–1."""
        raw = self.ask(prompt, max_tokens=50)
        nums = re.findall(r"\b(\d+(?:\.\d+)?)\b", raw)
        for n in nums:
            val = float(n)
            if 0 <= val <= 10:
                return round(val / 10.0, 3)
        return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Промпты-судьи
# ══════════════════════════════════════════════════════════════════════════════

PROMPT_FAITHFULNESS = """
Ты — строгий судья качества AI-ответов. Оцени ВЕРНОСТЬ ответа относительно контекста.

ВОПРОС: {question}

КОНТЕКСТ (извлечённые фрагменты документации):
{context}

ОТВЕТ СИСТЕМЫ:
{answer}

Критерий — Faithfulness (Верность):
Содержит ли ответ ТОЛЬКО информацию из контекста? Нет ли в нём домыслов или фактов вне контекста?

Оцени по шкале от 0 до 10, где:
  10 — ответ полностью основан на контексте, нет ничего лишнего
   5 — частично домыслы, частично из контекста
   0 — ответ полностью придуман, контекст проигнорирован

Ответь ТОЛЬКО одним числом от 0 до 10.
"""

PROMPT_ANSWER_RELEVANCY = """
Ты — строгий судья качества AI-ответов. Оцени РЕЛЕВАНТНОСТЬ ответа вопросу.

ВОПРОС: {question}

ОТВЕТ СИСТЕМЫ:
{answer}

Критерий — Answer Relevancy (Релевантность ответа):
Отвечает ли ответ напрямую на поставленный вопрос? Нет ли в нём лишней воды?

Оцени по шкале от 0 до 10, где:
  10 — ответ точно и полно отвечает на вопрос
   5 — частично отвечает, много лишнего
   0 — ответ не по теме или «не знаю»

Ответь ТОЛЬКО одним числом от 0 до 10.
"""

PROMPT_CONTEXT_PRECISION = """
Ты — строгий судья качества AI-ответов. Оцени ТОЧНОСТЬ найденного контекста.

ВОПРОС: {question}

НАЙДЕННЫЕ ФРАГМЕНТЫ ДОКУМЕНТАЦИИ:
{context}

Критерий — Context Precision (Точность контекста):
Какая доля фрагментов действительно нужна для ответа на вопрос?

Оцени по шкале от 0 до 10, где:
  10 — все фрагменты релевантны вопросу
   5 — примерно половина фрагментов по теме
   0 — все фрагменты нерелевантны

Ответь ТОЛЬКО одним числом от 0 до 10.
"""

PROMPT_CONTEXT_RECALL = """
Ты — строгий судья качества AI-ответов. Оцени ПОЛНОТУ найденного контекста.

ВОПРОС: {question}

ЭТАЛОННЫЙ ОТВЕТ (если есть): {reference}

НАЙДЕННЫЕ ФРАГМЕНТЫ ДОКУМЕНТАЦИИ:
{context}

Критерий — Context Recall (Полнота контекста):
Содержит ли найденный контекст достаточно информации для полного ответа?

Оцени по шкале от 0 до 10, где:
  10 — контекст содержит всё необходимое для полного ответа
   5 — контекст содержит часть нужной информации
   0 — контекст совсем не помогает ответить

Ответь ТОЛЬКО одним числом от 0 до 10.
"""


# ══════════════════════════════════════════════════════════════════════════════
# Метрика F1 по токенам (без LLM)
# ══════════════════════════════════════════════════════════════════════════════

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text

def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = set(normalize_text(prediction).split())
    ref_tokens  = set(normalize_text(reference).split())
    if not ref_tokens:
        return 0.0
    common = pred_tokens & ref_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens) if pred_tokens else 0
    recall    = len(common) / len(ref_tokens)
    return round(2 * precision * recall / (precision + recall), 3)


# ══════════════════════════════════════════════════════════════════════════════
# Структура результата
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvalResult:
    question_id:       str
    question:          str
    koib_model:        Optional[str]
    category:          str
    answer:            str
    context_chunks:    int
    faithfulness:      float = 0.0
    answer_relevancy:  float = 0.0
    context_precision: float = 0.0
    context_recall:    float = 0.0
    token_f1:          float = 0.0
    has_reference:     bool  = False
    error:             Optional[str] = None
    latency_sec:       float = 0.0

    @property
    def rag_score(self) -> float:
        """Итоговый RAG-score: среднее 4 LLM-метрик."""
        return round(
            (self.faithfulness + self.answer_relevancy +
             self.context_precision + self.context_recall) / 4, 3
        )


# ══════════════════════════════════════════════════════════════════════════════
# Основной оценщик
# ══════════════════════════════════════════════════════════════════════════════

class RAGEvaluator:
    def __init__(self, credentials: str, top_k: int = 5, gigachat_model: str = "GigaChat"):
        print("Инициализация KoibQueryEngine...")
        self.engine = KoibQueryEngine()
        print("Инициализация GigaChat (судья)...")
        self.judge = GigaChatJudge(credentials, model=gigachat_model)
        self.top_k = top_k

    def _format_context(self, docs: list) -> str:
        parts = []
        for i, doc in enumerate(docs, 1):
            model_label = doc.get("model_display", "—")
            source = doc.get("source_name", "—")
            page   = doc.get("page", "—")
            text   = doc.get("text_preview", "")[:500]
            parts.append(f"[{i}] {model_label} | {source} | стр. {page}\n{text}")
        return "\n\n".join(parts)

    def evaluate_one(self, item: dict) -> EvalResult:
        q_id   = item["id"]
        q_text = item["question"]
        model  = item.get("koib_model")
        cat    = item.get("category", "—")
        ref    = item.get("reference_answer")

        print(f"\n{'─'*60}")
        print(f"[{q_id}] {q_text}")
        if model:
            print(f"  Модель КОИБ: {model}")

        result = EvalResult(
            question_id=q_id,
            question=q_text,
            koib_model=model,
            category=cat,
            answer="",
            context_chunks=0,
            has_reference=bool(ref),
        )

        # ── Шаг 1: Получить ответ от RAG ──────────────────────────────────
        t0 = time.time()
        try:
            answer = self.engine.ask_with_gigachat(
                question=q_text,
                koib_model=model,
                gigachat_credentials=self.judge.credentials,  # используем тот же токен
                top_k=self.top_k,
            )
            _, docs, _ = self.engine.ask(
                question=q_text,
                koib_model=model,
                top_k=self.top_k,
            )
        except Exception as e:
            result.error = str(e)
            print(f"  [!] Ошибка получения ответа: {e}")
            return result

        result.latency_sec   = round(time.time() - t0, 2)
        result.answer        = answer
        result.context_chunks = len(docs)
        context_str          = self._format_context(docs)

        print(f"  Ответ получен за {result.latency_sec}s | чанков: {len(docs)}")
        print(f"  Ответ (превью): {answer[:120]}...")

        # ── Шаг 2: LLM-оценки ─────────────────────────────────────────────
        def safe_score(prompt_template: str, **kwargs) -> float:
            try:
                return self.judge.score(prompt_template.format(**kwargs))
            except Exception as ex:
                print(f"  [!] Ошибка оценки: {ex}")
                return 0.0

        print("  Оцениваю Faithfulness...")
        result.faithfulness = safe_score(
            PROMPT_FAITHFULNESS,
            question=q_text, context=context_str, answer=answer
        )

        print("  Оцениваю Answer Relevancy...")
        result.answer_relevancy = safe_score(
            PROMPT_ANSWER_RELEVANCY,
            question=q_text, answer=answer
        )

        print("  Оцениваю Context Precision...")
        result.context_precision = safe_score(
            PROMPT_CONTEXT_PRECISION,
            question=q_text, context=context_str
        )

        print("  Оцениваю Context Recall...")
        result.context_recall = safe_score(
            PROMPT_CONTEXT_RECALL,
            question=q_text,
            reference=ref or "эталонный ответ не задан",
            context=context_str
        )

        # ── Шаг 3: Token F1 (если есть эталон) ───────────────────────────
        if ref:
            result.token_f1 = token_f1(answer, ref)

        print(f"  ✓ RAG Score: {result.rag_score:.3f}  "
              f"(F={result.faithfulness:.2f} AR={result.answer_relevancy:.2f} "
              f"CP={result.context_precision:.2f} CR={result.context_recall:.2f})")

        return result

    def evaluate_all(self, dataset: list[dict]) -> list[EvalResult]:
        results = []
        for item in dataset:
            res = self.evaluate_one(item)
            results.append(res)
            time.sleep(1)  # небольшая пауза между запросами
        return results


# ══════════════════════════════════════════════════════════════════════════════
# Отчёт
# ══════════════════════════════════════════════════════════════════════════════

def print_report(results: list[EvalResult]) -> None:
    ok = [r for r in results if r.error is None]
    if not ok:
        print("\n[!] Нет успешных результатов.")
        return

    def avg(attr):
        vals = [getattr(r, attr) for r in ok]
        return round(sum(vals) / len(vals), 3)

    print("\n" + "═" * 65)
    print("  ИТОГОВЫЙ ОТЧЁТ КАЧЕСТВА RAG-СИСТЕМЫ КОИБ")
    print("═" * 65)
    print(f"  Вопросов обработано : {len(ok)}/{len(results)}")
    print(f"  Среднее время ответа: {avg('latency_sec')} сек")
    print()
    print(f"  Faithfulness       (верность контексту)  : {avg('faithfulness'):.3f}")
    print(f"  Answer Relevancy   (релевантность ответа): {avg('answer_relevancy'):.3f}")
    print(f"  Context Precision  (точность контекста)  : {avg('context_precision'):.3f}")
    print(f"  Context Recall     (полнота контекста)   : {avg('context_recall'):.3f}")
    print(f"  {'─'*45}")
    total_rag = round(sum(r.rag_score for r in ok) / len(ok), 3)
    print(f"  ★ Итоговый RAG Score                     : {total_rag:.3f}")

    ref_results = [r for r in ok if r.has_reference]
    if ref_results:
        avg_f1 = round(sum(r.token_f1 for r in ref_results) / len(ref_results), 3)
        print(f"  Token F1 (по эталонам, n={len(ref_results)})           : {avg_f1:.3f}")

    print()
    print("  Детализация по вопросам:")
    print(f"  {'ID':<8} {'RAG':>6} {'F':>6} {'AR':>6} {'CP':>6} {'CR':>6}  Категория")
    print(f"  {'─'*8} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6}  {'─'*15}")
    for r in results:
        if r.error:
            print(f"  {r.question_id:<8} {'ОШИБКА':>6}  {r.error[:40]}")
        else:
            print(
                f"  {r.question_id:<8} {r.rag_score:>6.3f} "
                f"{r.faithfulness:>6.2f} {r.answer_relevancy:>6.2f} "
                f"{r.context_precision:>6.2f} {r.context_recall:>6.2f}  "
                f"{r.category}"
            )

    # Самые слабые места
    if len(ok) >= 2:
        metrics = {
            "Faithfulness":      avg("faithfulness"),
            "Answer Relevancy":  avg("answer_relevancy"),
            "Context Precision": avg("context_precision"),
            "Context Recall":    avg("context_recall"),
        }
        worst_metric = min(metrics, key=metrics.get)
        print()
        print(f"  ⚠ Слабейшая метрика: {worst_metric} = {metrics[worst_metric]:.3f}")
        print(_improvement_tip(worst_metric))

    print("═" * 65)


def _improvement_tip(metric: str) -> str:
    tips = {
        "Faithfulness": (
            "  → Ответ содержит домыслы. Ужесточите системный промпт:\n"
            '     «Отвечай ТОЛЬКО на основе предоставленного контекста.»'
        ),
        "Answer Relevancy": (
            "  → Ответы не по теме. Проверьте промпт-шаблон в query_engine.py\n"
            "     и убедитесь, что вопрос правильно передаётся в GigaChat."
        ),
        "Context Precision": (
            "  → Много нерелевантных чанков. Попробуйте:\n"
            "     • уменьшить top_k (сейчас извлекается слишком много)\n"
            "     • снизить CHUNK_SIZE для более точечных фрагментов"
        ),
        "Context Recall": (
            "  → Контекст неполный. Попробуйте:\n"
            "     • увеличить top_k\n"
            "     • увеличить CHUNK_OVERLAP для сохранения связности текста"
        ),
    }
    return tips.get(metric, "")


def save_report(results: list[EvalResult], path: str) -> None:
    data = {
        "summary": {
            "total":             len(results),
            "successful":        len([r for r in results if not r.error]),
            "avg_faithfulness":  round(sum(r.faithfulness for r in results if not r.error) / max(1, len([r for r in results if not r.error])), 3),
            "avg_answer_relevancy":  round(sum(r.answer_relevancy for r in results if not r.error) / max(1, len([r for r in results if not r.error])), 3),
            "avg_context_precision": round(sum(r.context_precision for r in results if not r.error) / max(1, len([r for r in results if not r.error])), 3),
            "avg_context_recall":    round(sum(r.context_recall for r in results if not r.error) / max(1, len([r for r in results if not r.error])), 3),
        },
        "results": [asdict(r) for r in results]
    }
    ok = [r for r in results if not r.error]
    if ok:
        data["summary"]["avg_rag_score"] = round(sum(r.rag_score for r in ok) / len(ok), 3)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n  Отчёт сохранён: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Оценка качества RAG-системы КОИБ через GigaChat (LLM-as-Judge)"
    )
    parser.add_argument(
        "--credentials", required=True,
        help="GigaChat credentials (base64 токен из личного кабинета Сбербанк)"
    )
    parser.add_argument(
        "--dataset", default="eval_dataset.json",
        help="Путь к JSON-файлу с вопросами (default: eval_dataset.json)"
    )
    parser.add_argument(
        "--output", default="eval_report.json",
        help="Путь для сохранения отчёта (default: eval_report.json)"
    )
    parser.add_argument(
        "--top_k", type=int, default=5,
        help="Сколько чанков извлекать из FAISS (default: 5)"
    )
    parser.add_argument(
        "--gigachat_model", default="GigaChat",
        help="Модель GigaChat: GigaChat / GigaChat-Pro / GigaChat-Max (default: GigaChat)"
    )
    parser.add_argument(
        "--ids", nargs="*",
        help="Оценить только конкретные вопросы по ID, напр. --ids q001 q003"
    )
    args = parser.parse_args()

    # Загрузка датасета
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"[ERROR] Файл датасета не найден: {dataset_path}")
        sys.exit(1)

    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    if args.ids:
        dataset = [q for q in dataset if q["id"] in args.ids]
        print(f"Выбрано вопросов: {len(dataset)} (фильтр: {args.ids})")

    print(f"\n{'═'*60}")
    print(f"  RAG Evaluation | КОИБ v4.1")
    print(f"  Датасет    : {dataset_path} ({len(dataset)} вопросов)")
    print(f"  top_k      : {args.top_k}")
    print(f"  LLM судья  : {args.gigachat_model}")
    print(f"{'═'*60}")

    evaluator = RAGEvaluator(
        credentials=args.credentials,
        top_k=args.top_k,
        gigachat_model=args.gigachat_model,
    )

    results = evaluator.evaluate_all(dataset)
    print_report(results)
    save_report(results, args.output)


if __name__ == "__main__":
    main()
