# -*- coding: utf-8 -*-
"""
KOIB RAG - Preprocessing Module (v4.1 — fixed)

Часть 1 системы RAG для обработки документации КОИБ:
извлечение текста, OCR, рисунков, определение моделей.

Исправления v4.1:
- [BUG FIX] Заменён устаревший Rect.expand() на _expand_rect()
  (совместимо со всеми версиями PyMuPDF)
- [BUG FIX] Поле "file" -> "source" согласовано с index_building.py
- [BUG FIX] Поле "model" добавляется в каждый text-блок
- [BUG FIX] FIGURES_DIR убрана из глобального состояния функций
- [BUG FIX] Добавлены поля block_type и headings в каждый блок
- [IMPROVEMENT] Надёжная обработка ошибок
"""

import os
import re
import json
import csv
import io
import time
import hashlib
from pathlib import Path
from collections import defaultdict
import logging
from typing import List, Dict, Any, Tuple, Optional

import fitz               # pymupdf
from docx import Document as DocxDocument
from PIL import Image
import pytesseract
import numpy as np
from tqdm import tqdm

try:
    from .utils import (
        clean_text, text_hash, normalize_model_key,
        detect_model_in_text, detect_model_from_filename,
        find_figure_caption, ensure_dirs, get_docs_dir, get_output_dir,
        KOIB_MODEL_PATTERNS, MODEL_DISPLAY_NAMES, FIGURE_CAPTION_PATTERNS
    )
except ImportError:
    from utils import (
        clean_text, text_hash, normalize_model_key,
        detect_model_in_text, detect_model_from_filename,
        find_figure_caption, ensure_dirs, get_docs_dir, get_output_dir,
        KOIB_MODEL_PATTERNS, MODEL_DISPLAY_NAMES, FIGURE_CAPTION_PATTERNS
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

_easyocr_reader = None

OCR_DPI = int(os.getenv("OCR_DPI", "300"))
OCR_MIN_TEXT_CHARS = int(os.getenv("OCR_MIN_TEXT_CHARS", "50"))
MIN_IMAGE_WIDTH = int(os.getenv("MIN_IMAGE_WIDTH", "80"))
MIN_IMAGE_HEIGHT = int(os.getenv("MIN_IMAGE_HEIGHT", "80"))
SCREENSHOT_AREA_THRESHOLD = float(os.getenv("SCREENSHOT_AREA_THRESHOLD", "0.80"))


def _expand_rect(rect: fitz.Rect, margin: float) -> fitz.Rect:
    """
    Расширить прямоугольник на margin со всех сторон.

    BUG FIX: Заменяет удалённый Rect.expand() / Rect.inflate().
    Работает со всеми версиями PyMuPDF.
    """
    return fitz.Rect(
        rect.x0 - margin,
        rect.y0 - margin,
        rect.x1 + margin,
        rect.y1 + margin,
    )


def get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        try:
            import easyocr
            _easyocr_reader = easyocr.Reader(['ru', 'en'], gpu=False, verbose=False)
        except ImportError:
            logging.warning("EasyOCR не установлен, fallback на Tesseract")
    return _easyocr_reader


def ocr_image(image_pil: Image.Image, lang: str = 'rus+eng') -> str:
    if image_pil is None:
        return ""
    text_tess = ""
    text_easy = ""

    try:
        text_tess = clean_text(
            pytesseract.image_to_string(image_pil, lang=lang, config='--psm 6')
        )
        if len(text_tess) >= 30:
            return text_tess
    except Exception as exc:
        logging.debug(f"Tesseract error: {exc}")
        text_tess = ""

    try:
        reader = get_easyocr_reader()
        if reader is not None:
            results = reader.readtext(np.array(image_pil), paragraph=True, detail=0)
            text_easy = clean_text('\n'.join(results))
            if len(text_easy) >= 20:
                return text_easy
    except Exception as exc:
        logging.debug(f"EasyOCR error: {exc}")
        text_easy = ""

    return text_tess if len(text_tess) >= len(text_easy) else text_easy


def detect_scanned_page(page: fitz.Page, min_text_chars: int = 50) -> bool:
    try:
        text = page.get_text("text").strip()
        if len(text) >= min_text_chars:
            return False
        images = page.get_images(full=True)
        if not images:
            return len(text) < min_text_chars
        page_area = page.rect.width * page.rect.height
        for img_info in images:
            try:
                xref = img_info[0]
                base_image = page.parent.extract_image(xref)
                if not base_image:
                    continue
                img = Image.open(io.BytesIO(base_image["image"]))
                if img.width * img.height / page_area > SCREENSHOT_AREA_THRESHOLD:
                    return True
            except Exception:
                continue
        return True
    except Exception as exc:
        logging.warning(f"detect_scanned_page: {exc}")
        return False


def _extract_headings_from_text(text: str) -> List[str]:
    patterns = [
        re.compile(r'^(\d+(?:\.\d+)*)\s+([А-ЯЁA-Z][^\n]{3,80})$', re.MULTILINE),
        re.compile(r'^([А-ЯЁ][А-ЯЁ\s]{4,60})$', re.MULTILINE),
    ]
    headings = []
    seen: set = set()
    for pat in patterns:
        for m in pat.finditer(text):
            h = m.group(0).strip()
            if h not in seen and len(h) > 4:
                headings.append(h)
                seen.add(h)
            if len(headings) >= 5:
                break
    return headings[:5]


def extract_text_from_pdf(
    pdf_path: Path,
    figures_dir: Path,
    model: str = "unknown",
) -> Tuple[List[Dict], List[Dict], int]:
    """Извлечь текст и рисунки из PDF. Возвращает (text_blocks, figures, ocr_count)."""
    text_blocks: List[Dict] = []
    figures: List[Dict] = []
    ocr_count = 0

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        logging.error(f"Не удалось открыть {pdf_path}: {exc}")
        return text_blocks, figures, ocr_count

    for page_num, page in enumerate(doc):
        try:
            text = page.get_text("text").strip()
            is_scanned = detect_scanned_page(page)

            if is_scanned:
                matrix = fitz.Matrix(OCR_DPI / 72, OCR_DPI / 72)
                pix = page.get_pixmap(matrix=matrix)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_text = ocr_image(img)
                if len(ocr_text) >= OCR_MIN_TEXT_CHARS:
                    text = ocr_text
                    ocr_count += 1

            if text:
                text_blocks.append({
                    "source": str(pdf_path),      # согласовано с index_building.py
                    "file": str(pdf_path),         # alias
                    "page": page_num + 1,
                    "text": text,
                    "caption": find_figure_caption(text),
                    "headings": _extract_headings_from_text(text),
                    "block_type": "ocr_text" if is_scanned else "text",
                    "model": model,
                    "hash": text_hash(text),
                })

            for img_info in page.get_images(full=True):
                try:
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    if not base_image:
                        continue
                    img_bytes = base_image["image"]
                    img = Image.open(io.BytesIO(img_bytes))
                    if img.width < MIN_IMAGE_WIDTH or img.height < MIN_IMAGE_HEIGHT:
                        continue

                    img_hash = hashlib.md5(img_bytes).hexdigest()[:12]
                    ext = base_image.get('ext', 'png')
                    img_fname = f"fig_{pdf_path.stem}_p{page_num+1}_{img_hash}.{ext}"
                    img_path = figures_dir / img_fname
                    try:
                        img.save(img_path)
                    except Exception:
                        img_fname = f"fig_{pdf_path.stem}_p{page_num+1}_{img_hash}.png"
                        img_path = figures_dir / img_fname
                        img.convert("RGB").save(img_path)

                    # ── BUG FIX: вместо rect.expand(50,50,50,50) ──────────
                    raw_rect = fitz.Rect(img_info[1:5])
                    clip_rect = _expand_rect(raw_rect, 50)
                    # ─────────────────────────────────────────────────────
                    nearby_text = page.get_text("text", clip=clip_rect)

                    figures.append({
                        "source": str(pdf_path),
                        "file": str(pdf_path),
                        "page": page_num + 1,
                        "image_path": str(img_path),
                        "caption": find_figure_caption(nearby_text),
                        "surrounding_text": clean_text(nearby_text)[:300],
                        "width": img.width,
                        "height": img.height,
                        "model": model,
                    })
                except Exception as exc:
                    logging.debug(f"Пропуск изображения стр.{page_num+1}: {exc}")
                    continue

        except Exception as exc:
            logging.warning(f"Ошибка стр.{page_num+1} в {pdf_path.name}: {exc}")
            continue

    doc.close()
    return text_blocks, figures, ocr_count


def extract_text_from_docx(
    docx_path: Path,
    figures_dir: Path,
    model: str = "unknown",
) -> Tuple[List[Dict], List[Dict]]:
    """Извлечь текст и рисунки из DOCX. Возвращает (text_blocks, figures)."""
    text_blocks: List[Dict] = []
    figures: List[Dict] = []

    try:
        doc = DocxDocument(docx_path)

        full_text_parts: List[str] = []
        headings: List[str] = []
        for para in doc.paragraphs:
            stripped = para.text.strip()
            if not stripped:
                continue
            full_text_parts.append(stripped)
            if para.style and para.style.name and 'Heading' in para.style.name:
                headings.append(stripped)

        if full_text_parts:
            text = '\n'.join(full_text_parts)
            text_blocks.append({
                "source": str(docx_path),
                "file": str(docx_path),
                "page": 0,
                "text": text,
                "caption": find_figure_caption(text),
                "headings": headings[:5],
                "block_type": "text",
                "model": model,
                "hash": text_hash(text),
            })

        for rel in doc.part.rels.values():
            if "image" not in rel.target_ref:
                continue
            try:
                img_bytes = rel.target_part.blob
                img = Image.open(io.BytesIO(img_bytes))
                if img.width < MIN_IMAGE_WIDTH or img.height < MIN_IMAGE_HEIGHT:
                    continue
                img_hash = hashlib.md5(img_bytes).hexdigest()[:12]
                ext = rel.target_ref.rsplit('.', 1)[-1] if '.' in rel.target_ref else 'png'
                img_fname = f"fig_{docx_path.stem}_{img_hash}.{ext}"
                img_path = figures_dir / img_fname
                try:
                    img.save(img_path)
                except Exception:
                    img_fname = f"fig_{docx_path.stem}_{img_hash}.png"
                    img_path = figures_dir / img_fname
                    img.convert("RGB").save(img_path)
                figures.append({
                    "source": str(docx_path),
                    "file": str(docx_path),
                    "page": 0,
                    "image_path": str(img_path),
                    "caption": "",
                    "surrounding_text": "",
                    "width": img.width,
                    "height": img.height,
                    "model": model,
                })
            except Exception as exc:
                logging.debug(f"Пропуск изображения {docx_path.name}: {exc}")
                continue

    except Exception as exc:
        logging.error(f"Ошибка обработки DOCX {docx_path}: {exc}")

    return text_blocks, figures


def generate_source_models(docs_dir: Path, metadata_dir: Path) -> Dict[str, str]:
    source_models: Dict[str, str] = {}
    for fp in docs_dir.glob("*"):
        if fp.suffix.lower() in ('.pdf', '.docx'):
            source_models[fp.name] = detect_model_from_filename(fp.name)
    ensure_dirs(metadata_dir)
    sm_path = metadata_dir / "source_models.json"
    with open(sm_path, 'w', encoding='utf-8') as f:
        json.dump(source_models, f, ensure_ascii=False, indent=2)
    print(f"✅ source_models.json: {len(source_models)} файлов")
    return source_models


class KoibPreprocessingPipeline:
    """Пайплайн предобработки документов КОИБ."""

    def __init__(
        self,
        docs_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        source_models: Optional[Dict[str, str]] = None,
    ):
        self.docs_dir = docs_dir or get_docs_dir()
        self.output_dir = output_dir or get_output_dir()
        self.classified_dir = self.output_dir / "classified"
        self.ocr_results_dir = self.output_dir / "ocr_results"
        self.figures_dir = self.output_dir / "figures"
        self.metadata_dir = self.output_dir / "metadata"
        self.logs_dir = self.output_dir / "logs"
        ensure_dirs(
            self.output_dir, self.classified_dir, self.ocr_results_dir,
            self.figures_dir, self.metadata_dir, self.logs_dir,
        )
        self.source_models = source_models or {}
        self.text_blocks: List[Dict] = []
        self.figures_index: List[Dict] = []
        self.processing_log: List[Dict] = []
        self.total_files = 0
        self.total_ocr_pages = 0

    def process_all(self) -> List[Dict]:
        print(f"\n📂 Документы: {self.docs_dir}")
        files = list(self.docs_dir.glob("*.pdf")) + list(self.docs_dir.glob("*.docx"))
        self.total_files = len(files)
        if not files:
            print("⚠️  Файлы не найдены!")
            return []
        for fp in tqdm(files, desc="Обработка файлов"):
            self._process_file(fp)
        return self.text_blocks

    def save_artifacts(self) -> None:
        print("\n--- Сохранение артефактов ---")
        blocks_path = self.metadata_dir / "text_blocks.json"
        with open(blocks_path, 'w', encoding='utf-8') as f:
            json.dump(self.text_blocks, f, ensure_ascii=False, indent=2)
        print(f"  ✅ text_blocks.json: {len(self.text_blocks)} блоков")

        figures_path = self.metadata_dir / "figures_index.json"
        with open(figures_path, 'w', encoding='utf-8') as f:
            json.dump(self.figures_index, f, ensure_ascii=False, indent=2)
        print(f"  ✅ figures_index.json: {len(self.figures_index)} рисунков")

        log_path = self.logs_dir / "processing_log.json"
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.processing_log, f, ensure_ascii=False, indent=2)
        print(f"  ✅ processing_log.json")

        csv_path = self.classified_dir / "classification_report.csv"
        with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.writer(f)
            w.writerow(["Файл", "Тип", "Модель", "Блоков", "Рисунков", "OCR стр.", "Время(с)"])
            for e in self.processing_log:
                w.writerow([e["file"], e["type"], e["model"],
                            e["text_blocks"], e["figures"], e["ocr_pages"], e["time_sec"]])
        print(f"  ✅ classification_report.csv")

    def print_summary(self) -> None:
        by_type: Dict[str, int] = defaultdict(int)
        by_model: Dict[str, int] = defaultdict(int)
        for e in self.processing_log:
            by_type[e["type"]] += 1
            by_model[e["model"]] += 1
        print("\n" + "=" * 70)
        print("📊 ИТОГИ PREPROCESSING")
        print("=" * 70)
        print(f"  Всего файлов:     {self.total_files}")
        print(f"  PDF / DOCX:       {by_type.get('pdf', 0)} / {by_type.get('docx', 0)}")
        print(f"  По моделям:       {dict(by_model)}")
        print(f"  Текст. блоков:    {len(self.text_blocks)}")
        print(f"  Рисунков:         {len(self.figures_index)}")
        print(f"  OCR страниц:      {self.total_ocr_pages}")
        print("=" * 70)

    def _process_file(self, file_path: Path) -> None:
        start = time.time()
        filename = file_path.name
        suffix = file_path.suffix.lower()
        model = self.source_models.get(filename) or detect_model_from_filename(filename)

        try:
            if suffix == '.pdf':
                text_blocks, figures, ocr_count = extract_text_from_pdf(
                    file_path, self.figures_dir, model
                )
                file_type = 'pdf'
            elif suffix == '.docx':
                text_blocks, figures = extract_text_from_docx(
                    file_path, self.figures_dir, model
                )
                ocr_count = 0
                file_type = 'docx'
            else:
                return
        except Exception as exc:
            logging.error(f"Критическая ошибка {filename}: {exc}")
            text_blocks, figures, ocr_count = [], [], 0
            file_type = suffix.lstrip('.')

        self.text_blocks.extend(text_blocks)
        self.figures_index.extend(figures)
        self.total_ocr_pages += ocr_count
        self.processing_log.append({
            "file": filename,
            "type": file_type,
            "model": model,
            "text_blocks": len(text_blocks),
            "figures": len(figures),
            "ocr_pages": ocr_count,
            "time_sec": round(time.time() - start, 2),
        })


def main() -> None:
    print("╔" + "═" * 68 + "╗")
    print("║" + "  KOIB RAG v4.1 – ЧАСТЬ 1: PREPROCESSING".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    t0 = time.time()

    docs_dir = get_docs_dir()
    output_dir = get_output_dir()
    metadata_dir = output_dir / "metadata"

    sm_path = metadata_dir / "source_models.json"
    if sm_path.exists():
        with open(sm_path, 'r', encoding='utf-8') as f:
            source_models = json.load(f)
        print(f"📂 Загружен source_models.json ({len(source_models)} записей)")
    else:
        source_models = generate_source_models(docs_dir, metadata_dir)

    pipeline = KoibPreprocessingPipeline(docs_dir, output_dir, source_models)
    pipeline.process_all()
    pipeline.save_artifacts()
    pipeline.print_summary()

    elapsed = time.time() - t0
    print(f"\n⏱️  Время: {elapsed:.1f}с ({elapsed/60:.1f} мин)")
    print("✅ ЧАСТЬ 1 ЗАВЕРШЕНА. Запустите ЧАСТЬ 2: python -m src.index_building")


if __name__ == "__main__":
    main()
