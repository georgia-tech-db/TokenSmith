#!/usr/bin/env python3
"""TokenSmith local Python worker.

The worker handles document processing and retrieval over newline-delimited JSON.
Model execution belongs to Ollama or a remote provider.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from datetime import datetime

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    from tokensmith_store import (
        append_material_chunks,
        backfill_collection_vocabularies,
        begin_material_index,
        collection_vocabulary_words,
        delete_material,
        dump_index,
        embedded_chunk_signatures,
        embedding_models_by_collection_ids,
        enabled_material_ids_for_requests,
        fetch_sources,
        expand_source_units,
        find_material_id_by_import_path,
        has_chunks,
        init_db,
        keyword_search,
        save_collection_vocabulary,
        keyword_terms_for_query,
        list_materials,
        set_material_active,
        source_document_for_source,
        starter_source_rows,
        update_material_index_state,
        vector_search,
    )
except ImportError:  # pragma: no cover - allows direct package imports in tests
    from python_engine.tokensmith_store import (
        append_material_chunks,
        backfill_collection_vocabularies,
        begin_material_index,
        collection_vocabulary_words,
        delete_material,
        dump_index,
        embedded_chunk_signatures,
        embedding_models_by_collection_ids,
        enabled_material_ids_for_requests,
        fetch_sources,
        expand_source_units,
        find_material_id_by_import_path,
        has_chunks,
        init_db,
        keyword_search,
        save_collection_vocabulary,
        keyword_terms_for_query,
        list_materials,
        set_material_active,
        source_document_for_source,
        starter_source_rows,
        update_material_index_state,
        vector_search,
    )

try:
    from tokensmith_vocab import DEFAULT_TYPO_CUTOFF, build_vocabulary, correct_query
except ImportError:  # pragma: no cover - allows direct package imports in tests
    from python_engine.tokensmith_vocab import DEFAULT_TYPO_CUTOFF, build_vocabulary, correct_query

try:
    from tokensmith_cleaning import (
        CLEANING_PROFILE_VERSION,
        DEFAULT_CLEANING_PROFILE_ID,
        clean_pages,
        cleaning_profiles,
        cleaning_rules,
        resolve_cleaning_rule_ids,
        resolve_cleaning_profile,
        section_header_from_line,
    )
except ImportError:  # pragma: no cover - allows direct package imports in tests
    from python_engine.tokensmith_cleaning import (
        CLEANING_PROFILE_VERSION,
        DEFAULT_CLEANING_PROFILE_ID,
        clean_pages,
        cleaning_profiles,
        cleaning_rules,
        resolve_cleaning_rule_ids,
        resolve_cleaning_profile,
        section_header_from_line,
    )

try:
    import pypdfium2 as pdfium  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    pdfium = None  # type: ignore

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt"}
MAX_FOLDER_FILES = 120
DEFAULT_CHUNK_SIZE = 1000
CHUNK_OVERLAP = 0
TOKENSMITH_CHUNK_MARKER_RE = re.compile(r"<!--\s*tokensmith:chunk\b(?P<attrs>.*?)-->", re.IGNORECASE | re.DOTALL)
TOKENSMITH_CHUNK_ATTR_RE = re.compile(
    r"""([A-Za-z_][\w:-]*)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'>]+))"""
)
PDF_THUMBNAIL_CACHE_DIR = "tokensmith-pdf-thumbnails"
PDF_THUMBNAIL_SCALE = 0.2
PDF_THUMBNAIL_MAX_SIZE = (180, 240)
REMOTE_EMBEDDING_TEXT_LIMIT = 8000
OLLAMA_EMBEDDING_TEXT_LIMIT = 8000
STOP_WORDS = {
    "a",
    "always",
    "also",
    "an",
    "and",
    "are",
    "as",
    "at",
    "bad",
    "be",
    "best",
    "better",
    "by",
    "does",
    "for",
    "from",
    "good",
    "how",
    "i",
    "in",
    "is",
    "it",
    "just",
    "mean",
    "need",
    "of",
    "on",
    "one",
    "or",
    "prefer",
    "preferable",
    "preferred",
    "prefers",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "worse",
    "worst",
}


VISIBLE_LOG_EVENTS = {
    "query_typo_correction",
    "vocab_built",
    "vocab_backfilled",
    "follow_up_suggestions",
    "chat_request_context",
    "chat_response_context",
    "chat_runtime_context_budget",
    "question_suggestion_request_context",
    "question_suggestion_runtime_context_budget",
    "library_search_request",
    "library_search_result",
}

class EngineError(Exception):
    pass


try:
    INDEX_CHECKPOINT_BATCH_SIZE = max(1, int(os.environ.get("TOKENSMITH_INDEX_CHECKPOINT_BATCH_SIZE", 50)))
except ValueError:
    INDEX_CHECKPOINT_BATCH_SIZE = 50


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


try:
    VOCAB_MIN_COUNT = max(1, int(os.environ.get("TOKENSMITH_VOCAB_MIN_COUNT", 4)))
except ValueError:
    VOCAB_MIN_COUNT = 4
# Developer knob for tuning the cutoff against a real collection.
TYPO_CUTOFF = min(1.0, max(0.0, env_float("TOKENSMITH_TYPO_CUTOFF", DEFAULT_TYPO_CUTOFF)))


def log_event(event: str, **details: Any) -> None:
    lower_event = event.lower()
    if event not in VISIBLE_LOG_EVENTS and not any(token in lower_event for token in ("error", "failed", "failure", "timeout", "stderr", "force_kill")):
        return

    log_file = os.environ.get("TOKENSMITH_LOG_FILE")
    if not log_file:
        return
    payload = {
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "event": event,
        **details,
    }
    try:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        with Path(log_file).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def now_ms() -> int:
    return int(time.time() * 1000)


def create_id(prefix: str) -> str:
    return f"{prefix}-{now_ms()}-{os.urandom(3).hex()}"


def normalize_text(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]{2,}", " ", text.replace("\x00", ""))).strip()


def count_words(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*", text))


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+(?:['-][a-z0-9]+)?", text.lower())
    return [token for token in tokens if len(token) > 1 and token not in STOP_WORDS]


def normalize_vector(values: Iterable[float]) -> List[float]:
    vector = [float(value) for value in values]
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 1e-12:
        return [0.0 for _value in vector]
    return [value / norm for value in vector]


def is_remote_embedding_spec(model: Dict[str, Any]) -> bool:
    return (
        model.get("engine") == "remote"
        and model.get("role") in {"embedder", "both"}
        and bool(str(model.get("baseUrl") or "").strip())
        and bool(str(model.get("remoteModelName") or "").strip())
    )


def is_ollama_embedding_spec(model: Dict[str, Any]) -> bool:
    return (
        model.get("engine") == "ollama"
        and model.get("role") in {"embedder", "both"}
        and bool(str(model.get("ollamaModelName") or "").strip())
    )


def normalize_remote_base_url(base_url: str) -> str:
    return base_url.strip().rstrip("/")


def normalize_ollama_base_url(base_url: str) -> str:
    normalized = (base_url or "http://127.0.0.1:11434").strip().rstrip("/")
    if normalized.endswith("/api"):
        normalized = normalized[:-4].rstrip("/")
    if normalized in {"", "http://localhost:11434"}:
        return "http://127.0.0.1:11434"
    return normalized


def ollama_embedding_model_key(model: Dict[str, Any]) -> str:
    base_url = normalize_ollama_base_url(str(model.get("ollamaBaseUrl") or model.get("baseUrl") or ""))
    model_name = str(model.get("ollamaModelName") or "").strip()
    safe_name = re.sub(r"[^a-z0-9_.:-]+", "-", model_name.lower()).strip("-") or "model"
    if base_url == "http://127.0.0.1:11434":
        return f"ollama:{safe_name}"
    digest = hashlib.sha256(f"{base_url}\0{model_name}".encode("utf-8")).hexdigest()[:12]
    return f"ollama:{safe_name}:{digest}"


def remote_embedding_model_key(model: Dict[str, Any]) -> str:
    base_url = normalize_remote_base_url(str(model.get("baseUrl") or ""))
    model_name = str(model.get("remoteModelName") or "").strip()
    digest = hashlib.sha256(f"{base_url}\0{model_name}".encode("utf-8")).hexdigest()[:24]
    provider = re.sub(r"[^a-z0-9]+", "-", str(model.get("providerId") or "remote").lower()).strip("-") or "remote"
    return f"remote-openai:{provider}:{digest}"


def embedding_model_key_from_spec(model: Dict[str, Any]) -> str:
    if is_remote_embedding_spec(model):
        return remote_embedding_model_key(model)
    if is_ollama_embedding_spec(model):
        return ollama_embedding_model_key(model)
    return ""


def format_bytes(size: int) -> str:
    if size < 1024 * 1024:
        return f"{max(1, round(size / 1024))} KB"
    if size < 1024 * 1024 * 1024:
        return f"{size / 1024 / 1024:.1f} MB"
    return f"{size / 1024 / 1024 / 1024:.2f} GB"


def format_number(value: int) -> str:
    return f"{value:,}"


def load_index(user_data_path: str) -> Dict[str, Any]:
    return dump_index(user_data_path)


def extract_text_plain(path: Path) -> Tuple[str, Optional[int]]:
    return normalize_text(path.read_text(encoding="utf-8", errors="ignore")), None


def extract_pdf_raw_pages_pdfium(path: Path, page_limit: Optional[int] = None, include_layout_hints: bool = False) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    if pdfium is None:
        raise EngineError("pypdfium2 is not installed in the Python runtime.")

    log_event("pdfium_extract_start", path=str(path))
    pages: List[Dict[str, Any]] = []
    pdf = pdfium.PdfDocument(str(path))
    try:
        page_count = len(pdf) or None
        pages_to_read = page_count or 0
        if page_limit is not None:
            pages_to_read = min(pages_to_read, max(0, int(page_limit)))
        for index in range(pages_to_read):
            page_number = index + 1
            page = pdf[index]
            text_page = None
            try:
                text_page = page.get_textpage()
                page_text = text_page.get_text_range() or ""
                image_count = sum(1 for _ in page.get_objects(filter=[pdfium.raw.FPDF_PAGEOBJ_IMAGE])) if include_layout_hints else 0
            finally:
                if text_page is not None:
                    text_page.close()
                page.close()
            if page_text or include_layout_hints:
                pages.append({"page": page_number, "text": page_text, **({"imageCount": image_count} if include_layout_hints else {})})
            log_event("pdfium_page_extracted", path=str(path), page=page_number, chars=len(page_text))
    finally:
        pdf.close()

    log_event("pdfium_extract_success", path=str(path), pageCount=page_count, chars=sum(len(page["text"]) for page in pages))
    return pages, page_count


def extract_pdf_pages_pdfium(
    path: Path,
    cleaning_profile_id: str = DEFAULT_CLEANING_PROFILE_ID,
    cleaning_rule_ids: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    raw_pages, page_count = extract_pdf_raw_pages_pdfium(path)
    profile = resolve_cleaning_profile(cleaning_profile_id)
    pages = clean_pages(raw_pages, profile["id"], cleaning_rule_ids)
    log_event(
        "pdf_cleaning_complete",
        path=str(path),
        cleaningProfile=profile["id"],
        cleanedPages=len(pages),
        chars=sum(len(page["text"]) for page in pages),
    )
    return pages, page_count


def extract_pdf_pdfium(
    path: Path,
    cleaning_profile_id: str = DEFAULT_CLEANING_PROFILE_ID,
    cleaning_rule_ids: Optional[List[str]] = None,
) -> Tuple[str, Optional[int]]:
    pages, page_count = extract_pdf_pages_pdfium(path, cleaning_profile_id, cleaning_rule_ids)
    text = normalize_text("\n\n".join(page["text"] for page in pages))
    return text, page_count


def extract_text_pdf(
    path: Path,
    cleaning_profile_id: str = DEFAULT_CLEANING_PROFILE_ID,
    cleaning_rule_ids: Optional[List[str]] = None,
) -> Tuple[str, Optional[int]]:
    return extract_pdf_pdfium(path, cleaning_profile_id, cleaning_rule_ids)


def thumbnail_cache_dir(user_data_path: str, pdf_path: Path) -> Path:
    digest = hashlib.sha256(str(pdf_path).encode("utf-8")).hexdigest()[:20]
    return Path(user_data_path) / PDF_THUMBNAIL_CACHE_DIR / digest


def render_pdf_thumbnail(page: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bitmap = page.render(scale=PDF_THUMBNAIL_SCALE)
    try:
        image = bitmap.to_pil()
        image.thumbnail(PDF_THUMBNAIL_MAX_SIZE)
        if getattr(image, "mode", "RGB") != "RGB":
            image = image.convert("RGB")
        image.save(output_path, format="PNG", optimize=True)
    finally:
        close = getattr(bitmap, "close", None)
        if callable(close):
            close()


def generate_pdf_thumbnails(user_data_path: str, path: Path, page_count: Optional[int]) -> List[Dict[str, Any]]:
    if path.suffix.lower() != ".pdf" or pdfium is None:
        return []

    thumbnails: List[Dict[str, Any]] = []
    pdf = pdfium.PdfDocument(str(path))
    try:
        total_pages = min(len(pdf), page_count or len(pdf))
        cache_dir = thumbnail_cache_dir(user_data_path, path)
        for index in range(total_pages):
            page_number = index + 1
            output_path = cache_dir / f"page-{page_number:04d}.png"
            page = pdf[index]
            try:
                render_pdf_thumbnail(page, output_path)
                thumbnails.append({"page": page_number, "path": str(output_path)})
                log_event("pdf_thumbnail_rendered", path=str(path), page=page_number, thumbnailPath=str(output_path))
            except Exception as error:
                log_event("pdf_thumbnail_failed", path=str(path), page=page_number, error=str(error))
            finally:
                page.close()
    finally:
        pdf.close()

    return thumbnails


def extract_text(
    path: Path,
    cleaning_profile_id: str = DEFAULT_CLEANING_PROFILE_ID,
    cleaning_rule_ids: Optional[List[str]] = None,
) -> Tuple[str, Optional[int]]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".markdown"}:
        return extract_text_plain(path)
    if suffix == ".pdf":
        return extract_text_pdf(path, cleaning_profile_id, cleaning_rule_ids)
    raise EngineError("Unsupported course material type.")


def split_long_segment(segment: str, chunk_size: int) -> List[str]:
    sentences = re.findall(r"[^.!?]+[.!?]+[\"')\]]?|[^.!?]+$", segment) or [segment]
    chunks: List[str] = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > chunk_size:
            if current.strip():
                chunks.append(current.strip())
                current = ""
            chunks.extend(sentence[index : index + chunk_size].strip() for index in range(0, len(sentence), chunk_size))
            continue
        next_value = f"{current} {sentence}".strip()
        if current and len(next_value) > chunk_size:
            chunks.append(current.strip())
            current = sentence
        else:
            current = next_value
    if current.strip():
        chunks.append(current.strip())
    return [chunk for chunk in chunks if chunk]


def estimate_pages(start: int, end: int, total: int, page_count: Optional[int]) -> Dict[str, int]:
    if not page_count or total <= 0:
        return {}
    page_start = min(page_count, max(1, math.floor((start / total) * page_count) + 1))
    page_end = min(page_count, max(page_start, math.floor((end / total) * page_count) + 1))
    return {"pageStart": page_start, "pageEnd": page_end}


def section_headers_in_text(text: str, cleaning_rule_ids: Optional[List[str]] = None) -> List[str]:
    headers: List[str] = []
    for line in text.splitlines():
        header = section_header_from_line(line, cleaning_rule_ids)
        if header and (not headers or headers[-1] != header):
            headers.append(header)
    return headers


def has_tokensmith_chunk_markers(text: str) -> bool:
    return bool(TOKENSMITH_CHUNK_MARKER_RE.search(text))


def parse_tokensmith_chunk_attrs(raw_attrs: str) -> Dict[str, str]:
    attrs: Dict[str, str] = {}
    for match in TOKENSMITH_CHUNK_ATTR_RE.finditer(raw_attrs):
        value = next((group for group in match.groups()[1:] if group is not None), "")
        attrs[match.group(1)] = value.strip()
    return attrs


def line_number_at_offset(text: str, offset: int) -> int:
    return text.count("\n", 0, max(0, offset)) + 1


def chunk_tokensmith_markdown(text: str, cleaning_rule_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    markers = list(TOKENSMITH_CHUNK_MARKER_RE.finditer(text))
    if not markers:
        return []

    chunks: List[Dict[str, Any]] = []
    for index, marker in enumerate(markers):
        next_start = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        raw_content = text[marker.end():next_start]
        content = normalize_text(raw_content)
        if not content:
            continue

        leading = len(raw_content) - len(raw_content.lstrip())
        trailing = len(raw_content.rstrip())
        start_offset = marker.end() + leading
        end_offset = marker.end() + trailing
        attrs = parse_tokensmith_chunk_attrs(marker.group("attrs") or "")
        section_header = attrs.get("section")
        if not section_header:
            detected_headers = section_headers_in_text(content, cleaning_rule_ids)
            section_header = detected_headers[-1] if detected_headers else None

        chunk = {
            "text": content,
            "wordCount": count_words(content),
            "startOffset": start_offset,
            "endOffset": end_offset,
            "lineFrom": line_number_at_offset(text, start_offset),
            "lineTo": line_number_at_offset(text, max(start_offset, end_offset - 1)),
            "chunkSize": len(content),
            "tokensmithChunkIndex": index + 1,
        }
        if section_header:
            chunk["sectionHeader"] = section_header
        if attrs.get("id"):
            chunk["tokensmithChunkId"] = attrs["id"]
        if attrs.get("chapter"):
            chunk["tokensmithChapter"] = attrs["chapter"]
        if attrs.get("kind"):
            chunk["tokensmithChunkKind"] = attrs["kind"]
        chunks.append(chunk)
    return chunks


def chunk_text(
    text: str,
    page_count: Optional[int],
    cleaning_rule_ids: Optional[List[str]] = None,
    inherited_section_header: Optional[str] = None,
) -> List[Dict[str, Any]]:
    chunk_size = DEFAULT_CHUNK_SIZE
    clean = normalize_text(text)
    if not clean:
        return []
    segments: List[str] = []
    for segment in re.split(r"\n{2,}", clean):
        segment = segment.strip()
        if not segment:
            continue
        segments.extend(split_long_segment(segment, chunk_size) if len(segment) > chunk_size else [segment])

    chunks: List[Dict[str, Any]] = []
    current = ""
    cursor = 0
    start_offset = 0
    current_section_header = inherited_section_header

    def flush() -> None:
        nonlocal current, cursor, start_offset, current_section_header
        text_to_store = current.strip()
        if not text_to_store:
            return
        location = clean.find(text_to_store[:80], start_offset)
        if location < 0:
            location = max(0, cursor)
        end_offset = min(len(clean), location + len(text_to_store))
        detected_headers = section_headers_in_text(text_to_store, cleaning_rule_ids)
        if detected_headers:
            current_section_header = detected_headers[-1]
        chunk = {
            "text": text_to_store,
            "wordCount": count_words(text_to_store),
            "startOffset": location,
            "endOffset": end_offset,
            "chunkSize": DEFAULT_CHUNK_SIZE,
            **estimate_pages(location, end_offset, len(clean), page_count),
        }
        if current_section_header:
            chunk["sectionHeader"] = current_section_header
        if chunk["wordCount"] > 4:
            chunks.append(chunk)
        overlap = text_to_store[-CHUNK_OVERLAP:] if CHUNK_OVERLAP else ""
        current = overlap
        start_offset = max(0, end_offset - len(overlap))
        cursor = end_offset

    for segment in segments:
        next_value = f"{current}\n\n{segment}".strip() if current else segment
        if current.strip() and len(next_value) > chunk_size:
            flush()
            current = segment
            start_offset = max(0, clean.find(segment[:80], cursor))
        else:
            if not current.strip():
                start_offset = max(0, clean.find(segment[:80], cursor))
            current = next_value
    flush()
    return chunks


def chunk_pdf_pages(pages: List[Dict[str, Any]], cleaning_rule_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    current_section_header: Optional[str] = None
    for page in pages:
        page_number = int(page["page"])
        page_chunks = chunk_text(page["text"], None, cleaning_rule_ids, current_section_header)
        for chunk in page_chunks:
            chunk["pageStart"] = page_number
            chunk["pageEnd"] = page_number
            if chunk.get("sectionHeader"):
                current_section_header = chunk["sectionHeader"]
            chunks.append(chunk)
    return chunks


def supported_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED_EXTENSIONS else []
    found: List[Path] = []
    for root, _dirs, files in os.walk(path):
        for name in files:
            if len(found) >= MAX_FOLDER_FILES:
                return found
            candidate = Path(root) / name
            if candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
                found.append(candidate)
    return found


def material_kind(path: Path) -> str:
    if path.is_dir():
        return "folder"
    return "pdf" if path.suffix.lower() == ".pdf" else "document"


def file_type_label(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "PDF"
    if suffix in {".md", ".markdown"}:
        return "Markdown"
    if suffix == ".txt":
        return "Text file"
    return "Document"


def indexed_chunks(
    material_id: str,
    material_title: str,
    document_id: str,
    document_title: str,
    path: Path,
    chunks: List[Dict[str, Any]],
    embedding_model: str,
    embed_text: Any,
    on_embedding_progress: Optional[Any] = None,
    existing_embedding_signatures: Optional[set[Tuple[str, str, Optional[int], Optional[int], Optional[int], Optional[int]]]] = None,
    on_chunk_indexed: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    indexed: List[Dict[str, Any]] = []
    for index, chunk in enumerate(chunks, start=1):
        text = chunk["text"]
        chunk_embedding_model = embedding_model
        signature = chunk_signature(path, chunk)
        if existing_embedding_signatures is not None and signature in existing_embedding_signatures:
            log_event(
                "chunk_embedding_skipped",
                path=str(path),
                chunkIndex=index,
                embeddingModel=chunk_embedding_model,
            )
            stored_chunk = {
                "id": create_id("chunk"),
                "materialId": material_id,
                "documentId": document_id,
                "materialTitle": material_title,
                "documentTitle": document_title,
                "path": str(path),
                "text": text,
                "wordCount": chunk["wordCount"],
                "pageStart": chunk.get("pageStart"),
                "pageEnd": chunk.get("pageEnd"),
                "lineFrom": chunk.get("lineFrom"),
                "lineTo": chunk.get("lineTo"),
                "chunkIndex": index,
                "chunkSize": chunk.get("chunkSize"),
                "sectionHeader": chunk.get("sectionHeader"),
                "tokensmithChunkId": chunk.get("tokensmithChunkId"),
                "tokensmithChapter": chunk.get("tokensmithChapter"),
                "tokensmithChunkKind": chunk.get("tokensmithChunkKind"),
                "parentId": chunk.get("parentId"),
                "part": chunk.get("part"),
                "parts": chunk.get("parts"),
                "embeddingModel": chunk_embedding_model,
            }
            indexed.append(stored_chunk)
            if on_chunk_indexed:
                on_chunk_indexed(stored_chunk)
            if on_embedding_progress:
                on_embedding_progress(index, len(chunks))
            continue

        log_event(
            "chunk_embedding_start",
            path=str(path),
            chunkIndex=index,
            chars=len(text),
            embeddingModel=embedding_model,
        )
        try:
            embedding = embed_text(text)
            log_event(
                "chunk_embedding_success",
                path=str(path),
                chunkIndex=index,
                embeddingModel=chunk_embedding_model,
            )
        except Exception as error:
            log_event("index_embedding_failed", path=str(path), chunkIndex=index, error=str(error))
            raise EngineError(f"The selected embedding model could not embed {path.name}.") from error

        stored_chunk = {
            "id": create_id("chunk"),
            "materialId": material_id,
            "documentId": document_id,
            "materialTitle": material_title,
            "documentTitle": document_title,
            "path": str(path),
            "text": text,
            "wordCount": chunk["wordCount"],
            "pageStart": chunk.get("pageStart"),
            "pageEnd": chunk.get("pageEnd"),
            "lineFrom": chunk.get("lineFrom"),
            "lineTo": chunk.get("lineTo"),
            "chunkIndex": index,
            "chunkSize": chunk.get("chunkSize"),
            "sectionHeader": chunk.get("sectionHeader"),
            "tokensmithChunkId": chunk.get("tokensmithChunkId"),
            "tokensmithChapter": chunk.get("tokensmithChapter"),
            "tokensmithChunkKind": chunk.get("tokensmithChunkKind"),
            "parentId": chunk.get("parentId"),
            "part": chunk.get("part"),
            "parts": chunk.get("parts"),
            "embeddingModel": chunk_embedding_model,
            "embedding": embedding,
        }
        indexed.append(stored_chunk)
        if on_chunk_indexed:
            on_chunk_indexed(stored_chunk)
        if on_embedding_progress:
            on_embedding_progress(index, len(chunks))
    return indexed


def chunk_signature(path: Path, chunk: Dict[str, Any]) -> Tuple[str, str, Optional[int], Optional[int], Optional[int], Optional[int]]:
    return (
        str(path),
        str(chunk.get("text") or ""),
        chunk.get("pageStart"),
        chunk.get("lineFrom"),
        chunk.get("lineTo"),
        chunk.get("chunkSize"),
    )


def prepare_index_file(
    material_id: str,
    path: Path,
    user_data_path: Optional[str] = None,
    cleaning_profile_id: str = DEFAULT_CLEANING_PROFILE_ID,
    cleaning_rule_ids: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    try:
        if path.suffix.lower() == ".pdf":
            pdf_pages, page_count = extract_pdf_pages_pdfium(path, cleaning_profile_id, cleaning_rule_ids)
            text = normalize_text("\n\n".join(page["text"] for page in pdf_pages))
            word_count = count_words(text)
            chunks = chunk_pdf_pages(pdf_pages, cleaning_rule_ids) if word_count >= 20 else []
        elif path.suffix.lower() in {".md", ".markdown"}:
            raw_text = path.read_text(encoding="utf-8", errors="ignore")
            page_count = None
            if has_tokensmith_chunk_markers(raw_text):
                chunks = chunk_tokensmith_markdown(raw_text, cleaning_rule_ids)
                text = normalize_text("\n\n".join(chunk["text"] for chunk in chunks))
                word_count = sum(int(chunk.get("wordCount") or 0) for chunk in chunks)
            else:
                text = normalize_text(raw_text)
                word_count = count_words(text)
                chunks = chunk_text(text, page_count, cleaning_rule_ids) if word_count >= 20 else []
        else:
            text, page_count = extract_text(path, cleaning_profile_id, cleaning_rule_ids)
            word_count = count_words(text)
            chunks = chunk_text(text, page_count, cleaning_rule_ids) if word_count >= 20 else []
        thumbnails = generate_pdf_thumbnails(user_data_path, path, page_count) if user_data_path else []
        log_event(
            "document_chunked",
            path=str(path),
            wordCount=word_count,
            pageCount=page_count,
            chunkCount=len(chunks),
            chunkSize=DEFAULT_CHUNK_SIZE,
            thumbnailCount=len(thumbnails),
        )
        document_id = create_id("document")
        status = "ready" if chunks else "needsReview"
        return (
            {
                "id": document_id,
                "materialId": material_id,
                "title": path.stem,
                "path": str(path),
                "kind": material_kind(path),
                "wordCount": word_count,
                "pageCount": page_count,
                "chunkCount": len(chunks),
                "thumbnails": thumbnails,
                "status": status,
                "error": None if status == "ready" else "No readable study text was found.",
            },
            chunks,
        )
    except Exception as error:
        return (
            {
                "id": create_id("document"),
                "materialId": material_id,
                "title": path.stem,
                "path": str(path),
                "kind": material_kind(path),
                "wordCount": 0,
                "chunkCount": 0,
                "status": "needsReview",
                "error": str(error) or "This file could not be read.",
            },
            [],
        )


def index_file(
    material_id: str,
    material_title: str,
    path: Path,
    embedding_model: str,
    embed_text: Any,
    user_data_path: Optional[str] = None,
    cleaning_profile_id: str = DEFAULT_CLEANING_PROFILE_ID,
    cleaning_rule_ids: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    document, chunks = prepare_index_file(
        material_id,
        path,
        user_data_path,
        cleaning_profile_id,
        cleaning_rule_ids,
    )
    stored_chunks = indexed_chunks(
        material_id,
        material_title,
        document["id"],
        path.stem,
        path,
        chunks,
        embedding_model,
        embed_text,
    )
    document["chunkCount"] = len(stored_chunks)
    document["status"] = "ready" if stored_chunks else "needsReview"
    document["error"] = None if stored_chunks else document.get("error") or "No readable study text was found."
    return document, stored_chunks


def truncate_preview_text(text: str, max_chars: int = 6000) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def preview_text_pages(
    path: Path,
    cleaning_profile_id: str,
    cleaning_rule_ids: Optional[List[str]] = None,
    page_limit: int = 2,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Optional[int]]:
    if path.suffix.lower() == ".pdf":
        raw_pages, page_count = extract_pdf_raw_pages_pdfium(path, page_limit=page_limit)
        profile = resolve_cleaning_profile(cleaning_profile_id)
        return raw_pages, clean_pages(raw_pages, profile["id"], cleaning_rule_ids), page_count

    text, page_count = extract_text(path, cleaning_profile_id, cleaning_rule_ids)
    raw_pages = [{"page": 1, "text": text}]
    profile = resolve_cleaning_profile(cleaning_profile_id)
    return raw_pages, clean_pages(raw_pages, profile["id"], cleaning_rule_ids), page_count


def preview_chunks_for_file(
    path: Path,
    pages: List[Dict[str, Any]],
    page_count: Optional[int],
    cleaning_rule_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    if not pages:
        return []

    if path.suffix.lower() == ".pdf":
        chunks = chunk_pdf_pages(pages, cleaning_rule_ids)
    else:
        raw_text = path.read_text(encoding="utf-8", errors="ignore")
        if path.suffix.lower() in {".md", ".markdown"} and has_tokensmith_chunk_markers(raw_text):
            chunks = chunk_tokensmith_markdown(raw_text, cleaning_rule_ids)
        else:
            text = normalize_text("\n\n".join(page["text"] for page in pages))
            chunks = chunk_text(text, page_count, cleaning_rule_ids)

    return [
        {
            "text": truncate_preview_text(chunk["text"], 1500),
            "wordCount": chunk.get("wordCount") or count_words(chunk.get("text", "")),
            "pageStart": chunk.get("pageStart"),
            "pageEnd": chunk.get("pageEnd"),
            "chunkSize": chunk.get("chunkSize"),
            "sectionHeader": chunk.get("sectionHeader"),
            "tokensmithChunkId": chunk.get("tokensmithChunkId"),
            "tokensmithChapter": chunk.get("tokensmithChapter"),
            "tokensmithChunkKind": chunk.get("tokensmithChunkKind"),
        }
        for chunk in chunks[:6]
    ]


def preview_cleaning(payload: Dict[str, Any]) -> Dict[str, Any]:
    path = Path(payload["path"]).expanduser().resolve()
    if not path.exists():
        raise EngineError("The selected material no longer exists.")

    files = supported_files(path)
    if not files:
        raise EngineError("No supported files were found in the selected collection.")

    profile = resolve_cleaning_profile(payload.get("cleaningProfileId"))
    rule_ids = resolve_cleaning_rule_ids(profile["id"], payload.get("cleaningRuleIds"))
    sample_path = sorted(files, key=lambda item: str(item).lower())[0]
    raw_pages, cleaned_pages, page_count = preview_text_pages(sample_path, profile["id"], rule_ids)
    chunks = preview_chunks_for_file(sample_path, cleaned_pages, page_count, rule_ids)

    return {
        "profile": {
            "id": profile["id"],
            "name": profile["name"],
            "description": profile["description"],
            "version": CLEANING_PROFILE_VERSION,
        },
        "document": {
            "title": sample_path.stem,
            "path": str(sample_path),
            "kind": material_kind(sample_path),
            "pageCount": page_count,
        },
        "rawPages": [
            {"page": page.get("page"), "text": truncate_preview_text(page.get("text") or "")}
            for page in raw_pages[:2]
        ],
        "cleanedPages": [
            {"page": page.get("page"), "text": truncate_preview_text(page.get("text") or "")}
            for page in cleaned_pages[:2]
        ],
        "chunks": chunks,
        "profiles": cleaning_profiles(),
        "rules": [
            {**rule, "enabled": rule["id"] in rule_ids}
            for rule in cleaning_rules()
        ],
        "cleaningRuleIds": rule_ids,
    }


def summarize_material(path: Path, material_id: str, documents: List[Dict[str, Any]], chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    word_count = sum(document.get("wordCount", 0) for document in documents)
    page_count = sum(document.get("pageCount") or 0 for document in documents)
    ready_documents = [document for document in documents if document.get("status") == "ready"]
    status = "ready" if chunks else "needsReview"
    size = path.stat().st_size if path.exists() else 0
    file_count = len(documents) if path.is_dir() else 1
    first_error = next((document.get("error") for document in documents if document.get("error")), None)
    file_label = f"{len(ready_documents)}/{len(documents)} files" if path.is_dir() else f"{file_type_label(path)} - {format_bytes(size)}"
    stats = f"{format_number(word_count)} words - {format_number(len(chunks))} chunks" if chunks else first_error or "Needs review"
    return {
        "id": material_id,
        "title": path.name,
        "detail": f"{file_label} - {stats}",
        "status": status,
        "kind": material_kind(path),
        "path": str(path),
        "addedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "fileCount": file_count,
        "sizeBytes": size,
        "wordCount": word_count,
        "pageCount": page_count or None,
        "chunkCount": len(chunks),
        "indexedAt": datetime.now().astimezone().isoformat(timespec="seconds") if chunks else None,
        "error": first_error if status == "needsReview" else None,
    }


def index_material(payload: Dict[str, Any]) -> Dict[str, Any]:
    if payload.get("preparation"):
        try:
            from tokensmith_preparation_job import index
        except ImportError:
            from python_engine.tokensmith_preparation_job import index
        return index(payload, sys.modules[__name__])
    user_data_path = payload["userDataPath"]
    init_db(user_data_path)

    request_id = payload.get("_requestId")
    path = Path(payload["path"]).expanduser().resolve()
    if not path.exists():
        raise EngineError("The selected material no longer exists.")

    resume_indexing = bool(payload.get("resume"))
    requested_material_id = str(payload.get("materialId") or "")
    existing_material_id = find_material_id_by_import_path(user_data_path, str(path))
    material_id = existing_material_id if resume_indexing and existing_material_id else requested_material_id or existing_material_id or create_id("material")
    progress_material_id = requested_material_id or material_id
    material_title = str(payload.get("title") or "").strip() or path.name
    chunk_size = DEFAULT_CHUNK_SIZE
    cleaning_profile = resolve_cleaning_profile(payload.get("cleaningProfileId"))
    cleaning_rule_ids = resolve_cleaning_rule_ids(cleaning_profile["id"], payload.get("cleaningRuleIds"))
    documents: List[Dict[str, Any]] = []
    chunks: List[Dict[str, Any]] = []
    files = supported_files(path)
    prepared_files: List[Tuple[Path, Dict[str, Any], List[Dict[str, Any]]]] = []
    total_files = max(len(files), 1)

    def emit_index_progress(
        phase: str,
        percent: int,
        message: str,
        *,
        processed_files: int = 0,
        processed_embeddings: int = 0,
        total_embeddings: int = 0,
    ) -> None:
        send_progress(
            request_id,
            {
                "materialId": progress_material_id,
                "phase": phase,
                "percent": max(0, min(100, percent)),
                "processedFiles": processed_files,
                "totalFiles": len(files),
                "processedEmbeddings": processed_embeddings,
                "totalEmbeddings": total_embeddings,
                "message": message,
            },
        )

    emit_index_progress("parsing", 1, "Parsing")

    for file_index, file_path in enumerate(files, start=1):
        parse_percent = 1 + round(((file_index - 1) / total_files) * 29)
        emit_index_progress("parsing", parse_percent, f"Parsing {file_path.name}", processed_files=file_index - 1)
        document, file_chunks = prepare_index_file(
            material_id,
            file_path,
            user_data_path,
            cleaning_profile["id"],
            cleaning_rule_ids,
        )
        documents.append(document)
        prepared_files.append((file_path, document, file_chunks))
        emit_index_progress(
            "chunking",
            30 + round((file_index / total_files) * 15),
            f"Chunking {file_path.name}",
            processed_files=file_index,
        )

    total_embeddings = sum(len(file_chunks) for _file_path, _document, file_chunks in prepared_files)
    emit_index_progress(
        "chunking",
        45,
        "Chunking complete" if total_embeddings else "No chunks found",
        processed_files=len(files),
        total_embeddings=total_embeddings,
    )

    model = payload.get("model") if isinstance(payload.get("model"), dict) else {}
    embedding_key = embedding_model_key_from_spec(model)
    embed_text = None
    if total_embeddings:
        if not embedding_key:
            raise EngineError("An embedding model is required to index document collections.")

        emit_index_progress(
            "embedding",
            45,
            "Preparing embedding model",
            processed_files=len(files),
            total_embeddings=total_embeddings,
        )
        log_event("embedding_provider_resolve_start", embeddingKey=embedding_key)
        embedding_key, embed_text, embedding_reason = resolve_embedding_provider_from_spec(model, payload.get("embeddingGpuEnabled") is not False)
        log_event(
            "embedding_provider_resolve_success",
            embeddingKey=embedding_key,
            reason=embedding_reason,
        )
        if embedding_reason or embed_text is None:
            log_event("index_embedding_provider_failed", embeddingKey=embedding_key, reason=embedding_reason)
            raise EngineError(f"The selected embedding model was not available: {embedding_reason or 'No embedder was returned.'}")

    existing_embeddings = (
        embedded_chunk_signatures(user_data_path, str(path), embedding_key)
        if resume_indexing and embedding_key
        else set()
    )
    completed_embeddings = 0
    replace_existing_index = not resume_indexing
    index_started = False
    pending_chunks: List[Dict[str, Any]] = []
    deleted_embedding_models: List[str] = []

    def emit_embedding_progress(file_path: Path, chunk_index: int, _file_chunk_total: int) -> None:
        nonlocal completed_embeddings
        completed_embeddings += 1
        embedding_percent = 90 if total_embeddings == 0 else 45 + round((completed_embeddings / total_embeddings) * 45)
        emit_index_progress(
            "embedding",
            min(90, embedding_percent),
            f"Embedding {file_path.name}",
            processed_files=len(files),
            processed_embeddings=completed_embeddings,
            total_embeddings=total_embeddings,
        )

    emit_index_progress(
        "embedding",
        45 if total_embeddings else 90,
        "Embedding in progress" if total_embeddings else "No chunks to embed",
        processed_files=len(files),
        total_embeddings=total_embeddings,
    )

    def material_checkpoint(status: str, is_active: bool, indexed_at: Optional[str], error: Optional[str]) -> Dict[str, Any]:
        checkpoint = summarize_material(path, material_id, documents, chunks)
        checkpoint["chunkSize"] = chunk_size
        checkpoint["title"] = material_title
        checkpoint["status"] = status
        checkpoint["isActive"] = is_active
        checkpoint["indexedAt"] = indexed_at
        checkpoint["error"] = error
        checkpoint["embeddingModel"] = embedding_key
        checkpoint["cleaningProfileId"] = cleaning_profile["id"]
        checkpoint["cleaningProfileName"] = cleaning_profile["name"]
        checkpoint["cleaningProfileVersion"] = CLEANING_PROFILE_VERSION
        checkpoint["cleaningRuleIds"] = cleaning_rule_ids
        if isinstance(model, dict):
            checkpoint["embeddingModelId"] = model.get("id")
            checkpoint["embeddingModelName"] = model.get("name") or model.get("remoteModelName")
        return checkpoint

    def ensure_material_index_started() -> None:
        nonlocal index_started, material_id, material_title, replace_existing_index, progress_material_id, deleted_embedding_models

        if index_started:
            return

        checkpoint = material_checkpoint("indexing", False, None, None)
        deleted_embedding_models = begin_material_index(
            user_data_path,
            checkpoint,
            documents,
            embedding_model=embedding_key,
            replace_existing=replace_existing_index,
        )
        material_id = checkpoint["id"]
        material_title = checkpoint["title"]
        progress_material_id = requested_material_id or material_id
        replace_existing_index = False
        index_started = True

    def flush_pending_chunks() -> None:
        if not pending_chunks:
            return

        ensure_material_index_started()
        append_material_chunks(
            user_data_path,
            material_id,
            pending_chunks,
            embedding_model=embedding_key,
        )
        pending_chunks.clear()
        update_material_index_state(
            user_data_path,
            material_checkpoint("indexing", False, None, None),
            embedding_model=embedding_key,
            rebuild_index=False,
        )

    for file_path, document, file_chunks in prepared_files:
        def persist_indexed_chunk(stored_chunk: Dict[str, Any]) -> None:
            chunks.append(stored_chunk)
            if stored_chunk.get("embedding") is not None:
                pending_chunks.append(stored_chunk)
                if not index_started or len(pending_chunks) >= INDEX_CHECKPOINT_BATCH_SIZE:
                    flush_pending_chunks()

        stored_chunks = indexed_chunks(
            material_id,
            material_title,
            document["id"],
            document["title"],
            file_path,
            file_chunks,
            embedding_key,
            embed_text,
            lambda chunk_index, file_chunk_total, current_file=file_path: emit_embedding_progress(
                current_file,
                chunk_index,
                file_chunk_total,
            ),
            existing_embeddings if resume_indexing else None,
            persist_indexed_chunk,
        )
        document["chunkCount"] = len(stored_chunks)
        document["status"] = "ready" if stored_chunks else "needsReview"
        document["error"] = None if stored_chunks else document.get("error") or "No readable study text was found."
        flush_pending_chunks()

    if not index_started:
        ensure_material_index_started()

    material = summarize_material(path, material_id, documents, chunks)
    material["chunkSize"] = chunk_size
    material["title"] = material_title
    material["isActive"] = material.get("status") == "ready"
    material["embeddingModel"] = embedding_key
    material["cleaningProfileId"] = cleaning_profile["id"]
    material["cleaningProfileName"] = cleaning_profile["name"]
    material["cleaningProfileVersion"] = CLEANING_PROFILE_VERSION
    material["cleaningRuleIds"] = cleaning_rule_ids
    if isinstance(model, dict):
        material["embeddingModelId"] = model.get("id")
        material["embeddingModelName"] = model.get("name") or model.get("remoteModelName")
    emit_index_progress(
        "saving",
        95,
        "Saving index",
        processed_files=len(files),
        processed_embeddings=completed_embeddings,
        total_embeddings=total_embeddings,
    )
    update_material_index_state(
        user_data_path,
        material,
        embedding_model=embedding_key,
        rebuild_index=True,
        deleted_embedding_models=deleted_embedding_models,
    )
    store_collection_vocabulary(user_data_path, material_id, chunks)
    emit_index_progress(
        "complete",
        100,
        "Ready",
        processed_files=len(files),
        processed_embeddings=completed_embeddings,
        total_embeddings=total_embeddings,
    )
    return {"material": material}


def locator_for_chunk(chunk: Dict[str, Any]) -> str:
    start = chunk.get("pageStart")
    end = chunk.get("pageEnd")
    if start and end and end > start:
        return f"Pages {start}-{end}"
    if start:
        return f"Page {start}"
    return f"Chunk {chunk.get('chunkIndex', 1)}"


def source_from_chunk(chunk: Dict[str, Any], score: float, query_tokens: List[str]) -> Dict[str, Any]:
    material_title = chunk.get("materialTitle") or "Course material"
    document_title = chunk.get("documentTitle") or material_title
    title = material_title if material_title == document_title else f"{material_title} / {document_title}"
    text = normalize_text(chunk.get("text", ""))
    chunk_id = chunk.get("stableChunkId") or chunk.get("tokensmithChunkId") or chunk.get("id")
    source_id = "|".join(
        str(part)
        for part in (chunk.get("materialId"), chunk.get("path"), chunk_id)
        if part is not None and str(part).strip()
    )
    return {
        "title": title,
        "locator": locator_for_chunk(chunk),
        "excerpt": excerpt_for(text, query_tokens),
        "context": text,
        "materialId": chunk.get("materialId"),
        "sourceId": source_id or None,
        "chunkId": chunk_id,
        "documentId": chunk.get("documentId"),
        "documentTitle": document_title,
        "collectionName": material_title,
        "path": chunk.get("path"),
        "lineFrom": chunk.get("lineFrom"),
        "lineTo": chunk.get("lineTo"),
        "pageStart": chunk.get("pageStart"),
        "pageEnd": chunk.get("pageEnd"),
        "thumbnailPath": chunk.get("thumbnailPath"),
        "chunkSize": chunk.get("chunkSize"),
        "sectionHeader": chunk.get("sectionHeader"),
        "chunkKind": chunk.get("chunkKind") or chunk.get("tokensmithChunkKind"),
        "tokensmithChunkId": chunk.get("tokensmithChunkId"),
        "tokensmithChapter": chunk.get("tokensmithChapter"),
        "tokensmithChunkKind": chunk.get("tokensmithChunkKind"),
        "score": score,
    }


def remote_openai_embedding(text: str, model: Dict[str, Any]) -> List[float]:
    api_key = str(model.get("apiKey") or "").strip()
    base_url = normalize_remote_base_url(str(model.get("baseUrl") or ""))
    model_name = str(model.get("remoteModelName") or "").strip()
    if not api_key or not base_url or not model_name:
        raise EngineError("Remote embedding model configuration is incomplete.")

    endpoint = f"{base_url}/embeddings"
    payload = json.dumps(
        {
            "model": model_name,
            "input": normalize_text(text)[:REMOTE_EMBEDDING_TEXT_LIMIT],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:300].replace(api_key, "[redacted]")
        raise EngineError(f"Remote embedding request failed with HTTP {error.code}: {detail}") from error
    except Exception as error:
        raise EngineError(f"Remote embedding request failed: {error}") from error

    data = response_payload.get("data") if isinstance(response_payload, dict) else None
    embedding = data[0].get("embedding") if isinstance(data, list) and data else None
    if not isinstance(embedding, list) or not embedding:
        raise EngineError("Remote embedding model returned no embedding vector.")
    return [float(value) for value in embedding]


def ollama_embedding(text: str, model: Dict[str, Any], gpu_enabled: bool = True) -> List[float]:
    base_url = normalize_ollama_base_url(str(model.get("ollamaBaseUrl") or model.get("baseUrl") or ""))
    model_name = str(model.get("ollamaModelName") or "").strip()
    if not model_name:
        raise EngineError("Ollama embedding model configuration is incomplete.")

    endpoint = f"{base_url}/api/embed"
    payload = json.dumps(
        {
            "model": model_name,
            "input": normalize_text(text)[:OLLAMA_EMBEDDING_TEXT_LIMIT],
            "truncate": True,
            "options": {"num_gpu": -1 if gpu_enabled else 0},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:300]
        raise EngineError(f"Ollama embedding request failed with HTTP {error.code}: {detail}") from error
    except Exception as error:
        raise EngineError(f"Ollama embedding request failed: {error}") from error

    embeddings = response_payload.get("embeddings") if isinstance(response_payload, dict) else None
    embedding = embeddings[0] if isinstance(embeddings, list) and embeddings else None
    if not isinstance(embedding, list) or not embedding:
        raise EngineError("Ollama embedding model returned no embedding vector.")
    return [float(value) for value in embedding]


def resolve_embedding_provider_from_spec(model: Dict[str, Any], gpu_enabled: bool = True) -> Tuple[str, Any, Optional[str]]:
    if is_remote_embedding_spec(model):
        if not str(model.get("apiKey") or "").strip():
            return remote_embedding_model_key(model), None, "Remote embedding model API key is missing."
        key = remote_embedding_model_key(model)
        return key, lambda text: remote_openai_embedding(text, model), None

    if is_ollama_embedding_spec(model):
        key = ollama_embedding_model_key(model)
        return key, lambda text: ollama_embedding(text, model, gpu_enabled), None

    return "", None, "Choose an Ollama or remote embedding model."


def embedding_model_specs(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [model for model in payload.get("embeddingModels") or [] if isinstance(model, dict)]


def resolve_embedding_provider_for_key(
    target_key: str,
    specs: List[Dict[str, Any]],
    gpu_enabled: bool = True,
) -> Tuple[Optional[Any], Optional[str]]:
    for spec in specs:
        if is_remote_embedding_spec(spec):
            if remote_embedding_model_key(spec) != target_key:
                continue
            _resolved_key, embed_text, reason = resolve_embedding_provider_from_spec(spec, gpu_enabled)
            return embed_text, reason

        if is_ollama_embedding_spec(spec):
            if ollama_embedding_model_key(spec) != target_key:
                continue
            _resolved_key, embed_text, reason = resolve_embedding_provider_from_spec(spec, gpu_enabled)
            return embed_text, reason


    return None, "The embedding model used for this collection is not installed."


def excerpt_for(text: str, query_tokens: List[str]) -> str:
    lower = text.lower()
    hits: List[int] = []
    for token in query_tokens:
        hits.extend(match.start() for match in re.finditer(re.escape(token.lower()), lower))

    if hits:
        center = max(hits, key=lambda hit: (sum(1 for other in hits if abs(other - hit) <= 260), -hit))
    else:
        center = 0

    start = max(0, center - 420)
    end = min(len(text), center + 420)
    prefix = "..." if start else ""
    suffix = "..." if end < len(text) else ""
    excerpt = re.sub(r"\s+", " ", text[start:end]).strip()
    return f"{prefix}{excerpt}{suffix}"


def source_from_sqlite_chunk(
    row: Dict[str, Any],
    query_tokens: List[str],
    *,
    strip_exercises: bool = False,
) -> Dict[str, Any]:
    text = row.get("text") or ""
    if strip_exercises:
        text = strip_embedded_exercise_text(str(text))
    chunk = {
        "id": row.get("id"),
        "materialId": row.get("material_id"),
        "documentId": row.get("document_id"),
        "materialTitle": row.get("material_title"),
        "documentTitle": row.get("document_title"),
        "path": row.get("path"),
        "text": text,
        "lineFrom": row.get("line_from"),
        "lineTo": row.get("line_to"),
        "pageStart": row.get("page_start"),
        "pageEnd": row.get("page_end"),
        "thumbnailPath": row.get("thumbnail_path"),
        "stableChunkId": row.get("stable_chunk_id"),
        "tokensmithChunkId": row.get("tokensmith_chunk_id"),
        "tokensmithChapter": row.get("tokensmith_chapter"),
        "tokensmithChunkKind": row.get("chunk_kind"),
        "chunkKind": row.get("chunk_kind"),
        "chunkIndex": row.get("chunk_index"),
        "chunkSize": row.get("chunk_size"),
        "sectionHeader": row.get("section_header"),
    }
    source = source_from_chunk(chunk, float(row.get("score") or 0), query_tokens)
    source["chunkRowid"] = row.get("rowid")
    source["sourceUnitId"] = row.get("parent_id")
    source["sourceUnitComplete"] = row.get("unit_complete")
    source["sourceChunkIds"] = row.get("source_chunk_ids") or ([row["stable_chunk_id"]] if row.get("stable_chunk_id") else [])
    source["retrievalMode"] = row.get("retrieval_mode") or "vector"
    source["embeddingModel"] = row.get("query_embedding_model") or row.get("embedding_model")
    source["chunkEmbeddingModel"] = row.get("embedding_model")
    return source


def no_enabled_materials_reason(user_data_path: str) -> str:
    try:
        materials = list_materials(user_data_path)
    except Exception:
        return "no_enabled_materials"
    ready_count = sum(1 for material in materials if material.get("status") == "ready")
    if ready_count:
        return "no_enabled_materials"
    return "no_materials"


SEARCH_MODES = ("vector", "keyword", "hybrid")
DEFAULT_SEARCH_MODE = "hybrid"
RRF_RANK_CONSTANT = 60.0
HYBRID_KEYWORD_WEIGHT = 1.5
HYBRID_VECTOR_WEIGHT = 1.0
EXERCISE_QUERY_TERMS = {
    "check",
    "exercise",
    "exercises",
    "practice",
    "problem",
    "problems",
    "question",
    "questions",
    "quiz",
    "understanding",
}
EXERCISE_HEADING_RE = re.compile(
    r"(?im)^[ \t>]*(?:#{1,6}\s*)?(?:\d+(?:\.\d+)*\s*)?check your understanding\b.*$"
)


def normalize_search_mode(value: Any) -> str:
    mode = str(value or "").strip().lower()
    return mode if mode in SEARCH_MODES else DEFAULT_SEARCH_MODE


def reciprocal_rank_fusion(
    rankings: Sequence[Tuple[float, Sequence[int]]],
    limit: int,
    rank_constant: float = RRF_RANK_CONSTANT,
) -> List[Tuple[int, float]]:
    scores: Dict[int, float] = {}
    first_seen: Dict[int, int] = {}
    order = 0

    for weight, rowids in rankings:
        if weight <= 0:
            continue
        for index, rowid in enumerate(rowids):
            if rowid not in first_seen:
                first_seen[rowid] = order
                order += 1
            scores[rowid] = scores.get(rowid, 0.0) + (weight / (rank_constant + index + 1))

    ranked = sorted(scores.items(), key=lambda item: (-item[1], first_seen[item[0]]))
    return ranked[:limit]


def combine_search_hits(
    mode: str,
    vector_hits: Sequence[Tuple[int, float]],
    keyword_hits: Sequence[Tuple[int, float]],
    limit: int,
) -> List[Tuple[int, float]]:
    vector_ranked = sorted(vector_hits, key=lambda item: item[1], reverse=True)
    keyword_ranked = list(keyword_hits)

    if mode == "vector":
        return vector_ranked[:limit]
    if mode == "keyword":
        return keyword_ranked[:limit]

    return reciprocal_rank_fusion(
        [
            (HYBRID_KEYWORD_WEIGHT, [rowid for rowid, _score in keyword_ranked]),
            (HYBRID_VECTOR_WEIGHT, [rowid for rowid, _score in vector_ranked]),
        ],
        limit,
    )


def strip_embedded_exercise_text(text: str) -> str:
    match = EXERCISE_HEADING_RE.search(str(text or ""))
    if not match:
        return str(text or "")
    return str(text or "")[: match.start()].rstrip()


def source_row_text(row: Dict[str, Any], *, strip_exercises: bool = False) -> str:
    text = str(row.get("text") or "")
    if strip_exercises:
        text = strip_embedded_exercise_text(text)
    metadata = " ".join(
        str(row.get(key) or "") for key in ("section_header", "document_title", "material_title")
    )
    return f"{metadata} {text}"


def source_row_query_matches(row: Dict[str, Any], terms: Sequence[str], *, strip_exercises: bool = False) -> int:
    tokens = set(tokenize(source_row_text(row, strip_exercises=strip_exercises)))
    return sum(1 for term in terms if term in tokens)


def source_row_query_coverage(row: Dict[str, Any], terms: Sequence[str], *, strip_exercises: bool = False) -> float:
    if not terms:
        return 0.0
    return source_row_query_matches(row, terms, strip_exercises=strip_exercises) / float(len(terms))


def source_row_is_exercise(row: Dict[str, Any]) -> bool:
    chunk_kind = str(row.get("chunk_kind") or "").casefold()
    section = str(row.get("section_header") or "").casefold()
    text = str(row.get("text") or "").casefold()
    return (
        chunk_kind in {"exercise", "question", "quiz"}
        or "check your understanding" in section
        or section.strip() in {"exercise", "exercises", "questions"}
        or bool(EXERCISE_HEADING_RE.search(text))
        or text.lstrip().startswith(("check your understanding", "#### check your understanding"))
    )


def query_asks_for_exercise(query_terms: Sequence[str]) -> bool:
    return any(term in EXERCISE_QUERY_TERMS for term in query_terms)


def select_source_rows(
    rows: Sequence[Dict[str, Any]],
    query_terms: Sequence[str],
    keyword_terms: Sequence[str],
    limit: int,
) -> List[Dict[str, Any]]:
    if limit <= 0:
        return []

    useful_terms = list(dict.fromkeys([*keyword_terms, *query_terms]))
    exercise_requested = query_asks_for_exercise(query_terms)
    scored_rows: List[Tuple[float, int, Dict[str, Any]]] = []
    total_rows = max(1, len(rows))

    for index, row in enumerate(rows):
        strip_exercises = not exercise_requested
        coverage = source_row_query_coverage(row, useful_terms, strip_exercises=strip_exercises)
        match_count = source_row_query_matches(row, useful_terms, strip_exercises=strip_exercises)
        rank_score = 1.0 - (index / float(total_rows))
        exercise_penalty = 0.0 if exercise_requested or not source_row_is_exercise(row) else 2.5
        adjusted_score = rank_score + coverage * 1.5 + min(match_count, 4) * 0.15 - exercise_penalty
        scored_rows.append((adjusted_score, index, row))

    selected: List[Dict[str, Any]] = []
    selected_rowids: set[int] = set()

    for _score, _index, row in sorted(scored_rows, key=lambda item: (-item[0], item[1])):
        if len(selected) >= limit:
            break
        rowid = int(row.get("rowid") or 0)
        if rowid in selected_rowids:
            continue
        selected.append(row)
        selected_rowids.add(rowid)
    return selected


def store_collection_vocabulary(
    user_data_path: str,
    material_id: str,
    chunks: Sequence[Dict[str, Any]],
) -> None:
    """Record the collection's high-frequency words as part of indexing.

    Always written, whatever the reader's typo-correction setting: the vocabulary
    belongs to the collection, so whoever indexes it produces it once for everyone
    the collection is later shared with.
    """

    if not chunks:
        return
    try:
        vocabulary = build_vocabulary(
            (str(chunk.get("text") or "") for chunk in chunks),
            min_count=VOCAB_MIN_COUNT,
        )
        save_collection_vocabulary(user_data_path, material_id, vocabulary)
        log_event("vocab_built", materialId=material_id, wordCount=len(vocabulary), minCount=VOCAB_MIN_COUNT)
    except Exception as error:  # a missing vocabulary must never fail indexing
        log_event("vocab_build_failed", materialId=material_id, error=str(error))


def apply_query_typo_correction(
    user_data_path: str,
    query: str,
    material_ids: Sequence[str],
    enabled: bool,
) -> str:
    """Return the query with likely typos repaired against the collection vocabulary.

    A no-op unless typo correction is enabled for this request. Never raises;
    retrieval falls back to the original query on any failure.
    """

    if not enabled or not query.strip() or not material_ids:
        return query
    try:
        vocabulary = collection_vocabulary_words(user_data_path, list(material_ids))
        if not vocabulary:
            return query
        corrected, replacements = correct_query(query, vocabulary, cutoff=TYPO_CUTOFF)
        if replacements:
            log_event(
                "query_typo_correction",
                original=query,
                corrected=corrected,
                replacements=replacements,
                vocabWords=len(vocabulary),
            )
        return corrected
    except Exception as error:  # correction must never break retrieval
        log_event("query_typo_correction_failed", error=str(error))
        return query


def search_library(payload: Dict[str, Any]) -> Dict[str, Any]:
    user_data_path = payload["userDataPath"]
    init_db(user_data_path)

    query = payload.get("query", "")
    limit = int(payload.get("limit") or 4)
    search_mode = normalize_search_mode(payload.get("searchMode"))
    candidate_limit = max(limit * 8, limit) if search_mode in ("keyword", "hybrid") else limit
    materials = payload.get("materials") or []
    embedding_specs = embedding_model_specs(payload)
    requested_active_materials = [
        material
        for material in materials
        if material.get("id") and (material.get("status") == "ready" or material.get("indexedAt")) and material.get("isActive") is not False
    ]
    requested_active_ids = [material["id"] for material in requested_active_materials]
    active_ids = enabled_material_ids_for_requests(user_data_path, requested_active_materials)
    if len(active_ids) != len(requested_active_ids):
        log_event(
            "search_ignored_inactive_or_unknown_materials",
            requested=len(requested_active_ids),
            enabled=len(active_ids),
        )

    if not active_ids:
        return {"sources": [], "reason": no_enabled_materials_reason(user_data_path)}

    if not has_chunks(user_data_path, active_ids):
        return {"sources": [], "reason": "no_indexed_chunks"}

    query = apply_query_typo_correction(
        user_data_path, query, active_ids, bool(payload.get("typoCorrectionEnabled"))
    )
    query_tokens = sorted(set(tokenize(query)))

    collection_embedding_models = embedding_models_by_collection_ids(user_data_path, active_ids)
    active_ids_by_embedding_model: Dict[str, List[str]] = {}
    for material_id in active_ids:
        embedding_key = collection_embedding_models.get(str(material_id))
        if not embedding_key:
            continue
        active_ids_by_embedding_model.setdefault(embedding_key, []).append(str(material_id))

    vector_hits_by_rowid: Dict[int, Tuple[float, str]] = {}
    skipped_embedding_models: List[str] = []

    for embedding_key, grouped_active_ids in (
        active_ids_by_embedding_model.items() if search_mode in ("vector", "hybrid") else []
    ):
        log_event("search_embedding_provider_resolve_start", embeddingKey=embedding_key)
        embed_text, embedding_reason = resolve_embedding_provider_for_key(embedding_key, embedding_specs, payload.get("embeddingGpuEnabled") is not False)
        log_event(
            "search_embedding_provider_resolve_success",
            embeddingKey=embedding_key,
            reason=embedding_reason,
        )
        if embedding_reason or embed_text is None:
            skipped_embedding_models.append(embedding_key)
            continue

        try:
            log_event("query_embedding_start", chars=len(query), embeddingModel=embedding_key)
            query_embedding = embed_text(query)
            log_event("query_embedding_success", embeddingModel=embedding_key)
        except Exception as error:
            log_event("query_embedding_failed", embeddingKey=embedding_key, error=str(error))
            skipped_embedding_models.append(embedding_key)
            continue

        for rowid, score in vector_search(
            user_data_path, query_embedding, grouped_active_ids, candidate_limit, embedding_key
        ):
            current = vector_hits_by_rowid.get(rowid)
            if current is None or score > current[0]:
                vector_hits_by_rowid[rowid] = (score, embedding_key)

    keyword_terms: List[str] = []
    keyword_hits: List[Tuple[int, float]] = []
    if search_mode in ("keyword", "hybrid"):
        keyword_terms = keyword_terms_for_query(user_data_path, query, active_ids)
        keyword_hits = keyword_search(user_data_path, query, active_ids, candidate_limit, keyword_terms)

    vector_hits = [(rowid, score) for rowid, (score, _key) in vector_hits_by_rowid.items()]
    ranked_limit = candidate_limit if search_mode in ("keyword", "hybrid") else limit
    ranked = combine_search_hits(search_mode, vector_hits, keyword_hits, ranked_limit)

    log_event(
        "search_results_ranked",
        queryChars=len(query),
        searchMode=search_mode,
        embeddingModels=list(active_ids_by_embedding_model),
        skippedEmbeddingModels=skipped_embedding_models,
        vectorHits=len(vector_hits),
        keywordHits=len(keyword_hits),
        keywordTerms=keyword_terms,
        candidateHits=len(ranked),
    )

    row_embedding_models = {
        rowid: embedding_key for rowid, (_score, embedding_key) in vector_hits_by_rowid.items()
    }
    rows = fetch_sources(user_data_path, ranked, active_ids)
    for row in rows:
        row["query_embedding_model"] = row_embedding_models.get(int(row["rowid"]))
    rows = expand_source_units(user_data_path, rows, active_ids)
    rows = select_source_rows(rows, query_tokens, keyword_terms, limit)
    for row in rows:
        row["retrieval_mode"] = search_mode
    source_query_tokens = sorted(set([*query_tokens, *keyword_terms]))
    strip_exercise_context = not query_asks_for_exercise(query_tokens)
    sources = [
        source_from_sqlite_chunk(row, source_query_tokens, strip_exercises=strip_exercise_context)
        for row in rows
    ]
    for source in sources:
        source["queryTerms"] = source_query_tokens
        source["keywordTerms"] = list(keyword_terms)
    log_event(
        "search_sources_selected",
        queryChars=len(query),
        searchMode=search_mode,
        selectedHits=len(rows),
        selectedChunkIds=[row.get("stable_chunk_id") or row.get("rowid") for row in rows],
        exerciseChunks=sum(1 for row in rows if source_row_is_exercise(row)),
    )
    return {
        "sources": sources,
        "reason": None if sources else "no_matching_sources",
        "queryTerms": source_query_tokens,
        "keywordTerms": keyword_terms,
    }


def starter_sources(payload: Dict[str, Any]) -> Dict[str, Any]:
    user_data_path = payload["userDataPath"]
    limit = int(payload.get("limit") or 4)
    materials = payload.get("materials") or []
    requested_active_materials = [
        material
        for material in materials
        if material.get("id") and (material.get("status") == "ready" or material.get("indexedAt")) and material.get("isActive") is not False
    ]
    requested_active_ids = [material["id"] for material in requested_active_materials]
    active_ids = enabled_material_ids_for_requests(user_data_path, requested_active_materials)
    if len(active_ids) != len(requested_active_ids):
        log_event(
            "starter_sources_ignored_inactive_or_unknown_materials",
            requested=len(requested_active_ids),
            enabled=len(active_ids),
        )

    if not active_ids:
        return {"sources": [], "reason": no_enabled_materials_reason(user_data_path)}

    if not has_chunks(user_data_path, active_ids):
        return {"sources": [], "reason": "no_indexed_chunks"}

    rows = starter_source_rows(user_data_path, active_ids, limit)
    rows = expand_source_units(user_data_path, rows, active_ids)
    for row in rows:
        row["retrieval_mode"] = "starter"
        row["query_embedding_model"] = row.get("embedding_model")

    sources = [source_from_sqlite_chunk(row, [], strip_exercises=True) for row in rows]
    return {"sources": sources, "reason": None if sources else "no_indexed_chunks"}


def health(payload: Dict[str, Any]) -> Dict[str, Any]:
    # Collections indexed before vocabularies were stored get one here rather than
    # on a later query, so no search pays for a one-time upgrade.
    user_data_path = payload.get("userDataPath")
    if user_data_path:
        try:
            built = backfill_collection_vocabularies(
                str(user_data_path),
                lambda texts: build_vocabulary(texts, min_count=VOCAB_MIN_COUNT),
            )
            if built:
                log_event("vocab_backfilled", collections=built)
        except Exception as error:
            log_event("vocab_backfill_failed", error=str(error))

    return {
        "ok": True,
        "engine": "python",
        "supports": [
            "pdf",
            "txt",
            "markdown",
            "vector-index",
            "sqlite-store",
            "faiss-index",
            "pdfium-pdf-extraction" if pdfium is not None else "pdfium-unavailable",
            "pdfium-pdf-thumbnails" if pdfium is not None else "pdfium-thumbnails-unavailable",
        ],
    }


def preparation_report(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from tokensmith_preparation_job import report
    except ImportError:
        from python_engine.tokensmith_preparation_job import report
    return report(payload)


def list_indexed_materials(payload: Dict[str, Any]) -> Dict[str, Any]:
    user_data_path = payload["userDataPath"]
    materials = list_materials(user_data_path)
    retired = [material for material in materials if str(material.get("embeddingModel") or "").startswith("llama-cpp:")]
    for material in retired:
        delete_material(user_data_path, material["id"])
    return {"materials": [material for material in materials if material not in retired]}


def set_material_enabled(payload: Dict[str, Any]) -> Dict[str, Any]:
    set_material_active(payload["userDataPath"], payload["materialId"], bool(payload["isActive"]))
    return {"ok": True}


def remove_material(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ok": delete_material(
            payload["userDataPath"],
            str(payload.get("materialId") or ""),
            str(payload.get("path") or "") or None,
        )
    }


def resolve_source_document(payload: Dict[str, Any]) -> Dict[str, Any]:
    source = payload.get("source") or {}
    if not isinstance(source, dict):
        return {"source": None}

    return {"source": source_document_for_source(payload["userDataPath"], source)}


COMMANDS = {
    "health": health,
    "preview_cleaning": preview_cleaning,
    "index_material": index_material,
    "preparation_report": preparation_report,
    "search": search_library,
    "starter_sources": starter_sources,
    "list_materials": list_indexed_materials,
    "set_material_enabled": set_material_enabled,
    "remove_material": remove_material,
    "resolve_source_document": resolve_source_document,
}


def send(message: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def send_progress(request_id: Optional[str], progress: Dict[str, Any]) -> None:
    if request_id:
        send({"id": request_id, "progress": progress})


def main() -> None:
    for raw_line in sys.stdin:
        if not raw_line.strip():
            continue
        try:
            request = json.loads(raw_line)
            command = request.get("command")
            request_id = request.get("id")
            if command not in COMMANDS:
                raise EngineError(f"Unknown command: {command}")
            log_event("worker_command_start", command=command, id=request_id)
            payload = request.get("payload") or {}
            if isinstance(payload, dict):
                payload["_requestId"] = request_id
            result = COMMANDS[command](payload)
            log_event("worker_command_success", command=command, id=request_id)
            send({"id": request_id, "ok": True, "result": result})
        except Exception as error:
            log_event(
                "worker_command_error",
                command=locals().get("command"),
                id=locals().get("request", {}).get("id"),
                error=str(error),
            )
            send({"id": locals().get("request", {}).get("id"), "ok": False, "error": str(error)})


if __name__ == "__main__":
    main()
