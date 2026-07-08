#!/usr/bin/env python3
"""TokenSmith local Python worker.

The worker speaks newline-delimited JSON over stdin/stdout. It deliberately
uses the Python standard library first, with optional llama-cpp-python support
when it is installed locally.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from datetime import datetime

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_BOOT_MODE = sys.argv[1] if len(sys.argv) > 1 else ""
_NEEDS_STORE = _BOOT_MODE != "--llama-embed-worker"
_NEEDS_LLAMA = True

if _NEEDS_STORE:
    try:
        # Import the store first so FAISS initializes its native runtime before
        # llama-cpp. Loading them in the opposite order can trip libomp on macOS.
        from tokensmith_store import (
            delete_material,
            dump_index,
            embedded_chunk_signatures,
            embedding_models_by_collection_ids,
            enabled_material_ids_for_requests,
            fetch_sources,
            find_material_id_by_import_path,
            has_chunks,
            init_db,
            list_materials,
            set_material_active,
            source_document_for_source,
            starter_source_rows,
            upsert_material,
            vector_search,
        )
    except ImportError:  # pragma: no cover - allows direct package imports in tests
        from python_engine.tokensmith_store import (
            delete_material,
            dump_index,
            embedded_chunk_signatures,
            embedding_models_by_collection_ids,
            enabled_material_ids_for_requests,
            fetch_sources,
            find_material_id_by_import_path,
            has_chunks,
            init_db,
            list_materials,
            set_material_active,
            source_document_for_source,
            starter_source_rows,
            upsert_material,
            vector_search,
        )

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

if _NEEDS_LLAMA:
    try:
        from llama_cpp import Llama, LlamaRAMCache  # type: ignore
    except Exception:  # pragma: no cover - optional runtime
        Llama = None  # type: ignore
        LlamaRAMCache = None  # type: ignore
else:
    Llama = None  # type: ignore
    LlamaRAMCache = None  # type: ignore

try:
    import pypdfium2 as pdfium  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    pdfium = None  # type: ignore

try:
    from jinja2 import StrictUndefined  # type: ignore
    from jinja2.sandbox import SandboxedEnvironment  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    StrictUndefined = None  # type: ignore
    SandboxedEnvironment = None  # type: ignore


SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt"}
MAX_FOLDER_FILES = 120
DEFAULT_CHUNK_SIZE = 1000
CHUNK_OVERLAP = 0
PDF_THUMBNAIL_CACHE_DIR = "tokensmith-pdf-thumbnails"
PDF_THUMBNAIL_SCALE = 0.2
PDF_THUMBNAIL_MAX_SIZE = (180, 240)
LLAMA_EMBEDDING_TEXT_LIMIT = 300
REMOTE_EMBEDDING_TEXT_LIMIT = 8000
OLLAMA_EMBEDDING_TEXT_LIMIT = 8000
ANSWER_START = "<<<ANSWER>>>"
ANSWER_END = "<<<END>>>"
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
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
}


def display_model_name(model: Dict[str, Any]) -> str:
    name = str(model.get("name") or "").strip()
    if name:
        return name

    remote_name = str(model.get("remoteModelName") or "").strip()
    if remote_name:
        return remote_name

    model_path = str(model.get("path") or "").strip()
    if model_path:
        return Path(model_path).stem

    return "Local Model"

_GENERATOR_CACHE: Dict[str, Any] = {}
_EMBEDDER_CACHE: Dict[str, Any] = {}
_EMBEDDER_FAILURES: Dict[str, str] = {}
_CHAT_TEMPLATE_CACHE: Dict[str, Optional[str]] = {}

class EngineError(Exception):
    pass


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def log_event(event: str, **details: Any) -> None:
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


def model_hash(model_path: str) -> str:
    resolved = str(Path(model_path).expanduser())
    return hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:16]


def normalize_model_path(model_path: str) -> str:
    return str(Path(model_path).expanduser().resolve())


GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_SCALAR_SIZES = {
    0: 1,   # uint8
    1: 1,   # int8
    2: 2,   # uint16
    3: 2,   # int16
    4: 4,   # uint32
    5: 4,   # int32
    6: 4,   # float32
    7: 1,   # bool
    10: 8,  # uint64
    11: 8,  # int64
    12: 8,  # float64
}


def read_exact(handle: Any, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise EngineError("Unexpected end of GGUF metadata.")
    return data


def read_gguf_string(handle: Any) -> str:
    length = struct.unpack("<Q", read_exact(handle, 8))[0]
    return read_exact(handle, length).decode("utf-8", errors="replace")


def skip_gguf_value(handle: Any, value_type: int) -> None:
    if value_type == GGUF_TYPE_STRING:
        length = struct.unpack("<Q", read_exact(handle, 8))[0]
        handle.seek(length, os.SEEK_CUR)
        return

    if value_type == GGUF_TYPE_ARRAY:
        element_type = struct.unpack("<I", read_exact(handle, 4))[0]
        length = struct.unpack("<Q", read_exact(handle, 8))[0]
        for _index in range(length):
            skip_gguf_value(handle, element_type)
        return

    size = GGUF_SCALAR_SIZES.get(value_type)
    if size is None:
        raise EngineError(f"Unsupported GGUF metadata type: {value_type}")
    handle.seek(size, os.SEEK_CUR)


def read_gguf_metadata_string(model_path: str, key: str) -> Optional[str]:
    path = Path(model_path).expanduser()
    if not path.exists():
        return None

    with path.open("rb") as handle:
        if read_exact(handle, 4) != b"GGUF":
            return None
        _version = struct.unpack("<I", read_exact(handle, 4))[0]
        _tensor_count = struct.unpack("<Q", read_exact(handle, 8))[0]
        metadata_count = struct.unpack("<Q", read_exact(handle, 8))[0]

        for _index in range(metadata_count):
            metadata_key = read_gguf_string(handle)
            value_type = struct.unpack("<I", read_exact(handle, 4))[0]
            if metadata_key == key:
                if value_type != GGUF_TYPE_STRING:
                    return None
                return read_gguf_string(handle)
            skip_gguf_value(handle, value_type)

    return None


def gguf_chat_template(model_path: Optional[str]) -> str:
    if not model_path:
        return ""

    try:
        cache_key = normalize_model_path(model_path)
    except Exception:
        cache_key = str(model_path)

    if cache_key in _CHAT_TEMPLATE_CACHE:
        return _CHAT_TEMPLATE_CACHE[cache_key] or ""

    try:
        template = read_gguf_metadata_string(cache_key, "tokenizer.chat_template")
    except Exception as error:
        log_event("gguf_chat_template_read_failed", modelPath=cache_key, error=str(error))
        template = None

    if template and len(template) >= 2 and template[-1] == "\n" and template[-2] != "\n":
        template = template[:-1]

    _CHAT_TEMPLATE_CACHE[cache_key] = template
    return template or ""


def embedding_model_key(model_path: Optional[str]) -> str:
    if model_path:
        return f"llama-cpp:{model_hash(normalize_model_path(model_path))}"
    return ""


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
    return embedding_model_key(embedding_model_path_from_spec(model))


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


def extract_pdf_raw_pages_pdfium(path: Path, page_limit: Optional[int] = None) -> Tuple[List[Dict[str, Any]], Optional[int]]:
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
            finally:
                if text_page is not None:
                    text_page.close()
                page.close()
            if page_text:
                pages.append({"page": page_number, "text": page_text})
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
    embedding_model_path = embedding_model_path_from_spec(model)
    embedding_key = embedding_model_key_from_spec(model)
    embed_text = None
    if total_embeddings:
        if not embedding_model_path and not is_remote_embedding_spec(model) and not is_ollama_embedding_spec(model):
            raise EngineError("An embedding model is required to index document collections.")

        emit_index_progress(
            "embedding",
            45,
            "Preparing embedding model",
            processed_files=len(files),
            total_embeddings=total_embeddings,
        )
        log_event("embedding_provider_resolve_start", embeddingKey=embedding_key)
        embedding_key, embed_text, embedding_reason = resolve_embedding_provider_from_spec(model)
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

    def save_partial_index() -> None:
        nonlocal material_id, replace_existing_index

        partial_material = summarize_material(path, material_id, documents, chunks)
        partial_material["chunkSize"] = chunk_size
        partial_material["title"] = material_title
        partial_material["status"] = "indexing"
        partial_material["isActive"] = False
        partial_material["indexedAt"] = None
        partial_material["error"] = None
        partial_material["embeddingModel"] = embedding_key
        partial_material["cleaningProfileId"] = cleaning_profile["id"]
        partial_material["cleaningProfileName"] = cleaning_profile["name"]
        partial_material["cleaningProfileVersion"] = CLEANING_PROFILE_VERSION
        partial_material["cleaningRuleIds"] = cleaning_rule_ids
        if isinstance(model, dict):
            partial_material["embeddingModelId"] = model.get("id")
            partial_material["embeddingModelName"] = model.get("name") or model.get("remoteModelName")

        upsert_material(
            user_data_path,
            partial_material,
            documents,
            chunks,
            embedding_model=embedding_key,
            replace_existing=replace_existing_index,
            rebuild_index=False,
        )
        material_id = partial_material["id"]
        replace_existing_index = False

    for file_path, document, file_chunks in prepared_files:
        def persist_indexed_chunk(stored_chunk: Dict[str, Any]) -> None:
            chunks.append(stored_chunk)
            if stored_chunk.get("embedding") is not None:
                save_partial_index()

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
        save_partial_index()

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
    upsert_material(
        user_data_path,
        material,
        documents,
        chunks,
        embedding_model=embedding_key,
        replace_existing=replace_existing_index,
    )
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
    return {
        "title": title,
        "locator": locator_for_chunk(chunk),
        "excerpt": excerpt_for(text, query_tokens),
        "context": text,
        "materialId": chunk.get("materialId"),
        "chunkId": chunk.get("id"),
        "documentId": chunk.get("documentId"),
        "documentTitle": document_title,
        "collectionName": material_title,
        "path": chunk.get("path"),
        "pageStart": chunk.get("pageStart"),
        "pageEnd": chunk.get("pageEnd"),
        "thumbnailPath": chunk.get("thumbnailPath"),
        "chunkSize": chunk.get("chunkSize"),
        "sectionHeader": chunk.get("sectionHeader"),
        "score": score,
    }


def llama_embedder_config() -> Dict[str, Any]:
    return {
        "n_ctx": env_int("TOKENSMITH_EMBED_N_CTX", 512),
        "n_batch": env_int("TOKENSMITH_EMBED_N_BATCH", 128),
        "n_threads": env_int("TOKENSMITH_EMBED_N_THREADS", 4),
        "n_gpu_layers": env_int("TOKENSMITH_EMBED_N_GPU_LAYERS", -1),
        "use_mmap": env_flag("TOKENSMITH_EMBED_USE_MMAP", True),
    }


def create_llama_embedder(model_path: str) -> Any:
    model_path = normalize_model_path(model_path)
    if Llama is None:
        raise EngineError("llama-cpp-python is not installed.")
    if not Path(model_path).expanduser().exists():
        raise EngineError("The selected GGUF model file was not found.")
    return Llama(model_path=model_path, embedding=True, verbose=False, **llama_embedder_config())


def spawn_llama_embedder_worker(model_path: str) -> Dict[str, Any]:
    model_path = normalize_model_path(model_path)
    config = llama_embedder_config()
    log_event("llama_embedder_worker_start", modelPath=model_path, modelHash=model_hash(model_path), **config)
    app_root = str(Path(__file__).resolve().parents[1])
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = os.pathsep.join(
        item for item in [app_root, child_env.get("PYTHONPATH", "")] if item
    )
    worker_code = (
        "import sys; "
        "model_path = sys.argv[1]; "
        "sys.argv = ['tokensmith_engine.py', '--llama-embed-worker', model_path]; "
        "from python_engine.tokensmith_engine import llama_embed_worker_main; "
        "llama_embed_worker_main(model_path)"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", worker_code, model_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        cwd=app_root,
        env=child_env,
    )

    return {"process": process, "nextId": 1, "ready": False}


def wait_for_llama_embedder_worker(model_path: str, worker: Dict[str, Any]) -> Dict[str, Any]:
    if worker.get("ready"):
        return worker

    process = worker["process"]
    assert process.stdout is not None
    ready_line = process.stdout.readline()
    if not ready_line:
        return_code = process.poll()
        raise EngineError(f"The GGUF embedding worker exited before it was ready ({return_code}).")

    try:
        ready = json.loads(ready_line)
    except Exception as error:
        process.kill()
        raise EngineError("The GGUF embedding worker returned unreadable startup output.") from error

    if not ready.get("ok"):
        process.kill()
        raise EngineError(ready.get("error") or "The GGUF embedding worker failed to start.")

    log_event(
        "llama_embedder_worker_ready",
        modelPath=model_path,
        modelHash=model_hash(model_path),
        dimension=ready.get("dimension"),
    )
    worker["ready"] = True
    return worker


def start_llama_embedder_worker(model_path: str) -> Dict[str, Any]:
    worker = spawn_llama_embedder_worker(model_path)
    return wait_for_llama_embedder_worker(model_path, worker)


def get_llama_embedder_worker(model_path: str) -> Dict[str, Any]:
    model_path = normalize_model_path(model_path)
    if model_path in _EMBEDDER_FAILURES:
        raise EngineError(_EMBEDDER_FAILURES[model_path])

    worker = _EMBEDDER_CACHE.get(model_path)
    process = worker.get("process") if isinstance(worker, dict) else None
    if process is not None and process.poll() is None:
        return wait_for_llama_embedder_worker(model_path, worker)

    _EMBEDDER_CACHE.pop(model_path, None)
    try:
        worker = start_llama_embedder_worker(model_path)
    except Exception as error:
        _EMBEDDER_FAILURES[model_path] = str(error)
        raise
    _EMBEDDER_CACHE[model_path] = worker
    return worker


def request_llama_embedding(model_path: str, text: str) -> List[float]:
    model_path = normalize_model_path(model_path)
    worker = get_llama_embedder_worker(model_path)
    process = worker["process"]
    if process.stdin is None or process.stdout is None:
        _EMBEDDER_CACHE.pop(model_path, None)
        raise EngineError("The GGUF embedding worker is not connected.")

    request_id = str(worker["nextId"])
    worker["nextId"] += 1
    process.stdin.write(json.dumps({"id": request_id, "text": text}, ensure_ascii=False) + "\n")
    process.stdin.flush()

    line = process.stdout.readline()
    if not line:
        return_code = process.poll()
        _EMBEDDER_CACHE.pop(model_path, None)
        raise EngineError(f"The GGUF embedding worker exited while embedding text ({return_code}).")

    response = json.loads(line)
    if not response.get("ok"):
        raise EngineError(response.get("error") or "The GGUF embedding worker could not embed text.")
    return normalize_vector(response["embedding"])


def load_llama_embedder(model_path: str) -> Any:
    return get_llama_embedder_worker(normalize_model_path(model_path))


def llama_embedding(text: str, model_path: str) -> List[float]:
    embedding_text = normalize_text(text)[:LLAMA_EMBEDDING_TEXT_LIMIT]
    return request_llama_embedding(model_path, embedding_text)


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


def ollama_embedding(text: str, model: Dict[str, Any]) -> List[float]:
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


def resolve_embedding_provider(model_path: Optional[str]) -> Tuple[str, Any, Optional[str]]:
    if not model_path:
        return "", None, "An embedding model is required."

    model_path = normalize_model_path(model_path)
    key = embedding_model_key(model_path)
    try:
        load_llama_embedder(model_path)
        return key, lambda text: llama_embedding(text, model_path), None
    except Exception as error:
        return key, None, str(error)


def resolve_embedding_provider_from_spec(model: Dict[str, Any]) -> Tuple[str, Any, Optional[str]]:
    if is_remote_embedding_spec(model):
        if not str(model.get("apiKey") or "").strip():
            return remote_embedding_model_key(model), None, "Remote embedding model API key is missing."
        key = remote_embedding_model_key(model)
        return key, lambda text: remote_openai_embedding(text, model), None

    if is_ollama_embedding_spec(model):
        key = ollama_embedding_model_key(model)
        return key, lambda text: ollama_embedding(text, model), None

    return resolve_embedding_provider(embedding_model_path_from_spec(model))


def embedding_model_path_from_spec(model: Dict[str, Any]) -> Optional[str]:
    if not isinstance(model, dict):
        return None
    role = model.get("role")
    if role in {"embedder", "both"}:
        return model.get("embeddingPath") or model.get("path")
    return model.get("embeddingPath")


def embedding_model_specs(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [model for model in payload.get("embeddingModels") or [] if isinstance(model, dict)]


def resolve_embedding_provider_for_key(
    target_key: str,
    specs: List[Dict[str, Any]],
) -> Tuple[Optional[Any], Optional[str]]:
    for spec in specs:
        if is_remote_embedding_spec(spec):
            if remote_embedding_model_key(spec) != target_key:
                continue
            _resolved_key, embed_text, reason = resolve_embedding_provider_from_spec(spec)
            return embed_text, reason

        if is_ollama_embedding_spec(spec):
            if ollama_embedding_model_key(spec) != target_key:
                continue
            _resolved_key, embed_text, reason = resolve_embedding_provider_from_spec(spec)
            return embed_text, reason

        model_path = embedding_model_path_from_spec(spec)
        if not model_path:
            continue
        normalized_path = normalize_model_path(model_path)
        if embedding_model_key(normalized_path) != target_key:
            continue
        resolved_key, embed_text, reason = resolve_embedding_provider(normalized_path)
        if reason:
            return None, reason
        if resolved_key != target_key:
            return None, "The installed embedding model did not match the collection index."
        return embed_text, None

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


def source_from_sqlite_chunk(row: Dict[str, Any], query_tokens: List[str]) -> Dict[str, Any]:
    chunk = {
        "id": row.get("id"),
        "materialId": row.get("material_id"),
        "documentId": row.get("document_id"),
        "materialTitle": row.get("material_title"),
        "documentTitle": row.get("document_title"),
        "path": row.get("path"),
        "text": row.get("text") or "",
        "pageStart": row.get("page_start"),
        "pageEnd": row.get("page_end"),
        "thumbnailPath": row.get("thumbnail_path"),
        "chunkIndex": row.get("chunk_index"),
        "chunkSize": row.get("chunk_size"),
        "sectionHeader": row.get("section_header"),
    }
    source = source_from_chunk(chunk, float(row.get("score") or 0), query_tokens)
    source["chunkRowid"] = row.get("rowid")
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


def search_library(payload: Dict[str, Any]) -> Dict[str, Any]:
    user_data_path = payload["userDataPath"]
    init_db(user_data_path)

    query = payload.get("query", "")
    limit = int(payload.get("limit") or 4)
    materials = payload.get("materials") or []
    embedding_specs = embedding_model_specs(payload)
    requested_active_materials = [
        material
        for material in materials
        if material.get("id") and material.get("status") == "ready" and material.get("isActive") is not False
    ]
    requested_active_ids = [material["id"] for material in requested_active_materials]
    active_ids = enabled_material_ids_for_requests(user_data_path, requested_active_materials)
    if len(active_ids) != len(requested_active_ids):
        log_event(
            "search_ignored_inactive_or_unknown_materials",
            requested=len(requested_active_ids),
            enabled=len(active_ids),
        )

    query_tokens = sorted(set(tokenize(query)))
    if not active_ids:
        return {"sources": [], "reason": no_enabled_materials_reason(user_data_path)}

    if not has_chunks(user_data_path, active_ids):
        return {"sources": [], "reason": "no_indexed_chunks"}

    collection_embedding_models = embedding_models_by_collection_ids(user_data_path, active_ids)
    active_ids_by_embedding_model: Dict[str, List[str]] = {}
    for material_id in active_ids:
        embedding_key = collection_embedding_models.get(str(material_id))
        if not embedding_key:
            continue
        active_ids_by_embedding_model.setdefault(embedding_key, []).append(str(material_id))

    vector_hits_by_rowid: Dict[int, Tuple[float, str]] = {}
    skipped_embedding_models: List[str] = []

    for embedding_key, grouped_active_ids in active_ids_by_embedding_model.items():
        log_event("search_embedding_provider_resolve_start", embeddingKey=embedding_key)
        embed_text, embedding_reason = resolve_embedding_provider_for_key(embedding_key, embedding_specs)
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

        for rowid, score in vector_search(user_data_path, query_embedding, grouped_active_ids, limit, embedding_key):
            current = vector_hits_by_rowid.get(rowid)
            if current is None or score > current[0]:
                vector_hits_by_rowid[rowid] = (score, embedding_key)

    vector_hits_with_models = sorted(
        ((rowid, score, embedding_key) for rowid, (score, embedding_key) in vector_hits_by_rowid.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    log_event(
        "search_results_ranked",
        queryChars=len(query),
        embeddingModels=list(active_ids_by_embedding_model),
        skippedEmbeddingModels=skipped_embedding_models,
        vectorHits=len(vector_hits_with_models),
        topMode="vector" if vector_hits_with_models else None,
    )

    row_embedding_models = {rowid: embedding_key for rowid, _score, embedding_key in vector_hits_with_models}
    rows = fetch_sources(
        user_data_path,
        [(rowid, score) for rowid, score, _embedding_key in vector_hits_with_models[:limit]],
        active_ids,
    )
    for row in rows:
        row["retrieval_mode"] = "vector"
        row["query_embedding_model"] = row_embedding_models.get(int(row["rowid"]))
    sources = [source_from_sqlite_chunk(row, query_tokens) for row in rows]
    return {"sources": sources, "reason": None if sources else "no_matching_sources"}


def starter_sources(payload: Dict[str, Any]) -> Dict[str, Any]:
    user_data_path = payload["userDataPath"]
    limit = int(payload.get("limit") or 4)
    materials = payload.get("materials") or []
    requested_active_materials = [
        material
        for material in materials
        if material.get("id") and material.get("status") == "ready" and material.get("isActive") is not False
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
    for row in rows:
        row["retrieval_mode"] = "starter"
        row["query_embedding_model"] = row.get("embedding_model")

    sources = [source_from_sqlite_chunk(row, []) for row in rows]
    return {"sources": sources, "reason": None if sources else "no_indexed_chunks"}


DEFAULT_SUGGESTED_FOLLOW_UP_PROMPT = (
    "Suggest {count} very short factual follow-up questions that have not been answered yet "
    "or cannot be found inspired by the previous conversation and excerpts."
)


MIN_FOLLOW_UP_SUGGESTION_COUNT = 2
DEFAULT_FOLLOW_UP_SUGGESTION_COUNT = 4


DEFAULT_MODEL_RUNTIME_SETTINGS: Dict[str, Any] = {
    "systemMessage": "",
    "chatTemplate": "",
    "suggestedFollowUpPrompt": DEFAULT_SUGGESTED_FOLLOW_UP_PROMPT,
    "contextLength": 2048,
    "maxLength": 4096,
    "promptBatchSize": 128,
    "temperature": 0.7,
    "topP": 0.4,
    "topK": 40,
    "minP": 0,
    "repeatPenaltyTokens": 64,
    "repeatPenalty": 1.18,
    "gpuLayers": -1,
    "device": "applicationDefault",
}

DEFAULT_APPLICATION_SETTINGS: Dict[str, Any] = {
    "cpuThreads": 4,
    "suggestionMode": "on",
    "followUpSuggestionCount": DEFAULT_FOLLOW_UP_SUGGESTION_COUNT,
}

def clamp_number(value: Any, default_value: float, minimum: float, maximum: float) -> float:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return default_value
    if not math.isfinite(numeric_value):
        return default_value
    return min(max(numeric_value, minimum), maximum)


def normalize_application_settings(settings: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    settings = settings or {}
    suggestion_mode = str(settings.get("suggestionMode") or DEFAULT_APPLICATION_SETTINGS["suggestionMode"])
    if suggestion_mode not in {"on", "off"}:
        suggestion_mode = DEFAULT_APPLICATION_SETTINGS["suggestionMode"]
    raw_follow_up_count = settings.get("followUpSuggestionCount")
    numeric_follow_up_count = clamp_number(
        raw_follow_up_count,
        DEFAULT_FOLLOW_UP_SUGGESTION_COUNT,
        0,
        DEFAULT_FOLLOW_UP_SUGGESTION_COUNT,
    )
    follow_up_count = (
        0
        if suggestion_mode == "off"
        else (
            MIN_FOLLOW_UP_SUGGESTION_COUNT
            if numeric_follow_up_count <= MIN_FOLLOW_UP_SUGGESTION_COUNT
            else DEFAULT_FOLLOW_UP_SUGGESTION_COUNT
        )
    )

    return {
        "cpuThreads": int(round(clamp_number(settings.get("cpuThreads"), DEFAULT_APPLICATION_SETTINGS["cpuThreads"], 1, 64))),
        "suggestionMode": suggestion_mode,
        "followUpSuggestionCount": follow_up_count,
    }


def normalize_model_runtime_settings(settings: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    settings = settings or {}
    device = str(settings.get("device") or DEFAULT_MODEL_RUNTIME_SETTINGS["device"])
    if device not in {"applicationDefault", "cpu", "gpu"}:
        device = DEFAULT_MODEL_RUNTIME_SETTINGS["device"]
    chat_template = str(settings.get("chatTemplate") or DEFAULT_MODEL_RUNTIME_SETTINGS["chatTemplate"])
    suggested_follow_up_prompt = str(
        settings.get("suggestedFollowUpPrompt") or DEFAULT_MODEL_RUNTIME_SETTINGS["suggestedFollowUpPrompt"]
    )

    return {
        "systemMessage": str(settings.get("systemMessage") or DEFAULT_MODEL_RUNTIME_SETTINGS["systemMessage"]),
        "chatTemplate": chat_template or DEFAULT_MODEL_RUNTIME_SETTINGS["chatTemplate"],
        "suggestedFollowUpPrompt": suggested_follow_up_prompt,
        "contextLength": int(round(clamp_number(
            settings.get("contextLength"),
            DEFAULT_MODEL_RUNTIME_SETTINGS["contextLength"],
            512,
            32768,
        ))),
        "maxLength": int(round(clamp_number(
            settings.get("maxLength"),
            DEFAULT_MODEL_RUNTIME_SETTINGS["maxLength"],
            64,
            8192,
        ))),
        "promptBatchSize": int(round(clamp_number(settings.get("promptBatchSize"), DEFAULT_MODEL_RUNTIME_SETTINGS["promptBatchSize"], 1, 4096))),
        "temperature": clamp_number(
            settings.get("temperature"),
            DEFAULT_MODEL_RUNTIME_SETTINGS["temperature"],
            0,
            2,
        ),
        "topP": clamp_number(
            settings.get("topP"),
            DEFAULT_MODEL_RUNTIME_SETTINGS["topP"],
            0,
            1,
        ),
        "topK": int(round(clamp_number(settings.get("topK"), DEFAULT_MODEL_RUNTIME_SETTINGS["topK"], 0, 1000))),
        "minP": clamp_number(settings.get("minP"), DEFAULT_MODEL_RUNTIME_SETTINGS["minP"], 0, 1),
        "repeatPenaltyTokens": int(round(clamp_number(settings.get("repeatPenaltyTokens"), DEFAULT_MODEL_RUNTIME_SETTINGS["repeatPenaltyTokens"], 0, 4096))),
        "repeatPenalty": clamp_number(
            settings.get("repeatPenalty"),
            DEFAULT_MODEL_RUNTIME_SETTINGS["repeatPenalty"],
            1,
            3,
        ),
        "gpuLayers": int(round(clamp_number(settings.get("gpuLayers"), DEFAULT_MODEL_RUNTIME_SETTINGS["gpuLayers"], -1, 999))),
        "device": device,
    }


def model_runtime_settings_from_payload(payload: Dict[str, Any], model: Dict[str, Any], settings: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(payload.get("modelSettings"), dict):
        return normalize_model_runtime_settings(payload["modelSettings"])

    model_id = str(model.get("id") or "")
    model_defaults = settings.get("modelDefaults") if isinstance(settings.get("modelDefaults"), dict) else {}
    model_settings_by_id = settings.get("modelSettingsById") if isinstance(settings.get("modelSettingsById"), dict) else {}
    model_settings = model_settings_by_id.get(model_id) if isinstance(model_settings_by_id.get(model_id), dict) else {}
    return normalize_model_runtime_settings({**model_defaults, **model_settings})


def application_settings_from_payload(payload: Dict[str, Any], settings: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(payload.get("applicationSettings"), dict):
        return normalize_application_settings(payload["applicationSettings"])
    application_settings = settings.get("application") if isinstance(settings.get("application"), dict) else {}
    return normalize_application_settings(application_settings)


def render_chat_template(chat_template: str, messages: List[Dict[str, str]]) -> Optional[str]:
    if SandboxedEnvironment is None or StrictUndefined is None or not chat_template.strip():
        return None

    def raise_template_exception(message: str) -> str:
        raise EngineError(message)

    try:
        environment = SandboxedEnvironment(autoescape=False, undefined=StrictUndefined)
        template = environment.from_string(chat_template)
        rendered = template.render(
            messages=messages,
            add_generation_prompt=True,
            tools=[],
            documents=[],
            controls=[],
            bos_token="",
            eos_token="",
            raise_exception=raise_template_exception,
            strftime_now=lambda fmt: time.strftime(fmt),
        )
    except Exception as error:
        log_event("chat_template_render_failed", error=str(error))
        return None

    return rendered if rendered.strip() else None


def local_source_context(sources: List[Dict[str, Any]], *, max_chars: Optional[int] = None) -> str:
    if not sources:
        return ""

    parts = [
        "Use the context below only when it is relevant to the question.\n",
        "Answer directly. Do not quote the context before answering. Do not mention context labels.\n",
        "If the context does not contain the answer, say that plainly.\n\n",
        "### Context:\n",
    ]
    for source in sources:
        text = normalize_text(source.get("context") or source.get("excerpt", ""))
        if max_chars is not None:
            text = text[:max_chars]
        section_header = str(source.get("sectionHeader") or "").strip()
        locator = str(source.get("locator") or "").strip()
        locator_line = f"Locator: {locator}\n" if locator else ""
        section_line = f"Section: {section_header}\n" if section_header else ""
        parts.append(
            "Collection: "
            f"{source.get('materialTitle') or source.get('collectionName') or source.get('collection') or 'Library'}\n"
            f"Path: {source.get('path') or source.get('title') or ''}\n"
            f"{locator_line}"
            f"{section_line}"
            f"Text: {text}\n\n"
        )
    return "".join(parts)


def generation_messages(
    prompt: str,
    sources: List[Dict[str, Any]],
    model_settings: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    system_message = str((model_settings or {}).get("systemMessage") or "").strip()
    user_content = f"{local_source_context(sources)}{normalize_text(prompt)}"
    messages: List[Dict[str, str]] = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_content})
    return messages


def format_generation_prompt(
    prompt: str,
    sources: List[Dict[str, Any]],
    model_settings: Optional[Dict[str, Any]] = None,
    model_path: Optional[str] = None,
) -> str:
    messages = generation_messages(prompt, sources, model_settings)
    chat_template = str((model_settings or {}).get("chatTemplate") or "") or gguf_chat_template(model_path)
    rendered_template = render_chat_template(chat_template, messages)

    if rendered_template is not None:
        return rendered_template

    return "\n\n".join(f"{message['role']}:\n{message['content']}" for message in messages) + "\n\nassistant:\n"


def format_follow_up_prompt(
    prompt: str,
    answer: str,
    sources: List[Dict[str, Any]],
    model_settings: Dict[str, Any],
    application_settings: Dict[str, Any],
    model_path: Optional[str] = None,
) -> str:
    suggestion_prompt = str(
        model_settings.get("suggestedFollowUpPrompt") or DEFAULT_MODEL_RUNTIME_SETTINGS["suggestedFollowUpPrompt"]
    ).strip()
    count = int(application_settings.get("followUpSuggestionCount", DEFAULT_APPLICATION_SETTINGS["followUpSuggestionCount"]))
    if "{count}" in suggestion_prompt:
        suggestion_prompt = suggestion_prompt.replace("{count}", str(count))
    else:
        suffix = "" if count == 1 else "s"
        suggestion_prompt = f"Generate {count} suggested follow-up question{suffix}.\n{suggestion_prompt}"
    messages = [
        *generation_messages(prompt, sources[:3], model_settings),
        {"role": "assistant", "content": normalize_text(answer)},
        {"role": "user", "content": suggestion_prompt},
    ]
    chat_template = str(model_settings.get("chatTemplate") or "") or gguf_chat_template(model_path)
    rendered_template = render_chat_template(chat_template, messages)

    if rendered_template is not None:
        return rendered_template

    return "\n\n".join(f"{message['role']}: {message['content']}" for message in messages) + "\nassistant:"


FOLLOW_UP_QUESTION_RE = re.compile(r"\b(?:What|Where|How|Why|When|Who|Which|Whose|Whom)\b[^?]*\?")


def parse_follow_up_suggestions(text: str, limit: int = 4) -> List[str]:
    limit = max(0, int(limit))
    if limit == 0:
        return []

    stripped_text = strip_answer_markers(text).strip()
    parsed_lines: List[str] = []

    try:
        parsed = json.loads(stripped_text)
        if isinstance(parsed, list):
            parsed_lines = [str(item) for item in parsed]
    except Exception:
        parsed_lines = []

    if not parsed_lines:
        parsed_lines = [match.group(0) for match in FOLLOW_UP_QUESTION_RE.finditer(stripped_text)]

    suggestions: List[str] = []
    seen: set[str] = set()
    for line in parsed_lines:
        suggestion = re.sub(r"^\s*(?:[-*\u2022]+|\d+[\).\:-])\s*", "", str(line)).strip()
        suggestion = suggestion.strip(" \"'`")
        match = FOLLOW_UP_QUESTION_RE.search(suggestion)
        if not match:
            continue
        suggestion = match.group(0).strip()
        if len(suggestion) > 180:
            suggestion = f"{suggestion[:177].rstrip()}..."
        suggestion_key = suggestion.casefold()
        if suggestion_key in seen:
            continue
        suggestions.append(suggestion)
        seen.add(suggestion_key)
        if len(suggestions) == limit:
            break

    return suggestions


def public_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{key: value for key, value in source.items() if key != "context"} for source in sources]


def llama_generator_config(model_settings: Dict[str, Any], application_settings: Dict[str, Any]) -> Dict[str, int]:
    n_gpu_layers = int(model_settings["gpuLayers"])
    if model_settings.get("device") == "cpu":
        n_gpu_layers = 0

    return {
        "n_ctx": int(model_settings["contextLength"]),
        "n_batch": int(model_settings["promptBatchSize"]),
        "n_threads": int(application_settings["cpuThreads"]),
        "n_gpu_layers": n_gpu_layers,
        "last_n_tokens_size": int(model_settings["repeatPenaltyTokens"]),
    }


def load_llama(model_path: str, model_settings: Optional[Dict[str, Any]] = None, application_settings: Optional[Dict[str, Any]] = None) -> Any:
    if Llama is None:
        raise EngineError("llama-cpp-python is not installed.")
    normalized_model_settings = normalize_model_runtime_settings(model_settings)
    normalized_application_settings = normalize_application_settings(application_settings)
    generator_config = llama_generator_config(normalized_model_settings, normalized_application_settings)
    cache_key = json.dumps({"modelPath": model_path, **generator_config}, sort_keys=True)

    if cache_key not in _GENERATOR_CACHE:
        _GENERATOR_CACHE[cache_key] = Llama(
            model_path=model_path,
            **generator_config,
            flash_attn=True,
            verbose=False,
        )
        if LlamaRAMCache is not None:
            _GENERATOR_CACHE[cache_key].set_cache(LlamaRAMCache())
    return _GENERATOR_CACHE[cache_key]


def strip_answer_markers(text: str) -> str:
    text = text.replace(ANSWER_START, "").replace(ANSWER_END, "")
    return text.strip()


def run_llama_completion(
    prompt: str,
    model_path: str,
    model_settings: Dict[str, Any],
    application_settings: Optional[Dict[str, Any]] = None,
) -> str:
    normalized_model_settings = normalize_model_runtime_settings(model_settings)
    normalized_application_settings = normalize_application_settings(application_settings)
    llm = load_llama(model_path, normalized_model_settings, normalized_application_settings)
    completion_kwargs = {
        "max_tokens": int(normalized_model_settings["maxLength"]),
        "temperature": float(normalized_model_settings["temperature"]),
        "top_p": float(normalized_model_settings["topP"]),
        "top_k": int(normalized_model_settings["topK"]),
        "repeat_penalty": float(normalized_model_settings["repeatPenalty"]),
        "stop": ["</s>", "<|im_end|>", "<|eot_id|>"],
    }
    if float(normalized_model_settings["minP"]) > 0:
        completion_kwargs["min_p"] = float(normalized_model_settings["minP"])

    try:
        result = llm.create_completion(prompt, **completion_kwargs)
    except TypeError:
        if "min_p" not in completion_kwargs:
            raise
        completion_kwargs.pop("min_p")
        result = llm.create_completion(prompt, **completion_kwargs)
    text = strip_answer_markers(result["choices"][0]["text"])
    if not text:
        raise EngineError("The local model returned an empty response.")
    return text


def should_generate_follow_ups(application_settings: Dict[str, Any], sources: List[Dict[str, Any]]) -> bool:
    return application_settings.get("suggestionMode") != "off"


def generate_follow_up_suggestions(
    prompt: str,
    answer: str,
    sources: List[Dict[str, Any]],
    model_path: str,
    model_settings: Dict[str, Any],
    application_settings: Dict[str, Any],
) -> List[str]:
    if not should_generate_follow_ups(application_settings, sources):
        return []
    suggestion_count = int(
        application_settings.get("followUpSuggestionCount", DEFAULT_APPLICATION_SETTINGS["followUpSuggestionCount"])
    )

    suggestion_settings = {
        **model_settings,
        "maxLength": min(int(model_settings.get("maxLength") or 128), 160),
        "temperature": min(max(float(model_settings.get("temperature") or 0.2), 0.2), 0.8),
    }
    try:
        suggestion_text = run_llama_completion(
            format_follow_up_prompt(prompt, answer, sources, suggestion_settings, application_settings, model_path),
            model_path,
            suggestion_settings,
            application_settings,
        )
    except Exception as error:
        log_event("follow_up_generation_failed", error=str(error))
        return []

    return parse_follow_up_suggestions(suggestion_text, suggestion_count)


def chat(payload: Dict[str, Any]) -> Dict[str, Any]:
    user_data_path = payload["userDataPath"]
    prompt = payload.get("prompt", "")
    materials = payload.get("materials") or []
    settings = payload.get("settings") or {}
    limit = max(1, int(settings.get("maxSources") or 4))
    model = payload.get("model") or {}
    application_settings = application_settings_from_payload(payload, settings)
    model_settings = model_runtime_settings_from_payload(payload, model, settings)
    if "retrievedSources" in payload:
        sources = payload.get("retrievedSources") or []
    else:
        try:
            search_result = search_library(
                {
                    "userDataPath": user_data_path,
                    "query": prompt,
                    "materials": materials,
                    "limit": limit,
                    "model": model,
                    "embeddingModels": payload.get("embeddingModels") or [],
                }
            )
            sources = search_result["sources"]
        except Exception as error:
            log_event("chat_retrieval_failed", error=str(error))
            sources = []
    model_path = model.get("path")
    model_name = display_model_name(model)
    if model_path:
        try:
            text = run_llama_completion(
                format_generation_prompt(prompt, sources, model_settings, model_path),
                model_path,
                model_settings,
                application_settings,
            )
            follow_up_suggestions = generate_follow_up_suggestions(
                prompt,
                text,
                sources,
                model_path,
                model_settings,
                application_settings,
            )
            return {
                "engineId": "tokensmith",
                "modelName": model_name,
                "text": text,
                "sources": public_sources(sources),
                "followUpSuggestions": follow_up_suggestions,
            }
        except Exception as error:
            log_event("chat_generation_failed", modelPath=model_path, error=str(error))
            return {
                "engineId": "tokensmith",
                "modelName": model_name,
                "text": f"The local model could not generate an answer: {error}",
                "sources": public_sources(sources),
                "followUpSuggestions": [],
            }
    return {
        "engineId": "tokensmith",
        "modelName": model_name,
        "text": "A local model is required to answer.",
        "sources": public_sources(sources),
        "followUpSuggestions": [],
    }


def health(_payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ok": True,
        "engine": "python",
        "llamaCppAvailable": Llama is not None,
        "supports": [
            "pdf",
            "txt",
            "markdown",
            "vector-index",
            "sqlite-store",
            "faiss-index",
            "pdfium-pdf-extraction" if pdfium is not None else "pdfium-unavailable",
            "pdfium-pdf-thumbnails" if pdfium is not None else "pdfium-thumbnails-unavailable",
            "gguf-embeddings-optional",
            "gguf-inference-optional",
        ],
    }


def list_indexed_materials(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"materials": list_materials(payload["userDataPath"])}


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
    "search": search_library,
    "starter_sources": starter_sources,
    "chat": chat,
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


def llama_embed_worker_main(model_path: str) -> None:
    try:
        embedder = create_llama_embedder(model_path)
        warmup = embedder.create_embedding("test")
        dimension = len(warmup["data"][0]["embedding"])
        send({"ok": True, "dimension": dimension})
    except Exception as error:
        send({"ok": False, "error": str(error)})
        raise SystemExit(1)

    for raw_line in sys.stdin:
        if not raw_line.strip():
            continue
        try:
            request = json.loads(raw_line)
            text = normalize_text(request.get("text") or "")[:LLAMA_EMBEDDING_TEXT_LIMIT]
            result = embedder.create_embedding(text)
            send({"id": request.get("id"), "ok": True, "embedding": result["data"][0]["embedding"]})
        except Exception as error:
            send({"id": locals().get("request", {}).get("id"), "ok": False, "error": str(error)})


def main() -> None:
    if len(sys.argv) >= 3 and sys.argv[1] == "--llama-embed-worker":
        llama_embed_worker_main(sys.argv[2])
        return

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
