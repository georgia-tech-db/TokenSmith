"""Resumable preparation and embedding; publish only after a complete replacement exists."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from tokensmith_preparation import completion_client, digest, prepare_blocks, read_json, source_blocks, write_json
except ImportError:
    from python_engine.tokensmith_preparation import completion_client, digest, prepare_blocks, read_json, source_blocks, write_json


def job_directory(user_data_path: str, path: str) -> Path:
    return Path(user_data_path) / 'preparation' / digest(str(Path(path).expanduser().resolve()))


def report(payload: dict) -> dict:
    saved = read_json(job_directory(payload['userDataPath'], payload['path']) / 'report.json', {'documents': []})
    selected = payload.get('documentPath')
    # The outline is cheap; load one document's chunks only when selected.
    result = {**saved, 'documents': []}
    for document in saved.get('documents', []):
        result['documents'].append({**document, 'chunks': (
            read_json(job_directory(payload['userDataPath'], payload['path']) / document['chunksFile'], [])
            if selected == document['path'] and document.get('chunksFile') else []
        )})
    return result


def index(payload: dict, engine: Any) -> dict:
    try:
        from tokensmith_store import upsert_material
    except ImportError:
        from python_engine.tokensmith_store import upsert_material
    path = Path(payload['path']).expanduser().resolve()
    if not path.exists():
        raise ValueError('The selected collection no longer exists.')
    user_data = payload['userDataPath']
    settings = payload['preparation']
    model = payload.get('preparationModel') or {}
    files = engine.supported_files(path)
    if not files:
        raise ValueError('No PDF, Markdown, or text documents were found.')
    directory = job_directory(user_data, str(path))
    material_id = str(payload.get('materialId') or engine.find_material_id_by_import_path(user_data, str(path)) or '')
    previous = next((m for m in engine.list_materials(user_data) if m['id'] == material_id or m.get('path') == str(path)), None)
    title = payload.get('title') or (previous or {}).get('title') or path.name
    if previous:
        material_id = previous['id']

    def progress(phase: str, percent: int, message: str, **extra: Any) -> None:
        engine.send_progress(payload.get('_requestId'), {'materialId': payload.get('materialId') or material_id,
                             'phase': phase, 'percent': percent, 'totalFiles': len(files), 'message': message, **extra})

    def local_complete(messages: list[dict], schema: dict) -> str:
        if not model.get('path'):
            raise ValueError('The local preparation model file is unavailable.')
        llm = engine.load_llama(model['path'], {'contextLength': model.get('contextLength') or 8192})
        result = llm.create_chat_completion(messages=messages, temperature=0, max_tokens=3000,
                                            response_format={'type': 'json_object', 'schema': schema})
        return result['choices'][0]['message']['content']

    progress('parsing', 1, 'Reading documents')
    complete = completion_client(model, local_complete) if settings['mode'] == 'ai' else None
    # Cache keys contain model identity and policy, never credentials.
    identity = {key: model.get(key) for key in ('engine', 'id', 'path', 'remoteModelName', 'baseUrl', 'ollamaModelName', 'ollamaBaseUrl', 'contextLength')}
    model_cache = directory / 'boundaries' / digest(identity)
    documents, prepared, report_documents = [], [], []
    for file_index, file_path in enumerate(files):
        entry = {'title': file_path.stem, 'path': str(file_path), 'status': 'preparing', 'chunkCount': 0}
        progress('parsing', 1 + round(file_index / len(files) * 49), f'Reading {file_path.name}', processedFiles=file_index)
        try:
            if settings['mode'] == 'basic':
                document, chunks = engine.prepare_index_file(material_id, file_path, None,
                    payload.get('cleaningProfileId') or engine.DEFAULT_CLEANING_PROFILE_ID, payload.get('cleaningRuleIds'))
                if not chunks:
                    raise ValueError(document.get('error') or 'No readable text was found.')
            else:
                if file_path.suffix.lower() == '.pdf':
                    pages, page_count = engine.extract_pdf_raw_pages_pdfium(file_path, include_layout_hints=True)
                else:
                    pages, page_count = [{'text': file_path.read_text(encoding='utf-8')}], None
                if not any(p['text'].strip() for p in pages):
                    raise ValueError('No readable text was found. This document needs text recognition.')
                instructions = settings.get('documentInstructions', {}).get(str(file_path), settings.get('instructions') or '')
                blocks = source_blocks(pages)
                chunks = prepare_blocks(blocks, complete, model_cache, instructions,
                    lambda done, total, message: progress('chunking', 1 + round((file_index + done / max(total, 1)) / len(files) * 49),
                        f'{message}: {file_path.name}', processedFiles=file_index),
                    window_chars=max(2000, min(6000, (int(model.get('contextLength') or 8192) - 3000) * 2)))
                document = {'id': engine.create_id('document'), 'materialId': material_id, 'title': file_path.stem,
                            'path': str(file_path), 'kind': engine.material_kind(file_path), 'pageCount': page_count,
                            'wordCount': sum(c['wordCount'] for c in chunks), 'chunkCount': len(chunks), 'status': 'ready'}
                unreadable = sum(not p['text'].strip() for p in pages)
                image_pages = sum(bool(p.get('imageCount')) for p in pages)
                warnings = []
                if unreadable:
                    warnings.append(f'{unreadable} pages have no extractable text.')
                if image_pages:
                    warnings.append(f'{image_pages} pages contain images. Text inside images is not searchable yet; open the original for those details.')
                if warnings:
                    entry['warning'] = ' '.join(warnings)
            document['thumbnails'] = []
            entry.update(status='ready', pageCount=document.get('pageCount'), chunkCount=len(chunks),
                         chunksFile='documents/' + digest(str(file_path)) + '.json')
            write_json(directory / entry['chunksFile'], chunks)
            documents.append(document)
            prepared.append((file_path, document, chunks))
        except Exception as error:
            entry.update(status='needsReview', error=str(error))
        report_documents.append(entry)
        write_json(directory / 'report.json', {'documents': report_documents, 'updatedAt': datetime.now().isoformat()})

    failures = [d for d in report_documents if d['status'] != 'ready']
    if not prepared:
        raise ValueError(failures[0]['error'] if failures else 'No searchable text was found.')
    # A failed document must not make its old searchable content disappear during an update.
    # Other documents are prepared and cached; retry only needs to complete the failed work.
    if failures and previous:
        raise ValueError(f'{len(failures)} documents need attention. Completed preparation is saved; the previous index is still available. ' + failures[0]['error'])

    embedding_model = payload.get('model') or {}
    progress('embedding', 50, 'Preparing search model', processedFiles=len(files))
    embedding_key, embed, reason = engine.resolve_embedding_provider_from_spec(embedding_model)
    if reason or embed is None:
        raise ValueError('The embedding model is unavailable: ' + str(reason or 'Choose an embedder.'))
    total = sum(len(chunks) for _, _, chunks in prepared)
    count, stored = 0, []
    embedding_cache = directory / 'embeddings' / digest(embedding_key)

    def cached_embed(text: str) -> list:
        filename = embedding_cache / (digest(text) + '.json')
        vector = read_json(filename)
        if vector is None:
            vector = embed(text)
            write_json(filename, vector)
        return vector

    for file_path, document, chunks in prepared:
        def on_embedding(done: int, size: int) -> None:
            progress('embedding', 50 + round((count + done) / total * 44), f'Making {file_path.name} searchable',
                     processedFiles=len(files), processedEmbeddings=count + done, totalEmbeddings=total)
        stored.extend(engine.indexed_chunks(material_id, title, document['id'], document['title'], file_path,
                                             chunks, embedding_key, cached_embed, on_embedding))
        count += len(chunks)
    material = engine.summarize_material(path, material_id, documents, stored)
    material.update(title=title, fileCount=len(files), preparation=settings, preparationModelName=model.get('name'),
                    preparationIssueCount=len(failures) + sum(bool(d.get('warning')) for d in report_documents),
                    isActive=previous.get('isActive', True) if previous else True,
                    embeddingModel=embedding_key, embeddingModelId=embedding_model.get('id'),
                    embeddingModelName=embedding_model.get('name'))
    if failures:
        material['detail'] += f' · {len(failures)} documents need attention'
    progress('saving', 95, 'Saving prepared collection')
    # This transaction replaces SQLite content only after all required embeddings are cached.
    upsert_material(user_data, material, documents, stored, embedding_model=embedding_key, replace_existing=True)
    progress('complete', 100, 'Ready for chat', processedFiles=len(files), processedEmbeddings=total, totalEmbeddings=total)
    return {'material': material}
