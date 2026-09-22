"""Local PDF experiment: real production preparation and embeddings, isolated indexes.

Does not change the library, tune prompts, skip failed windows, or fall back to Basic.
The source PDF and extracted text are local-only artifacts, not distributed fixtures.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from python_engine import tokensmith_engine as engine
from python_engine import tokensmith_preparation as prep
from python_engine import tokensmith_store as store


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--document', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--arm', choices=['basic', 'ai'], required=True)
    args = parser.parse_args()
    source = Path(args.document).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    definitions = prep.read_json(Path('tests/experiments/db50-cases.json'))
    first, last = definitions['physicalPages']
    model = {'id': 'ollama:gemma4:e4b', 'engine': 'ollama', 'ollamaModelName': 'gemma4:e4b', 'contextLength': 8192}
    embedder = {'id': 'ollama:nomic-embed-text', 'engine': 'ollama', 'role': 'embedder', 'ollamaModelName': 'nomic-embed-text'}
    profile = str(out / args.arm / 'profile')
    result = {'arm': args.arm, 'profile': profile, 'model': model, 'embedder': embedder,
              'document': str(source), 'physicalPages': [first, last], 'preparationVersion': prep.VERSION,
              'cases': [], 'timing': {}, 'preparationCalls': []}
    result['systemPrompt'] = prep.SYSTEM if args.arm == 'ai' else None
    report_path = out / args.arm / 'results.json'
    save = lambda: prep.write_json(report_path, result)
    started = time.perf_counter()
    # Extract exactly the selected physical pages, preserving production PDFium text.
    pages = []
    pdf = engine.pdfium.PdfDocument(str(source))
    try:
        for number in range(first, last + 1):
            page = pdf[number - 1]
            text_page = page.get_textpage()
            try:
                pages.append({'page': number, 'text': text_page.get_text_range() or ''})
            finally:
                text_page.close()
                page.close()
    finally:
        pdf.close()
    result['timing']['extractionSeconds'] = time.perf_counter() - started
    result['sourceSha256'] = hashlib.sha256(json.dumps(pages, ensure_ascii=False).encode()).hexdigest()
    result['sourceCharacters'] = sum(len(p['text']) for p in pages)
    prep.write_json(out / 'pages.json', pages)
    split_started = time.perf_counter()
    try:
        if args.arm == 'basic':
            cleaned = engine.clean_pages(pages, engine.DEFAULT_CLEANING_PROFILE_ID, None)
            chunks = engine.chunk_pdf_pages(cleaned)
        else:
            complete = prep.completion_client(model)

            def measured_complete(messages, schema):
                call_start = time.perf_counter()
                response = complete(messages, schema)
                result['preparationCalls'].append({'seconds': time.perf_counter() - call_start,
                    'inputCharacters': sum(len(m['content']) for m in messages), 'response': response})
                save()
                return response

            def progress(done, total, message):
                result['progress'] = {'done': done, 'total': total, 'message': message}
                save()
                print(args.arm, done, '/', total, message, flush=True)

            blocks = prep.source_blocks(pages)
            chunks = prep.prepare_blocks(blocks, measured_complete, out / args.arm / 'boundary-cache', progress=progress)
            result['exactCoverage'] = ''.join(c['text'] for c in chunks) == ''.join(b['text'] for b in blocks)
    except Exception as error:
        result['error'] = str(error)
        result['timing']['preparationSeconds'] = time.perf_counter() - split_started
        result['timing']['totalSeconds'] = time.perf_counter() - started
        result['cases'] = [{'id': g['id'], 'error': str(error)} for g in definitions['groups']]
        save()
        print(args.arm, 'FAILED', result['error'], flush=True)
        return 1
    result['timing']['preparationSeconds'] = time.perf_counter() - split_started
    result['chunkCount'] = len(chunks)
    result['unitCount'] = len({c.get('parentId', str(i)) for i, c in enumerate(chunks)})
    prep.write_json(out / args.arm / 'chunks.json', chunks)
    save()
    print(args.arm, 'EMBED', len(chunks), 'chunks', flush=True)
    key, embed, reason = engine.resolve_embedding_provider_from_spec(embedder)
    if embed is None or reason:
        raise RuntimeError(reason or 'No embedder')
    embedding_started = time.perf_counter()
    document = {'id': 'db50', 'path': str(source), 'title': definitions['book'], 'kind': 'pdf',
                'status': 'ready', 'pageCount': 50, 'wordCount': sum(c['wordCount'] for c in chunks), 'chunkCount': len(chunks)}
    stored = engine.indexed_chunks('db50', 'db50', definitions['book'], definitions['book'], source, chunks, key, embed)
    result['timing']['embeddingSeconds'] = time.perf_counter() - embedding_started
    publication_started = time.perf_counter()
    material = engine.summarize_material(source, 'db50', [document], stored)
    material.update(title=definitions['book'], isActive=True, embeddingModel=key, embeddingModelId=embedder['id'])
    store.upsert_material(profile, material, [document], stored, embedding_model=key, replace_existing=True)
    material = next(m for m in store.list_materials(profile) if m['path'] == str(source))
    result['timing']['publicationSeconds'] = time.perf_counter() - publication_started
    result['timing']['totalSeconds'] = time.perf_counter() - started
    result['cases'] = [{'id': g['id'], 'material': material, 'questions': [q['question'] for q in g['questions']]} for g in definitions['groups']]
    save()
    print(args.arm, json.dumps(result['timing']), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
