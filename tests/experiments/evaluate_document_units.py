"""Opt-in real PDF preparation, embedding, and hybrid retrieval in an isolated profile.

--documents maps the case IDs to local PDFs. Nothing is downloaded or changed in the
live library. PDF contents and model outputs stay in the requested output directory.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from python_engine import tokensmith_engine as engine
from python_engine import tokensmith_store as store
from python_engine import tokensmith_preparation as prep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--documents', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--model', default='gemma4:e4b')
    parser.add_argument('--cases', nargs='*')
    args = parser.parse_args()
    files = json.loads(Path(args.documents).read_text())
    definitions = json.loads(Path('tests/benchmarks/document_units_cases.json').read_text())
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    profile = str(out / 'profile')
    store.init_db(profile)
    model = {'id': 'ollama:' + args.model, 'engine': 'ollama', 'ollamaModelName': args.model, 'contextLength': 8192}
    embedder = {'id': 'ollama:nomic-embed-text', 'engine': 'ollama', 'role': 'embedder', 'ollamaModelName': 'nomic-embed-text'}
    key, embed, reason = engine.resolve_embedding_provider_from_spec(embedder)
    if reason or embed is None:
        raise RuntimeError(reason or 'Embedder unavailable')
    results = []
    def save():
        prep.write_json(out / 'results.json', {'profile': profile, 'model': model, 'embedder': embedder, 'cases': results})
    for case in definitions:
        if args.cases and case['id'] not in args.cases:
            continue
        start = time.monotonic()
        filename = Path(files[case['id']]).resolve()
        pages, page_count = engine.extract_pdf_raw_pages_pdfium(filename, max(case['pages']), include_layout_hints=True)
        pages = [p for p in pages if p['page'] in case['pages']]
        blocks = prep.source_blocks(pages)
        print(case['id'], 'prepare', len(blocks), 'blocks', flush=True)
        try:
            chunks = prep.prepare_blocks(blocks, prep.completion_client(model), out / 'boundaries' / case['id'],
                progress=lambda done, total, message: print(case['id'], done, '/', total, message, flush=True))
        except Exception as error:
            results.append({**case, 'error': str(error), 'seconds': time.monotonic() - start})
            save()
            print(case['id'], 'FAILED', error, flush=True)
            continue
        prepare_seconds = time.monotonic() - start
        parents = {}
        for chunk in chunks:
            parents.setdefault(chunk['parentId'], []).append(chunk)
        texts = [''.join(c['text'] for c in parts) for parts in parents.values()]
        checks = []
        headings = case.get('independentHeadings', [])
        source_text = ''.join(b['text'] for b in blocks)
        for index, heading in enumerate(headings):
            matching = [text for text in texts if heading in text]
            end = source_text.index(headings[index + 1]) if index + 1 < len(headings) else len(source_text)
            complete_work = source_text[source_text.index(heading):end].strip()
            checks.append({'name': heading, 'passed': len(matching) == 1 and sum(
                other in matching[0] for other in headings) == 1 and complete_work in matching[0]})
        for unit in case['units']:
            matching = [text for text in texts if all(term in text for term in unit['together'])]
            checks.append({'name': unit['name'], 'passed': len(matching) == 1 and not any(
                term in matching[0] for term in unit['apart'])})
        document = {'id': case['id'], 'path': str(filename), 'title': case['id'], 'pageCount': page_count,
                    'kind': 'pdf', 'status': 'ready', 'wordCount': sum(c['wordCount'] for c in chunks), 'chunkCount': len(chunks)}
        stored = engine.indexed_chunks(case['id'], case['id'], case['id'], case['id'], filename, chunks, key, embed)
        material = engine.summarize_material(filename, case['id'], [document], stored)
        material.update(title=case['id'], isActive=True, embeddingModel=key, embeddingModelId=embedder['id'])
        store.upsert_material(profile, material, [document], stored, embedding_model=key, replace_existing=True)
        # Collections have database IDs after publication, not the temporary import IDs.
        material = next(m for m in store.list_materials(profile) if m['path'] == str(filename))
        searches = []
        for question in case['questions']:
            result = engine.search_library({'userDataPath': profile, 'query': question, 'materials': [material],
                'embeddingModel': embedder, 'embeddingModels': [embedder], 'limit': 4, 'searchMode': 'hybrid'})
            searches.append({'question': question, **result})
        result = {**case, 'material': material, 'prepareSeconds': prepare_seconds,
            'seconds': time.monotonic() - start, 'unitChecks': checks, 'chunkCount': len(chunks), 'unitCount': len(parents),
            'exactCoverage': ''.join(c['text'] for c in chunks) == ''.join(b['text'] for b in blocks), 'searches': searches}
        prep.write_json(out / (case['id'] + '-chunks.json'), chunks)
        results.append(result)
        save()
        print(case['id'], 'DONE', json.dumps(checks), round(result['seconds'], 1), 'seconds', flush=True)
    if any(c.get('error') or not c['exactCoverage'] or not all(u['passed'] for u in c['unitChecks']) for c in results):
        sys.exit(1)


if __name__ == '__main__':
    main()
