import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from python_engine import tokensmith_preparation as prep
from python_engine import tokensmith_engine as engine
from python_engine import tokensmith_store as store

PASSAGE = ('When the buffer pool is full the evict method selects the least recently used page, '
           'flushes it when it is dirty, and removes it from the page table so the frame can be reused.')
SENTENCE = 'When memory is full the system kicks out the page nobody has used for longest to make room.'


def chunk(text=PASSAGE, **overrides):
    value = {'text': text, 'wordCount': len(text.split()), 'chunkSize': len(text),
             'sectionHeader': 'Making room', 'tokensmithChunkKind': 'prose'}
    value.update(overrides)
    return value


def responder(sentence=SENTENCE, calls=None):
    def complete(messages, schema):
        if calls is not None:
            calls.append(messages)
        return json.dumps({'sentence': sentence})
    return complete


class PlainLanguageTermsTests(unittest.TestCase):
    def test_one_call_per_unit_and_terms_land_on_the_first_part(self):
        calls = []
        chunks = [chunk(parentId='unit', part=1, parts=2), chunk('A later part of the same unit. ' * 6, parentId='unit', part=2, parts=2)]
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(prep.plain_language_terms(chunks, responder(calls=calls), Path(root)), 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(chunks[0]['keywords'], SENTENCE)
        self.assertNotIn('keywords', chunks[1])

    def test_code_and_short_passages_are_skipped(self):
        calls = []
        chunks = [chunk(tokensmithChunkKind='code'), chunk('Too short.'), chunk()]
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(prep.plain_language_terms(chunks, responder(calls=calls), Path(root)), 1)
        self.assertEqual(len(calls), 1)

    def test_repeated_preparation_of_unchanged_text_makes_no_model_call(self):
        calls = []
        with tempfile.TemporaryDirectory() as root:
            prep.plain_language_terms([chunk()], responder(calls=calls), Path(root))
            second = [chunk()]
            prep.plain_language_terms(second, responder(calls=calls), Path(root))
        self.assertEqual(len(calls), 1)
        self.assertEqual(second[0]['keywords'], SENTENCE)

    def test_descriptions_of_the_passage_and_model_failures_leave_no_terms(self):
        def explode(messages, schema):
            raise RuntimeError('model unavailable')
        with tempfile.TemporaryDirectory() as root:
            described = [chunk()]
            self.assertEqual(prep.plain_language_terms(described, responder('This passage explains buffer eviction in detail.'), Path(root)), 0)
            self.assertNotIn('keywords', described[0])
            failed = [chunk()]
            self.assertEqual(prep.plain_language_terms(failed, explode, Path(root)), 0)
            self.assertNotIn('keywords', failed[0])

    def test_terms_are_searchable_and_embedded_without_changing_the_passage(self):
        with tempfile.TemporaryDirectory() as root:
            document_path = Path(root) / 'book.txt'
            document_path.write_text(PASSAGE + '\n')
            payload = {'userDataPath': root, 'path': str(document_path), 'materialId': 'test',
                       'preparation': {'mode': 'basic', 'instructions': '', 'documentInstructions': {},
                                       'plainLanguageTerms': True},
                       'preparationModel': {'id': 'test-model', 'engine': 'ollama', 'name': 'Test'},
                       'model': {'id': 'embed', 'name': 'Test embedder'}}
            embedded = []

            def embed(text):
                embedded.append(text)
                return [1.0, 0.0]

            with patch('python_engine.tokensmith_preparation_job.completion_client', return_value=responder()), \
                 patch.object(engine, 'resolve_embedding_provider_from_spec', return_value=('test-embed', embed, None)):
                material = engine.index_material(payload)['material']

            stored = store.dump_index(root)['chunks']
            self.assertEqual([row['chunk_text'] for row in stored], [PASSAGE])
            self.assertNotIn(SENTENCE, ''.join(row['chunk_text'] for row in stored))
            self.assertEqual([row['keywords'] for row in stored], [SENTENCE])
            self.assertTrue(any(SENTENCE in text and PASSAGE in text for text in embedded))
            hits = store.keyword_search(root, 'kicks out', [material['id']], 4)
            self.assertEqual(len(hits), 1)

    def test_basic_preparation_without_the_option_needs_no_model_and_stores_no_terms(self):
        with tempfile.TemporaryDirectory() as root:
            document_path = Path(root) / 'book.txt'
            document_path.write_text(PASSAGE + '\n')
            payload = {'userDataPath': root, 'path': str(document_path), 'materialId': 'test',
                       'preparation': {'mode': 'basic', 'instructions': '', 'documentInstructions': {}},
                       'model': {'id': 'embed', 'name': 'Test embedder'}}
            with patch('python_engine.tokensmith_preparation_job.completion_client') as client, \
                 patch.object(engine, 'resolve_embedding_provider_from_spec', return_value=('test-embed', lambda _: [1.0, 0.0], None)):
                engine.index_material(payload)
            client.assert_not_called()
            self.assertEqual([row['keywords'] for row in store.dump_index(root)['chunks']], [None])


if __name__ == '__main__':
    unittest.main()
