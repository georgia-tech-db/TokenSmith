import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from python_engine import tokensmith_engine as engine


MODEL = {'id': 'ollama:test', 'engine': 'ollama', 'role': 'embedder', 'ollamaModelName': 'nomic-embed-text'}


class EmbeddingDeviceTests(unittest.TestCase):
    def response(self, request, timeout=0):
        self.requests.append(json.loads(request.data))
        return io.BytesIO(json.dumps({'embeddings': [[1., 0., 0.]]}).encode())

    def test_indexing_and_hybrid_search_share_default_and_cpu_options(self):
        for mode in [None, False, True]:
            for preparation in [None, {'mode': 'basic', 'instructions': '', 'documentInstructions': {}}]:
                with self.subTest(gpu=mode, preparation=preparation), tempfile.TemporaryDirectory() as root:
                    self.requests = []
                    path = Path(root) / 'notes.txt'
                    path.write_text('Slotted pages keep record identifiers stable when record data moves inside a page.\n' * 12)
                    options = {} if mode is None else {'embeddingGpuEnabled': mode}
                    with patch.object(engine.urllib.request, 'urlopen', side_effect=self.response):
                        material = engine.index_material({'path': str(path), 'userDataPath': root, 'model': MODEL,
                                                          'preparation': preparation, **options})['material']
                        index_calls = len(self.requests)
                        result = engine.search_library({'userDataPath': root, 'materials': [material],
                                                        'embeddingModels': [MODEL], 'query': 'Why stable record identifiers?',
                                                        'searchMode': 'hybrid', **options})
                    self.assertGreater(index_calls, 0)
                    self.assertEqual(len(self.requests), index_calls + 1)
                    self.assertTrue(result['sources'])
                    self.assertTrue(all(call['options']['num_gpu'] == (0 if mode is False else -1) for call in self.requests))
                    self.assertEqual(material['embeddingModel'], 'ollama:nomic-embed-text')

    def test_cloud_requests_do_not_receive_device_options(self):
        calls = []
        def respond(request, timeout=0):
            calls.append(json.loads(request.data))
            return io.BytesIO(b'{"data":[{"embedding":[1,0]}]}')
        model = {'engine': 'remote', 'role': 'embedder', 'remoteModelName': 'embed', 'baseUrl': 'https://example.com/v1', 'apiKey': 'test'}
        with patch.object(engine.urllib.request, 'urlopen', side_effect=respond):
            for enabled in (True, False):
                _, embed, error = engine.resolve_embedding_provider_from_spec(model, enabled)
                self.assertIsNone(error)
                self.assertEqual(embed('hello'), [1., 0.])
        self.assertTrue(all('options' not in call for call in calls))

    def test_unsupported_embedding_models_fail_before_index_mutation(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'notes.txt'
            path.write_text('Slotted pages keep stable record identifiers when records move inside a database page.\n' * 12)
            with self.assertRaisesRegex(engine.EngineError, 'embedding model is required'):
                engine.index_material({'userDataPath': root, 'path': str(path), 'model': {'engine': 'python', 'path': 'old.gguf'}})
            self.assertEqual(engine.list_indexed_materials({'userDataPath': root})['materials'], [])
        self.assertNotIn('chat', engine.COMMANDS)

    def test_gguf_collection_cleanup_removes_index_only_and_preserves_source_file(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'notes.txt'
            path.write_text('A stable record ID names a slot rather than a byte offset.\n' * 12)
            with patch.object(engine, 'resolve_embedding_provider_from_spec', return_value=('llama-cpp:old', lambda _: [1., 0.], None)):
                old = engine.index_material({'path': str(path), 'userDataPath': root, 'model': MODEL})['material']
            with patch.object(engine, 'resolve_embedding_provider_from_spec', return_value=('ollama:nomic-embed-text', lambda _: [1., 0.], None)):
                second = Path(root) / 'new.txt'
                second.write_text('A B+ tree stores records in leaf nodes and uses separator keys to direct searches.\n' * 12)
                new = engine.index_material({'path': str(second), 'userDataPath': root, 'model': MODEL})['material']
            materials = engine.list_indexed_materials({'userDataPath': root})['materials']
            self.assertEqual([m['id'] for m in materials], [new['id']])
            self.assertTrue(path.exists())
            self.assertFalse(engine.has_chunks(root, [old['id']]))

    def test_search_mode_defaults_are_unchanged(self):
        self.assertEqual(engine.normalize_search_mode(None), 'hybrid')
        self.assertEqual(engine.normalize_search_mode('bad'), 'hybrid')
        for mode in ['hybrid', 'vector', 'keyword']:
            self.assertEqual(engine.normalize_search_mode(mode), mode)
