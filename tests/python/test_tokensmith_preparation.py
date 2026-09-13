import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from python_engine import tokensmith_preparation as prep
from python_engine import tokensmith_engine as engine
from python_engine import tokensmith_store as store


def group(end, title='Passage'):
    return {'end': end, 'title': title, 'kind': 'passage', 'reason': 'Related source content.'}


class PreparationTests(unittest.TestCase):
    def test_short_text_and_cross_page_source_are_preserved_exactly(self):
        pages = [{'page': 1, 'text': 'Title\r\n  α + β = 3\r\n'}, {'page': 2, 'text': 'continued\n\nNo.\n'}]
        blocks = prep.source_blocks(pages)
        with tempfile.TemporaryDirectory() as root:
            chunks = prep.prepare_blocks(blocks, lambda m: {'groups': [group(blocks[-1]['id'])]}, Path(root))
        self.assertEqual(''.join(c['text'] for c in chunks), ''.join(p['text'] for p in pages))
        self.assertEqual((chunks[0]['pageStart'], chunks[0]['pageEnd']), (1, 2))

    def test_invalid_or_incomplete_source_references_are_rejected(self):
        blocks = prep.source_blocks([{'text': 'a\nb\nc'}])
        for groups in [[], [group(2)], [group(4)], [group(2), group(1), group(3)], [group(2), group(2), group(3)], [group(True), group(3)]]:
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                prep.validate_groups({'groups': groups}, blocks)

    def test_retries_and_resumes_without_repeating_completed_model_work(self):
        blocks = prep.source_blocks([{'text': 'First\nSecond\nThird\n'}])
        calls = []
        def complete(messages):
            calls.append(messages)
            return {'groups': [group(1 if len(calls) == 1 else 3)]}
        with tempfile.TemporaryDirectory() as root:
            expected = prep.prepare_blocks(blocks, complete, Path(root))
            self.assertEqual(len(calls), 2)
            actual = prep.prepare_blocks(blocks, lambda _: self.fail('cached call repeated'), Path(root))
            self.assertEqual(actual, expected)
            prep.prepare_blocks(blocks, complete, Path(root), 'Keep the full passage together.')
            self.assertEqual(len(calls), 3)

    def test_window_edges_do_not_duplicate_or_lose_content(self):
        blocks = prep.source_blocks([{'text': ''.join(f'line {i}\n' for i in range(250))}])
        def complete(messages):
            supplied = json.loads(messages[1]['content'])['blocks']
            return {'groups': [group(b['id']) for b in supplied]}
        with tempfile.TemporaryDirectory() as root:
            chunks = prep.prepare_blocks(blocks, complete, Path(root), window_chars=60)
        self.assertEqual(''.join(c['text'] for c in chunks), ''.join(b['text'] for b in blocks))

    def test_long_units_have_bounded_linked_parts(self):
        text = 'long equation ' * 700
        blocks = prep.source_blocks([{'text': text}])
        chunks = prep.materialize(blocks, group(blocks[-1]['id']))
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c['text']) <= prep.MAX_CHUNK_CHARS for c in chunks))
        self.assertEqual(len({c['parentId'] for c in chunks}), 1)
        self.assertEqual(''.join(c['text'] for c in chunks), text)

    def test_failed_model_does_not_silently_use_basic_splitting(self):
        with tempfile.TemporaryDirectory() as root, self.assertRaisesRegex(ValueError, 'three attempts'):
            prep.prepare_blocks(prep.source_blocks([{'text': 'hello'}]), lambda _: '{}', Path(root))

    def test_model_classified_continuation_remains_with_its_passage(self):
        blocks = prep.source_blocks([{'page': 1, 'text': 'A title\nFirst line'}, {'page': 2, 'text': 'Last line\nNext title'}])
        groups = prep.validate_groups({'groups': [group(3), {**group(4), 'kind': 'continuation'}, group(5)]}, blocks)
        self.assertEqual([g['end'] for g in groups], [4, 5])

    def test_start_ids_must_cover_the_beginning_without_reordering(self):
        blocks = prep.source_blocks([{'text': 'one\ntwo\nthree\n'}])
        valid = [{'start': 1, **{k: v for k, v in group(1).items() if k != 'end'}},
                 {'start': 3, **{k: v for k, v in group(1).items() if k != 'end'}}]
        self.assertEqual([g['end'] for g in prep.validate_groups({'groups': valid}, blocks)], [2, 3])
        valid[0]['start'] = 2
        with self.assertRaises(ValueError):
            prep.validate_groups({'groups': valid}, blocks)

    def test_atomic_publication_settings_and_source_page_ranges(self):
        with tempfile.TemporaryDirectory() as root:
            document_path = Path(root) / 'study.txt'
            document_path.write_text('A complete example.\nA result with its explanation.\n')
            payload = {'userDataPath': root, 'path': str(document_path), 'materialId': 'test',
                       'preparation': {'mode': 'ai', 'instructions': '', 'documentInstructions': {}},
                       'preparationModel': {'id': 'test-model', 'engine': 'ollama', 'name': 'Test'},
                       'model': {'id': 'embed', 'name': 'Test embedder'}}
            def complete(messages):
                blocks = json.loads(messages[1]['content'])['blocks']
                return {'groups': [group(blocks[-1]['id'])]}
            with patch('python_engine.tokensmith_preparation_job.completion_client', return_value=complete), patch.object(engine, 'resolve_embedding_provider_from_spec', return_value=('test-embed', lambda _: [1., 0.], None)):
                material = engine.index_material(payload)['material']
            self.assertEqual(store.list_materials(root)[0]['preparation']['mode'], 'ai')
            before = store.dump_index(root)
            document_path.write_text('Changed text\nA changed explanation.\n')
            payload['materialId'] = material['id']
            def fail(_):
                raise ValueError('simulated embedding failure')
            with patch('python_engine.tokensmith_preparation_job.completion_client', return_value=complete), patch.object(engine, 'resolve_embedding_provider_from_spec', return_value=('test-embed', fail, None)), self.assertRaises(Exception):
                engine.index_material(payload)
            after = store.dump_index(root)
            self.assertEqual(before['chunks'], after['chunks'])
            self.assertTrue(store.list_materials(root)[0]['isActive'])
            summary = engine.preparation_report({'userDataPath': root, 'path': str(document_path)})
            self.assertEqual(len(summary['documents']), 1)
            self.assertEqual(summary['documents'][0]['chunks'], [])


if __name__ == '__main__':
    unittest.main()
