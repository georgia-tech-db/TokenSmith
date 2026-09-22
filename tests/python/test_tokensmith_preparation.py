import io
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
            chunks = prep.prepare_blocks(blocks, lambda m, s: {'starts': []}, Path(root))
        self.assertEqual(''.join(c['text'] for c in chunks), ''.join(p['text'] for p in pages))
        self.assertEqual((chunks[0]['pageStart'], chunks[0]['pageEnd']), (1, 2))

    def test_invalid_or_incomplete_source_references_are_rejected(self):
        blocks = prep.source_blocks([{'text': 'a\nb\nc'}])
        for starts in [None, [True], [False], ['1'], [1.0], [0], [-1], [4], [{}], [[]]]:
            with self.subTest(starts=starts), self.assertRaises(ValueError):
                prep.validate_groups({'starts': starts}, blocks)

    def test_boundary_order_and_repetition_do_not_change_selected_locations(self):
        blocks = prep.source_blocks([{'text': 'Earlier\nBody\nLead-in\nFirst\nBody\nSecond\n'}])[2:]
        value = {'starts': [4, 2, 4, 2]}
        groups = prep.validate_groups(value, blocks)
        self.assertEqual(groups, prep.validate_groups({'starts': [2, 4]}, blocks))
        self.assertEqual([(g['start'], g['end']) for g in groups], [(3, 3), (4, 5), (6, 6)])
        self.assertEqual(value, {'starts': [4, 2, 4, 2]})
        for invalid in [0, 5, True, '2', 2.0]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                prep.validate_groups({'starts': [4, 2, invalid, 2]}, blocks)

    def test_unordered_duplicate_boundaries_preserve_source_and_resume_without_retries(self):
        text = ''.join(f'line {i}\n' for i in range(120))
        blocks = prep.source_blocks([{'text': text}])
        calls = []
        def complete(messages, schema):
            calls.append(messages)
            last = schema['properties']['starts']['items']['enum'][-1]
            return {'starts': [last, 1, last]}
        with tempfile.TemporaryDirectory() as root:
            chunks = prep.prepare_blocks(blocks, complete, Path(root), window_chars=60)
            resumed = prep.prepare_blocks(blocks, lambda *_: self.fail('cached call repeated'),
                                          Path(root), window_chars=60)
        self.assertGreater(len(calls), 1)
        self.assertTrue(all(len(messages) == 2 for messages in calls))
        self.assertEqual(''.join(c['text'] for c in chunks), text)
        self.assertEqual(resumed, chunks)

    def test_retries_and_resumes_without_repeating_completed_model_work(self):
        blocks = prep.source_blocks([{'text': 'First\nSecond\nThird\n'}])
        calls = []
        def complete(messages, schema):
            calls.append(messages)
            self.assertEqual(schema, prep.boundary_schema(3))
            return {'starts': [99] if len(calls) == 1 else []}
        with tempfile.TemporaryDirectory() as root:
            expected = prep.prepare_blocks(blocks, complete, Path(root))
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[1][-2]['role'], 'assistant')
            self.assertIn('99', calls[1][-2]['content'])
            self.assertIn('Validation error:', calls[1][-1]['content'])
            actual = prep.prepare_blocks(blocks, lambda *_: self.fail('cached call repeated'), Path(root))
            self.assertEqual(actual, expected)
            prep.prepare_blocks(blocks, complete, Path(root), 'Keep the full passage together.')
            self.assertEqual(len(calls), 3)

    def test_window_edges_do_not_duplicate_or_lose_content(self):
        blocks = prep.source_blocks([{'text': ''.join(f'line {i}\n' for i in range(250))}])
        calls = []
        def complete(messages, schema):
            calls.append(messages)
            supplied = messages[1]['content'].split('Document excerpt (source line numbers in brackets):\n', 1)[1]
            ids = list(range(1, len(supplied.splitlines()) + 1))
            self.assertEqual(schema['properties']['starts']['items']['enum'], ids)
            self.assertTrue(supplied.startswith('[1] '))
            return {'starts': ids}
        with tempfile.TemporaryDirectory() as root:
            chunks = prep.prepare_blocks(blocks, complete, Path(root), window_chars=60)
        self.assertEqual(''.join(c['text'] for c in chunks), ''.join(b['text'] for b in blocks))
        self.assertGreater(len(calls), 1)
        self.assertEqual([c['text'] for c in chunks], [b['text'] for b in blocks])

    def test_long_units_have_bounded_linked_parts(self):
        text = 'long equation ' * 700
        blocks = prep.source_blocks([{'text': text}])
        chunks = prep.materialize(blocks, group(blocks[-1]['id']))
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c['text']) <= prep.MAX_CHUNK_CHARS for c in chunks))
        self.assertEqual(len({c['parentId'] for c in chunks}), 1)
        self.assertEqual(''.join(c['text'] for c in chunks), text)

    def test_continuation_across_windows_preserves_a_single_parent(self):
        blocks = prep.source_blocks([{'page': 1, 'text': 'Statement\n' + 'a row with values\n' * 100},
                                     {'page': 2, 'text': 'Closing note\n'}])
        def complete(messages, schema):
            return {'starts': []}
        with tempfile.TemporaryDirectory() as root:
            chunks = prep.prepare_blocks(blocks, complete, Path(root), window_chars=500)
        self.assertEqual(len({c['parentId'] for c in chunks}), 1)
        self.assertEqual(''.join(c['text'] for c in chunks), ''.join(b['text'] for b in blocks))
        self.assertEqual(chunks[-1]['pageEnd'], 2)

    def test_same_titles_and_text_at_different_positions_do_not_merge(self):
        blocks = prep.source_blocks([{'text': 'Repeat\nBody\nRepeat\nBody\n'}])
        first = prep.materialize(blocks[:2], group(2, 'Repeat'))
        second = prep.materialize(blocks[2:], group(4, 'Repeat'))
        self.assertNotEqual(first[0]['parentId'], second[0]['parentId'])

    def test_first_window_without_headings_is_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            chunks = prep.prepare_blocks(prep.source_blocks([{'text': 'A\n'}]), lambda *_: {'starts': []}, Path(root))
        self.assertEqual(chunks[0]['text'], 'A\n')

    def test_boundary_only_output_copies_headings_from_source(self):
        blocks = prep.source_blocks([{'text': 'The actual heading\nThe body.\nNext heading\nNext body.\n'}])
        groups = prep.validate_groups({'starts': [1, 3]}, blocks)
        self.assertEqual([g['title'] for g in groups], ['The actual heading', 'Next heading'])
        self.assertEqual([g['end'] for g in groups], [2, 4])
        with self.assertRaises(ValueError):
            prep.validate_groups({'starts': [1], 'unexpected': True}, blocks)

    def test_repeated_headings_are_unambiguous_by_line_number(self):
        blocks = prep.source_blocks([{'text': 'Example\r\nFirst body\r\nExample\r\nSecond body\r\n'}])
        groups = prep.validate_groups({'starts': [1, 3]}, blocks)
        self.assertEqual([g['end'] for g in groups], [2, 4])
        self.assertEqual([g['title'] for g in groups], ['Example', 'Example'])
        with self.assertRaises(ValueError):
            prep.validate_groups({'starts': ['Example']}, blocks)

    def test_window_line_numbers_map_to_document_blocks_not_global_ids(self):
        blocks = prep.source_blocks([{'text': 'Earlier\nBody\nRepeated\nBody\nRepeated\nBody\n'}])[2:]
        groups = prep.validate_groups({'starts': [1, 3]}, blocks)
        self.assertEqual([(g['start'], g['end']) for g in groups], [(3, 4), (5, 6)])
        with self.assertRaises(ValueError):
            prep.validate_groups({'starts': [5]}, blocks)

    def test_numbering_preserves_blank_lines_encoding_and_split_long_lines(self):
        text = 'Title\r\n\r\n  α + β = 3\r\n[99] literal text\n' + 'x' * 1300
        blocks = prep.source_blocks([{'page': 2, 'text': text}])
        def complete(messages, schema):
            self.assertIn('[2] \n[3] α + β = 3', messages[1]['content'])
            self.assertIn('[4] [99] literal text', messages[1]['content'])
            self.assertEqual(schema['properties']['starts']['items']['enum'], list(range(1, len(blocks) + 1)))
            return '```json\n{"starts":[1,4]}\n```'
        with tempfile.TemporaryDirectory() as root:
            chunks = prep.prepare_blocks(blocks, complete, Path(root))
            self.assertEqual(''.join(c['text'] for c in chunks), text)
            cached = prep.prepare_blocks(blocks, lambda *_: self.fail('cache missed'), Path(root))
            self.assertEqual(chunks, cached)

    def test_old_quote_boundary_cache_is_not_reused(self):
        blocks = prep.source_blocks([{'text': 'Title\nBody\n'}])
        with tempfile.TemporaryDirectory() as root:
            cache = Path(root)
            old_prompt = {'preferences': 'Preserve complete independent source units.', 'previous_unit': None,
                          'source': 'Title\nBody'}
            prep.write_json(cache / (prep.digest([14, 'Old quote prompt', old_prompt]) + '.json'), {'starts': ['Title']})
            with patch.object(prep, 'VERSION', 14):
                prep.prepare_blocks(blocks, lambda *_: {'starts': [1]}, cache)
            calls = []
            def complete(messages, schema):
                calls.append(messages)
                return {'starts': [1]}
            prep.prepare_blocks(blocks, complete, cache)
            self.assertEqual(len(calls), 1)
            self.assertEqual(len(list(cache.glob('*.json'))), 3)

    def test_ollama_uses_integer_schema_and_remote_still_validates_boundaries(self):
        blocks = prep.source_blocks([{'text': 'Title\nBody\n'}])
        for model in [{'engine': 'ollama', 'ollamaModelName': 'test'},
                      {'engine': 'remote', 'baseUrl': 'https://example.invalid/v1',
                       'apiKey': 'test-key', 'remoteModelName': 'test'}]:
            requests = []
            def respond(request, timeout):
                requests.append(json.loads(request.data))
                content = '{"starts":[true]}' if len(requests) == 1 else '{"starts":[1]}'
                result = {'message': {'content': content}} if model['engine'] == 'ollama' else {
                    'choices': [{'message': {'content': content}}]}
                return io.BytesIO(json.dumps(result).encode())
            with self.subTest(engine=model['engine']), tempfile.TemporaryDirectory() as root, patch.object(prep.urllib.request, 'urlopen', side_effect=respond):
                chunks = prep.prepare_blocks(blocks, prep.completion_client(model), Path(root))
            self.assertEqual(len(requests), 2)
            self.assertEqual(chunks[0]['text'], 'Title\nBody\n')
            if model['engine'] == 'ollama':
                self.assertEqual(requests[0]['format'], prep.boundary_schema(2))
            else:
                self.assertNotIn('format', requests[0])

    def test_local_completion_receives_numbered_source_and_schema(self):
        calls = []
        def local(messages, schema):
            calls.append((messages, schema))
            return {'starts': [1]}
        with tempfile.TemporaryDirectory() as root:
            prep.prepare_blocks(prep.source_blocks([{'text': 'Body'}]),
                                prep.completion_client({'engine': 'python'}, local), Path(root))
        self.assertIn('[1] Body', calls[0][0][1]['content'])
        self.assertEqual(calls[0][1], prep.boundary_schema(1))

    def test_failed_model_does_not_silently_use_basic_splitting(self):
        with tempfile.TemporaryDirectory() as root, self.assertRaisesRegex(ValueError, 'three attempts'):
            prep.prepare_blocks(prep.source_blocks([{'text': 'hello'}]), lambda *_: '{}', Path(root))

    def test_unlisted_leading_text_is_a_continuation(self):
        blocks = prep.source_blocks([{'page': 1, 'text': 'A title\nFirst line'}, {'page': 2, 'text': 'Last line\nNext title'}])
        groups = prep.validate_groups({'starts': [5]}, blocks)
        self.assertEqual([g['end'] for g in groups], [4, 5])
        self.assertEqual(groups[0]['kind'], 'continuation')

    def test_groups_follow_source_order_even_when_model_openings_do_not(self):
        blocks = prep.source_blocks([{'text': 'one\ntwo\nthree\n'}])
        valid = {'starts': [1, 3]}
        self.assertEqual([g['end'] for g in prep.validate_groups(valid, blocks)], [2, 3])
        valid['starts'].reverse()
        self.assertEqual([g['end'] for g in prep.validate_groups(valid, blocks)], [2, 3])

    def test_atomic_publication_settings_and_source_page_ranges(self):
        with tempfile.TemporaryDirectory() as root:
            document_path = Path(root) / 'study.txt'
            document_path.write_text('A complete example.\nA result with its explanation.\n')
            payload = {'userDataPath': root, 'path': str(document_path), 'materialId': 'test',
                       'preparation': {'mode': 'ai', 'instructions': '', 'documentInstructions': {}},
                       'preparationModel': {'id': 'test-model', 'engine': 'ollama', 'name': 'Test'},
                       'model': {'id': 'embed', 'name': 'Test embedder'}}
            def complete(messages, schema):
                return {'starts': []}

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

    def test_basic_preparation_needs_no_generator_and_remains_basic_on_update(self):
        with tempfile.TemporaryDirectory() as root:
            document_path = Path(root) / 'study.txt'
            explanation = 'The explanation connects each step with its purpose and shows how the final result follows from the original inputs.\n'
            document_path.write_text('A complete example describes the process and its expected result in detail.\n' + explanation)
            payload = {'userDataPath': root, 'path': str(document_path), 'materialId': 'test',
                       'preparation': {'mode': 'basic', 'instructions': '', 'documentInstructions': {}},
                       'model': {'id': 'embed', 'name': 'Test embedder'}}
            with patch('python_engine.tokensmith_preparation_job.completion_client') as generator, patch.object(engine, 'resolve_embedding_provider_from_spec', return_value=('test-embed', lambda _: [1., 0.], None)):
                material = engine.index_material(payload)['material']
                payload['materialId'] = material['id']
                document_path.write_text('Updated source text describes another process and explains the resulting behavior in detail.\n' + explanation)
                updated = engine.index_material(payload)['material']
            generator.assert_not_called()
            self.assertEqual(updated['preparation']['mode'], 'basic')
            self.assertIsNone(updated.get('preparationModelName'))
            self.assertIn('Updated source text', store.dump_index(root)['chunks'][0]['chunk_text'])

    def test_gguf_job_requests_the_numbered_boundary_schema(self):
        with tempfile.TemporaryDirectory() as root:
            document_path = Path(root) / 'study.txt'
            document_path.write_text('Title\nOriginal source text.\n')
            payload = {'userDataPath': root, 'path': str(document_path), 'materialId': 'test',
                       'preparation': {'mode': 'ai', 'instructions': '', 'documentInstructions': {}},
                       'preparationModel': {'id': 'gguf', 'engine': 'python', 'path': '/test/model.gguf'},
                       'model': {'id': 'embed', 'name': 'Test embedder'}}
            with patch.object(engine, 'load_llama') as load, patch.object(engine, 'resolve_embedding_provider_from_spec', return_value=('test-embed', lambda _: [1., 0.], None)):
                load.return_value.create_chat_completion.return_value = {'choices': [{'message': {'content': '{"starts":[1]}'}}]}
                engine.index_material(payload)
                request = load.return_value.create_chat_completion.call_args.kwargs
            self.assertEqual(request['response_format'], {'type': 'json_object', 'schema': prep.boundary_schema(2)})
            self.assertIn('[1] Title\n[2] Original source text.', request['messages'][1]['content'])


if __name__ == '__main__':
    unittest.main()
