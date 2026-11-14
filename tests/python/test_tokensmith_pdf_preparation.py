import base64
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from python_engine import tokensmith_engine as engine
from python_engine import tokensmith_pdf_preparation as pdf_prep
from python_engine import tokensmith_preparation as prep
from python_engine import tokensmith_preparation_job as job
from python_engine import tokensmith_store as store
from tests.python import test_tokensmith_engine as fixtures


def write_pdf(path, texts):
    fixtures.TokenSmithEngineUnitTests().write_text_pdf(path, texts)


def passage(text, continues=None):
    return {'text': text, 'continues': continues}


class PdfPreparationTests(unittest.TestCase):
    def test_ocr_entry_point_rejects_non_pdfs_before_reading_or_calling_model(self):
        for suffix in ('.md', '.MD', '.markdown', '.txt', '.png'):
            with self.subTest(suffix=suffix), patch.object(pdf_prep.pdfium, 'PdfDocument') as reader:
                complete = Mock()
                with self.assertRaisesRegex(ValueError, 'only supported for PDF'):
                    pdf_prep.prepare_pdf(Path('notes' + suffix), complete, Path('unused'))
                reader.assert_not_called()
                complete.assert_not_called()

    def test_page_continuations_keep_page_ranges_and_resume_without_model_calls(self):
        with tempfile.TemporaryDirectory() as root:
            path, cache = Path(root) / 'book.PDF', Path(root) / 'cache'
            write_pdf(path, ['Beginning', 'Middle', 'End'])
            answers = [
                {'passages': [passage('Beginning'), passage('A footnote')]},
                {'passages': [passage('Middle', 'p1-u1')]},
                {'passages': [passage('End', 'p2-u1'), passage('Next topic')]},
            ]
            calls = []
            def complete(messages, schema, *, image):
                self.assertTrue(image.startswith(b'\x89PNG\r\n\x1a\n'))
                calls.append(messages)
                return answers[len(calls) - 1]
            chunks, pages = pdf_prep.prepare_pdf(path, complete, cache)
            self.assertEqual(pages, 3)
            self.assertEqual([c['text'] for c in chunks], ['Beginning\n\nMiddle\n\nEnd\n\n', 'A footnote\n\n', 'Next topic\n\n'])
            self.assertEqual((chunks[0]['pageStart'], chunks[0]['pageEnd']), (1, 3))
            self.assertEqual(len({c['tokensmithChunkId'] for c in chunks}), 3)
            self.assertIn('p1-u1', calls[1][1]['content'])
            resumed = pdf_prep.prepare_pdf(path, Mock(side_effect=AssertionError('Repeated OCR')), cache)
            self.assertEqual(resumed, (chunks, pages))
            changed = Mock(return_value={'passages': [passage('New grouping')]})
            pdf_prep.prepare_pdf(path, changed, cache, 'Keep examples together.')
            self.assertEqual(changed.call_count, 3)

    def test_changed_page_invalidates_that_page_and_following_continuity(self):
        with tempfile.TemporaryDirectory() as root:
            path, cache = Path(root) / 'book.pdf', Path(root) / 'cache'
            write_pdf(path, ['Old beginning', 'Same end'])
            complete = Mock(side_effect=[{'passages': [passage('Old beginning')]}, {'passages': [passage('Same end', 'p1-u1')]}])
            pdf_prep.prepare_pdf(path, complete, cache)
            write_pdf(path, ['New beginning', 'Same end'])
            changed = Mock(side_effect=[{'passages': [passage('New beginning')]}, {'passages': [passage('Same end', 'p1-u1')]}])
            chunks, _ = pdf_prep.prepare_pdf(path, changed, cache)
            self.assertEqual(changed.call_count, 2)
            self.assertEqual(chunks[0]['text'], 'New beginning\n\nSame end\n\n')

    def test_model_connection_errors_do_not_trigger_format_retries(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'book.pdf'
            write_pdf(path, ['A page'])
            complete = Mock(side_effect=ValueError('This model does not accept images.'))
            with self.assertRaisesRegex(ValueError, 'does not accept images'):
                pdf_prep.prepare_pdf(path, complete, Path(root) / 'cache')
            self.assertEqual(complete.call_count, 1)

    def test_invalid_continuations_are_rejected_but_blank_pages_are_allowed(self):
        previous = [{'id': 'p1-u1'}]
        invalid = [None, [], {'passages': None}, {'passages': ['text']},
            {'passages': [passage('')]}, {'passages': [passage('text', 'p0-u1')]},
            {'passages': [passage('first'), passage('second', 'p1-u1')]},
            {'passages': [passage('text')], 'extra': 1}]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                pdf_prep.validate_page(value, previous)
        self.assertEqual(pdf_prep.validate_page({'passages': []}, previous), [])

    def test_blank_page_resets_continuity_and_duplicate_text_has_distinct_ids(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'book.pdf'
            write_pdf(path, ['Repeat', '', 'Repeat'])
            complete = Mock(side_effect=[{'passages': [passage('Repeat'), passage('Repeat')]},
                                        {'passages': []}, {'passages': [passage('Repeat')]}])
            chunks, pages = pdf_prep.prepare_pdf(path, complete, Path(root) / 'cache')
            self.assertEqual(pages, 3)
            self.assertEqual(len({c['tokensmithChunkId'] for c in chunks}), 3)
            self.assertEqual(json.loads(complete.call_args_list[2].args[0][1]['content'])['previous_page_passages'], [])


class IngestionRoutingTests(unittest.TestCase):
    def make_folder(self, root):
        folder = (Path(root) / 'materials').resolve()
        folder.mkdir()
        (folder / 'nested').mkdir()
        (folder / '.hidden').mkdir()
        text = '# Original\r\n\r\nA typo stays a tpyo.\r\n![A diagram](image.png)\r\n' + 'Source material explains a process and gives an example. ' * 4
        files = [folder / 'notes.MD', folder / 'nested' / 'more.markdown', folder / 'plain.TXT']
        for file in files:
            file.write_bytes(text.encode('utf-8'))
        (folder / '.hidden' / 'ignore.md').write_text('Hidden')
        (folder / '.ignore.txt').write_text('Hidden')
        (folder / 'image.png').write_bytes(b'not a document')
        write_pdf(folder / 'book.PDF', ['An explanation of an algorithm. ' * 8])
        return folder, files, text

    def payload(self, root, folder, mode):
        return {'userDataPath': str(Path(root) / 'data'), 'path': str(folder),
                'preparation': {'mode': mode, 'instructions': '', 'documentInstructions': {}},
                'preparationModel': {'id': 'local', 'engine': 'ollama', 'ollamaModelName': 'test'}}

    def test_mixed_ai_folder_uses_ocr_only_for_pdf_and_preserves_text_exactly(self):
        with tempfile.TemporaryDirectory() as root:
            folder, text_files, original = self.make_folder(root)
            calls = []
            def complete(messages, schema, **kwargs):
                calls.append((messages, kwargs))
                return {'passages': [passage('PDF transcription')]} if kwargs.get('image') else {'starts': [1, 3]}
            payload = self.payload(root, folder, 'ai')
            with patch.object(job, 'completion_client', return_value=complete), \
                 patch.object(engine, 'extract_pdf_raw_pages_pdfium', side_effect=AssertionError('AI used PDF text parser')), \
                 patch.object(engine, 'resolve_embedding_provider_from_spec', return_value=('test', lambda _: [1., 0.], None)):
                material = engine.index_material(payload)['material']
            self.assertTrue(material['id'])
            self.assertEqual(material['fileCount'], 4)
            self.assertEqual(sum(bool(kwargs.get('image')) for _, kwargs in calls), 1)
            self.assertEqual(len(calls), 2)  # Identical text files share their boundary cache.
            for file in text_files:
                result = job.report({**payload, 'documentPath': str(file)})
                doc = next(d for d in result['documents'] if d['path'] == str(file))
                self.assertEqual(''.join(c['text'] for c in doc['chunks']), original)
                self.assertIsNone(doc['pageCount'])
            self.assertEqual([p.name for p in engine.supported_files(folder)], ['book.PDF', 'notes.MD', 'plain.TXT', 'more.markdown'])

    def test_basic_folder_uses_no_ocr_or_preparation_model(self):
        with tempfile.TemporaryDirectory() as root:
            folder, _, _ = self.make_folder(root)
            with patch.object(job, 'completion_client') as model, patch.object(job, 'prepare_pdf') as ocr, \
                 patch.object(engine, 'resolve_embedding_provider_from_spec', return_value=('test', lambda _: [1., 0.], None)):
                material = engine.index_material(self.payload(root, folder, 'basic'))['material']
            self.assertEqual(material['preparationIssueCount'], 0)
            model.assert_not_called()
            ocr.assert_not_called()

    def test_failed_pdf_update_keeps_old_index_and_resumes_completed_pages(self):
        with tempfile.TemporaryDirectory() as root:
            file = Path(root) / 'book.pdf'
            write_pdf(file, ['Old page'])
            payload = self.payload(root, file, 'ai')
            embedding = ('test', lambda _: [1., 0.], None)
            with patch.object(job, 'completion_client', return_value=Mock(return_value={'passages': [passage('Old page')]})), \
                 patch.object(engine, 'resolve_embedding_provider_from_spec', return_value=embedding):
                engine.index_material(payload)
            before = store.dump_index(payload['userDataPath'])['chunks']
            write_pdf(file, ['Changed first page', 'New second page'])
            failing = Mock(side_effect=[{'passages': [passage('Changed first page')]}, '{}', '{}', '{}'])
            with patch.object(job, 'completion_client', return_value=failing), self.assertRaisesRegex(ValueError, 'page 2'):
                engine.index_material(payload)
            self.assertEqual(store.dump_index(payload['userDataPath'])['chunks'], before)
            resumed = Mock(return_value={'passages': [passage('New second page')]})
            with patch.object(job, 'completion_client', return_value=resumed), \
                 patch.object(engine, 'resolve_embedding_provider_from_spec', return_value=embedding):
                engine.index_material(payload)
            self.assertEqual(resumed.call_count, 1)


class ImageTransportTests(unittest.TestCase):
    def test_image_requests_use_provider_format_without_mutating_text_messages(self):
        models = [{'engine': 'ollama', 'ollamaModelName': 'test'},
                  {'engine': 'remote', 'baseUrl': 'https://example.invalid/v1', 'remoteModelName': 'test', 'apiKey': 'test'}]
        for model in models:
            with self.subTest(engine=model['engine']):
                response = {'message': {'content': 'ok'}} if model['engine'] == 'ollama' else {'choices': [{'message': {'content': 'ok'}}]}
                requests = []
                def respond(request, timeout):
                    requests.append(json.loads(request.data))
                    return io.BytesIO(json.dumps(response).encode())
                messages = [{'role': 'user', 'content': 'Read this page.'}]
                with patch.object(prep.urllib.request, 'urlopen', side_effect=respond):
                    complete = prep.completion_client(model)
                    complete(messages, {}, image=b'png')
                    complete(messages, {})
                self.assertEqual(requests[1]['messages'], messages)
                self.assertEqual(messages[0]['content'], 'Read this page.')
                attached = requests[0]['messages'][0]
                if model['engine'] == 'ollama':
                    self.assertEqual(base64.b64decode(attached['images'][0]), b'png')
                else:
                    self.assertEqual(attached['content'][1]['image_url']['url'], 'data:image/png;base64,cG5n')
                    self.assertNotIn('images', attached)

    def test_truncated_transcriptions_are_not_accepted_as_complete(self):
        models = [({'engine': 'ollama'}, {'done_reason': 'length', 'message': {'content': '{"passages":[]}'}}),
                  ({'engine': 'remote', 'apiKey': 'test', 'baseUrl': 'https://example.invalid/v1'},
                   {'choices': [{'finish_reason': 'length', 'message': {'content': '{"passages":[]}'}}]})]
        for model, response in models:
            with self.subTest(engine=model['engine']), patch.object(prep.urllib.request, 'urlopen', return_value=io.BytesIO(json.dumps(response).encode())):
                with self.assertRaisesRegex(ValueError, 'cut short'):
                    prep.completion_client(model)([{'role': 'user', 'content': 'Read'}], {}, image=b'png')
