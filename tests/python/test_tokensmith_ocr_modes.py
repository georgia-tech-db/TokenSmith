"""OCR selection and chunking must remain independent, including for mixed folders."""
import logging
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from python_engine import tokensmith_engine as engine
from python_engine import tokensmith_preparation_job as job
from python_engine import tokensmith_ocr as ocr
from tests.python.test_tokensmith_pdf_preparation import write_pdf

TEXT = 'Original source wording explains how a student can understand this example. ' * 3

class OcrModesTests(unittest.TestCase):
    def test_pdf_mode_matrix(self):
        for method in ('none', 'rapid', 'gemma'):
            for mode in ('basic', 'ai'):
                with self.subTest(ocr=method, mode=mode), tempfile.TemporaryDirectory() as root:
                    path = Path(root) / 'book.pdf'
                    write_pdf(path, [TEXT * 3])
                    complete = Mock(return_value={'starts': [1]})
                    payload = {'userDataPath': root, 'preparation': {'mode': mode, 'ocr': method}}
                    with patch.object(job, 'read_pdf', return_value=([{'page': 1, 'text': TEXT}], 1)) as read:
                        doc, chunks = job.prepare_document(path, 'm', payload, engine, complete, Path(root) / 'cache', Mock())
                    self.assertEqual(doc['pageCount'], 1)
                    self.assertTrue(chunks)
                    self.assertEqual(read.call_count, int(method != 'none'))
                    self.assertEqual(complete.call_count, int(mode == 'ai'))
                    if method != 'none':
                        self.assertEqual(read.call_args.args[1], method)

    def test_markdown_never_uses_ocr(self):
        for method in ('none', 'rapid', 'gemma'):
            for mode in ('basic', 'ai'):
                with self.subTest(ocr=method, mode=mode), tempfile.TemporaryDirectory() as root:
                    path = Path(root) / 'notes.md'; path.write_text(TEXT)
                    with patch.object(job, 'read_pdf', side_effect=AssertionError('Markdown entered OCR')):
                        _, chunks = job.prepare_document(path, 'm', {'preparation': {'mode': mode, 'ocr': method}},
                            engine, Mock(return_value={'starts': [1]}), Path(root) / 'cache', Mock())
                    self.assertIn(TEXT.strip(), chunks[0]['text'])

    def test_rapid_and_gemma_caches_are_separate_and_resume(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'book.pdf'; write_pdf(path, [TEXT])
            cache = Path(root) / 'cache'
            reader = Mock(return_value=SimpleNamespace(txts=(TEXT,)))
            complete = Mock(return_value={'text': TEXT})
            with patch.object(ocr, 'rapid_reader', return_value=reader):
                rapid, _ = ocr.read_pdf(path, 'rapid', complete, cache)
                ocr.read_pdf(path, 'rapid', complete, cache)
                complete.assert_not_called()
                self.assertEqual(reader.call_count, 1)
                gemma, _ = ocr.read_pdf(path, 'gemma', complete, cache)
                ocr.read_pdf(path, 'gemma', complete, cache)
                self.assertEqual(complete.call_count, 1)
                self.assertEqual(rapid, gemma)

    def test_default_pdf_llm_chunking_never_renders_images(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'book.pdf'; write_pdf(path, [TEXT * 3])
            with patch.object(ocr, 'render_page', side_effect=AssertionError('Unexpected OCR')):
                _, chunks = job.prepare_document(path, 'm', {'preparation': {'mode': 'ai'}}, engine,
                    Mock(return_value={'starts': [1]}), Path(root) / 'cache', Mock())
            self.assertTrue(chunks)

    def test_scanned_pdf_without_ocr_explains_how_to_resume(self):
        for mode in ('basic', 'ai'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as root:
                path = Path(root) / 'scan.pdf'; write_pdf(path, [''])
                with self.assertRaisesRegex(ValueError, 'This PDF needs OCR.*Set OCR to Yes.*resume preparation'):
                    job.prepare_document(path, 'm', {'preparation': {'mode': mode, 'ocr': 'none'}},
                        engine, Mock(), Path(root) / 'cache', Mock())


class RapidDownloadStatusTests(unittest.TestCase):
    def test_download_status_names_model_size_and_loading(self):
        messages = []
        handler = ocr.RapidDownloadStatus(messages.append)
        for text in ['Initiating download: https://example.test/models/recognition.onnx',
                     'Download size: 20.25MB', 'Successfully saved to: /tmp/recognition.onnx']:
            handler.emit(logging.LogRecord('RapidOCR', logging.INFO, '', 0, text, (), None))
        self.assertEqual(messages, ['Downloading Rapid OCR model — recognition.onnx',
                                   'Downloading Rapid OCR model — recognition.onnx · 20.25MB',
                                   'Loading Rapid OCR model — recognition.onnx'])

    def test_page_ocr_is_reported_only_after_model_setup(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'book.pdf'; write_pdf(path, [TEXT])
            messages = []
            def load_reader(model_dir, progress):
                progress('Downloading Rapid OCR model — recognition.onnx')
                return Mock(return_value=SimpleNamespace(txts=(TEXT,)))
            with patch.object(ocr, 'rapid_reader', side_effect=load_reader):
                ocr.read_pdf(path, 'rapid', Mock(), Path(root) / 'cache',
                             lambda done, total, message: messages.append(message))
            download = messages.index('Downloading Rapid OCR model — recognition.onnx')
            page = messages.index('Rapid OCR — page 1 of 1')
            self.assertLess(download, page)
            self.assertEqual(messages[-1], 'OCR complete')


class StageProgressTests(unittest.TestCase):
    def test_ocr_and_chunking_have_independent_counts(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'book.pdf'; write_pdf(path, [TEXT, TEXT])
            events = []
            def read(*args, **kwargs):
                callback = args[4]
                callback(0, 2, 'Downloading Rapid OCR model — test.onnx')
                callback(0, 2, 'Rapid OCR — page 1 of 2')
                callback(1, 2, 'Rapid OCR — page 2 of 2')
                callback(2, 2, 'OCR complete')
                return [{'page': 1, 'text': TEXT}, {'page': 2, 'text': TEXT}], 2
            with patch.object(job, 'read_pdf', side_effect=read):
                job.prepare_document(path, 'm', {'userDataPath':root,'preparation':{'mode':'basic','ocr':'rapid'}},
                    engine, Mock(), Path(root)/'cache', lambda *args, **kwargs: events.append(kwargs))
            self.assertEqual(events[0]['stage'], 'Model setup')
            self.assertEqual([(e['stageCompleted'],e['stageTotal']) for e in events if e['stage']=='OCR'],[(0,2),(1,2),(2,2)])
            self.assertEqual([(e['stageCompleted'],e['stageTotal']) for e in events if e['stage']=='Chunking'],[(0,1),(1,1)])

    def test_estimate_excludes_setup_and_resets_for_chunking(self):
        clock = {'name':None,'started':0,'initial':0}
        with patch.object(job,'monotonic',return_value=10):
            self.assertEqual(job.stage_timing(clock,{'stage':'Model setup'}),{})
        with patch.object(job,'monotonic',return_value=110):
            value=job.stage_timing(clock,{'stage':'OCR','stageCompleted':0,'stageTotal':10})
            self.assertNotIn('estimatedRemainingMs',value)
        with patch.object(job,'monotonic',return_value=120):
            value=job.stage_timing(clock,{'stage':'OCR','stageCompleted':2,'stageTotal':10})
            self.assertEqual(value['stagePercent'],20)
            self.assertEqual(value['estimatedRemainingMs'],40000)
            value=job.stage_timing(clock,{'stage':'Chunking','stageCompleted':0,'stageTotal':5})
            self.assertEqual(value['stagePercent'],0)
            self.assertNotIn('estimatedRemainingMs',value)


class SharedRapidCacheTests(unittest.TestCase):
    def test_two_libraries_reuse_verified_models_and_ignore_partial_downloads(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); app=root/'app'; first=app/'libraries'/'first'; second=app/'libraries'/'second'
            legacy=first/'models'/'rapidocr'; legacy.mkdir(parents=True)
            good=b'verified model'; (legacy/'recognition.onnx').write_bytes(good)
            (legacy/'partial.onnx').write_bytes(b'incomplete')
            manifest={'one':{'model_dir':'https://example.test/recognition.onnx','SHA256':hashlib.sha256(good).hexdigest()},
                      'two':{'model_dir':'https://example.test/partial.onnx','SHA256':hashlib.sha256(b'complete').hexdigest()}}
            (root/'default_models.yaml').write_text(json.dumps(manifest))
            with patch('rapidocr.__file__', str(root/'__init__.py')):
                target=ocr.shared_rapid_model_dir({'appUserDataPath':str(app),'userDataPath':str(first)})
                target2=ocr.shared_rapid_model_dir({'appUserDataPath':str(app),'userDataPath':str(second)})
            self.assertEqual(target,target2)
            self.assertEqual((target/'recognition.onnx').read_bytes(),good)
            self.assertFalse((target/'partial.onnx').exists())
            self.assertFalse((second/'models').exists())
            self.assertTrue((legacy/'recognition.onnx').exists())

    def test_download_bytes_update_without_page_ocr_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); messages=[]; handler=ocr.RapidDownloadStatus(messages.append)
            for text in ['Initiating download: https://example.test/model.onnx','Download size: 2.00MB']:
                handler.emit(logging.LogRecord('RapidOCR',logging.INFO,'',0,text,(),None))
            (root/'model.onnx').write_bytes(b'x'*1024*1024)
            handler.report_bytes(root)
            self.assertIn('1.0 MB of 2.00MB',messages[-1])
            count=len(messages); handler.report_bytes(root); self.assertEqual(len(messages),count)
            self.assertNotIn('page 1',' '.join(messages))

class GemmaLoadingTests(unittest.TestCase):
    def test_unloaded_model_reports_loading_and_reuses_warm_context(self):
        import io
        from python_engine import tokensmith_preparation as prep
        complete=prep.completion_client({'engine':'ollama','ollamaModelName':'gemma4:e4b','name':'Gemma 4 E4B'})
        messages=[]
        with patch.object(prep.urllib.request,'urlopen',side_effect=[io.StringIO('{"models":[]}'),io.StringIO('{"done":true}')]) as request:
            complete.prepare(image=True,progress=messages.append)
            complete.prepare(image=True,progress=messages.append)
            self.assertEqual(request.call_count,2)
            body=json.loads(request.call_args.args[0].data)
            self.assertEqual(body['options']['num_ctx'],16384)
            self.assertEqual(body['messages'],[])
        self.assertEqual(messages,['Loading Gemma 4 E4B for OCR'])

    def test_loaded_model_skips_loading_request_and_message(self):
        import io
        from python_engine import tokensmith_preparation as prep
        complete=prep.completion_client({'engine':'ollama','ollamaModelName':'gemma4:e4b'})
        messages=[]
        with patch.object(prep.urllib.request,'urlopen',return_value=io.StringIO('{"models":[{"name":"gemma4:e4b","context_length":16384}]}')) as request:
            complete.prepare(image=True,progress=messages.append)
            self.assertEqual(request.call_count,1)
        self.assertEqual(messages,[])

    def test_load_failure_does_not_mark_model_ready(self):
        import io
        from python_engine import tokensmith_preparation as prep
        complete=prep.completion_client({'engine':'ollama','ollamaModelName':'gemma4:e4b'})
        with patch.object(prep.urllib.request,'urlopen',side_effect=[io.StringIO('{"models":[]}'),OSError('offline')]):
            with self.assertRaisesRegex(ValueError,'Could not load the OCR model'):
                complete.prepare(image=True)


class AdaptiveEstimateTests(unittest.TestCase):
    def test_estimate_adapts_to_faster_and_slower_pages(self):
        clock={'name':None}
        def update(now,done):
            with patch.object(job,'monotonic',return_value=now):
                return job.stage_timing(clock,{'stage':'OCR','stageCompleted':done,'stageTotal':10})
        update(0,0)
        self.assertEqual(update(10,1)['estimatedRemainingMs'],90000)
        # The second page is twice as slow: the projected per-page time rises to 14 seconds.
        self.assertEqual(update(30,2)['estimatedRemainingMs'],112000)
        # The third page is faster, so the estimate responds before the stage ends.
        self.assertEqual(update(35,3)['estimatedRemainingMs'],72800)
        rate=clock['unit_ms']
        self.assertEqual(update(36,3)['estimatedRemainingMs'],71800)
        self.assertEqual(clock['unit_ms'],rate,'Repeated status messages must not change measured speed')
        self.assertEqual(update(90,10)['estimatedRemainingMs'],0)

    def test_resuming_at_nonzero_progress_measures_only_remaining_work(self):
        clock={'name':None}
        with patch.object(job,'monotonic',return_value=100):
            self.assertNotIn('estimatedRemainingMs',job.stage_timing(clock,{'stage':'Chunking','stageCompleted':5,'stageTotal':10}))
        with patch.object(job,'monotonic',return_value=110):
            self.assertEqual(job.stage_timing(clock,{'stage':'Chunking','stageCompleted':6,'stageTotal':10})['estimatedRemainingMs'],40000)


class StableOcrPrefixTests(unittest.TestCase):
    def test_reading_prefix_does_not_reappear_between_pages(self):
        for method in ('rapid','gemma'):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                root=Path(directory); path=root/'book.pdf'; write_pdf(path,[TEXT,TEXT+' Second page.'])
                messages=[]
                with patch.object(ocr,'rapid_reader',return_value=Mock(return_value=SimpleNamespace(txts=(TEXT,)))):
                    ocr.read_pdf(path,method,lambda *args,**kwargs:{'text':TEXT},root/'cache',
                                 lambda done,total,message:messages.append(message))
                self.assertEqual(messages[0],'Reading PDF')
                self.assertEqual(sum(message.startswith('Reading') for message in messages),1)
                prefix='Rapid' if method=='rapid' else 'Gemma'
                self.assertEqual(messages[1:-1],[f'{prefix} OCR — page 1 of 2',f'{prefix} OCR — page 2 of 2'])
                self.assertEqual(messages[-1],'OCR complete')
