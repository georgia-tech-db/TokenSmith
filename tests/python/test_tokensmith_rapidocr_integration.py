"""Real Rapid OCR regression. Set TOKENSMITH_RAPIDOCR_MODELS to cached recognition models."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
from PIL import Image, ImageDraw, ImageFont
from python_engine import tokensmith_engine as engine
from python_engine import tokensmith_preparation_job as preparation


@unittest.skipUnless(os.environ.get('TOKENSMITH_RAPIDOCR_MODELS'), 'Requires cached Rapid OCR models')
class RapidOcrIntegrationTests(unittest.TestCase):
    def test_image_only_pdf_uses_rapid_ocr_with_either_chunking_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'models').mkdir()
            (root / 'models' / 'rapidocr').symlink_to(Path(os.environ['TOKENSMITH_RAPIDOCR_MODELS']).resolve(), target_is_directory=True)
            pdf = root / 'scanned.pdf'
            with Image.new('RGB', (1600, 400), 'white') as image:
                draw = ImageDraw.Draw(image)
                draw.text((40, 80), 'Students learn algorithms by reading examples and practicing questions.',
                          fill='black', font=ImageFont.load_default(size=30))
                image.save(pdf, 'PDF', resolution=144)
            pages, _ = engine.extract_pdf_raw_pages_pdfium(pdf)
            self.assertFalse(any(page['text'].strip() for page in pages), 'Fixture must have no PDF text layer')
            for mode in ('basic', 'ai'):
                with self.subTest(mode=mode):
                    complete = Mock(return_value={'starts': [1]})
                    document, chunks = preparation.prepare_document(pdf, 'test',
                        {'userDataPath': str(root), 'preparation': {'mode': mode, 'ocr': 'rapid'}},
                        engine, complete, root / 'cache', lambda *args, **kwargs: None)
                    self.assertEqual(document['pageCount'], 1)
                    self.assertIn('Students learn algorithms', ' '.join(chunk['text'] for chunk in chunks))
                    self.assertTrue(all(chunk['pageStart'] == 1 for chunk in chunks))
                    if mode == 'basic':
                        complete.assert_not_called()
                    else:
                        self.assertGreater(complete.call_count, 0)
                        self.assertTrue(all('image' not in call.kwargs for call in complete.call_args_list))
