from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from python_engine import tokensmith_attachment as attachment


def write_pdf(path, page_count=1):
    stream = b'BT /F1 14 Tf 50 700 Td (Question 3: use 10 buffers.) Tj ET'
    objects = [b'<< /Type /Catalog /Pages 2 0 R >>',
               f'<< /Type /Pages /Count {page_count} /Kids [{" ".join(f"{5+i} 0 R" for i in range(page_count))}] >>'.encode(),
               b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',
               f'<< /Length {len(stream)} >>\nstream\n'.encode() + stream + b'\nendstream']
    for _ in range(page_count):
        objects.append(b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents 4 0 R >>')
    data, offsets = b'%PDF-1.4\n', [0]
    for number, body in enumerate(objects, 1):
        offsets.append(len(data))
        data += f'{number} 0 obj\n'.encode() + body + b'\nendobj\n'
    xref = len(data)
    data += f'xref\n0 {len(offsets)}\n0000000000 65535 f \n'.encode()
    data += b''.join(f'{offset:010d} 00000 n \n'.encode() for offset in offsets[1:])
    data += f'trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n'.encode()
    path.write_bytes(data)


class ChatAttachmentTests(unittest.TestCase):
    def test_corrupt_mapping_detector_accepts_real_unicode_and_rejects_broken_glyphs(self):
        valid = 'Question 3: use 10 buffers.\n\tα = 2; ∑ x² ≤ 10.\r\n数据库 데이터 😀\n'
        for text in (valid, valid * 30, valid * 30 + '\ufffd', 'a\tb\nc\rd'):
            self.assertFalse(attachment.native_text_is_corrupt(text))
        for text in ('GDWD\x11 V\\VWHP\x0f UHODWLRQVKLSV\x0b', '\ufffd' * 4 + 'unmapped', '\ue000' * 4):
            self.assertTrue(attachment.native_text_is_corrupt(text))

    def test_only_corrupt_native_page_falls_back_to_ocr(self):
        import pypdfium2 as pdfium
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'mixed.pdf'
            write_pdf(path, 3)
            native = ['Question 3: use 10 buffers.', 'GDWD\x11 V\\VWHP\x0f UHODWLRQVKLSV\x0b filler', 'Greek α and Chinese 数据库 remain native.']
            with patch.object(pdfium.PdfTextPage, 'get_text_range', side_effect=native), \
                    patch.object(attachment, 'ocr', return_value='Question 3: use 10 buffers.') as ocr:
                result = attachment.extract_attachment(path)
            self.assertEqual(ocr.call_count, 1)
            self.assertEqual([page['method'] for page in result['pages']], ['native', 'ocr', 'native'])
            self.assertEqual(result['pages'][0]['text'], native[0])
            self.assertEqual(result['pages'][1]['text'], 'Question 3: use 10 buffers.')
            self.assertEqual(result['pages'][2]['text'], native[2])
            self.assertTrue(any('Page 2' in warning and 'replaced' in warning for warning in result['warnings']))

    def test_failed_ocr_cannot_silently_restore_corrupt_native_text(self):
        import pypdfium2 as pdfium
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'broken.pdf'
            write_pdf(path)
            with patch.object(pdfium.PdfTextPage, 'get_text_range', return_value='GDWD\x11 V\\VWHP\x0f UHODWLRQVKLSV\x0b filler'), \
                    patch.object(attachment, 'ocr', return_value=''):
                with self.assertRaisesRegex(ValueError, 'Page 1.*OCR could not recover'):
                    attachment.extract_attachment(path)

    def test_text_and_markdown_are_read_without_models_or_indexing(self):
        with tempfile.TemporaryDirectory() as directory:
            for suffix in ('.txt', '.md', '.markdown'):
                path = Path(directory) / ('question' + suffix)
                path.write_text('\ufeffQuestion 3: use 10 buffers. α = 2', encoding='utf-8')
                with patch.object(attachment, 'get_ocr_engine', side_effect=AssertionError('Unexpected model load')):
                    result = attachment.extract_attachment(path)
                self.assertEqual(result['kind'], 'text')
                self.assertEqual(result['pages'][0]['text'], 'Question 3: use 10 buffers. α = 2')
                self.assertEqual(result['pages'][0]['method'], 'text')

    def test_native_pdf_keeps_page_numbers_and_does_not_call_ocr(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'question.pdf'
            write_pdf(path, attachment.MAX_PAGES)
            with patch.object(attachment, 'ocr', side_effect=AssertionError('Unexpected OCR')):
                result = attachment.extract_attachment(path)
            self.assertEqual([page['page'] for page in result['pages']], list(range(1, attachment.MAX_PAGES + 1)))
            self.assertTrue(all(page['method'] == 'native' for page in result['pages']))
            self.assertIn('10 buffers', result['pages'][0]['text'])

    def test_pdf_page_limit_is_checked_before_ocr(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'question.pdf'
            write_pdf(path, attachment.MAX_PAGES + 1)
            with patch.object(attachment, 'ocr', side_effect=AssertionError('Unexpected OCR')):
                with self.assertRaisesRegex(ValueError, 'up to 20 pages'):
                    attachment.extract_attachment(path)

    def test_file_and_text_limits_and_empty_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'question.txt'
            for text, error in [('', 'non-empty'), ('   ', 'No text'), ('x' * (attachment.MAX_CHARACTERS + 1), 'too much text')]:
                path.write_text(text)
                with self.assertRaisesRegex(ValueError, error):
                    attachment.extract_attachment(path)
            with path.open('wb') as stream:
                stream.truncate(attachment.MAX_BYTES + 1)
            with self.assertRaisesRegex(ValueError, '10 MiB'):
                attachment.extract_attachment(path)

    def test_png_and_jpeg_use_local_ocr(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            for suffix in ('.png', '.jpg', '.jpeg'):
                path = Path(directory) / ('screenshot' + suffix)
                with Image.new('RGB', (300, 100), 'white') as image:
                    image.save(path)
                with patch.object(attachment, 'ocr', return_value='Question 3: use 10 buffers.') as ocr:
                    result = attachment.extract_attachment(path)
                self.assertEqual(ocr.call_count, 1)
                self.assertEqual(result['kind'], 'image')
                self.assertEqual(result['pages'][0]['method'], 'ocr')
                self.assertTrue(result['warnings'])

    def test_missing_packaged_models_fail_without_downloading(self):
        from rapidocr.inference_engine.onnxruntime.main import DownloadFile
        with patch.object(attachment, '_ocr_engine', None), \
                patch.object(Path, 'is_file', return_value=False), \
                patch.object(DownloadFile, 'run', side_effect=AssertionError('Unexpected download')):
            with self.assertRaisesRegex(ValueError, 'OCR models are missing'):
                attachment.get_ocr_engine()

    def test_real_ocr_reads_screenshots_and_multiple_scanned_pages_offline(self):
        from PIL import Image, ImageDraw, ImageFont
        from rapidocr.inference_engine.onnxruntime.main import DownloadFile
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(attachment, '_ocr_engine', None), \
                patch.object(DownloadFile, 'run', side_effect=AssertionError('Unexpected download')), \
                Image.new('RGB', (900, 180), 'white') as image:
            ImageDraw.Draw(image).text((30, 40), 'Question 3: use 10 buffers.', fill='black', font=ImageFont.load_default(size=32))
            engine = attachment.get_ocr_engine()
            for suffix in ('.png', '.jpg', '.pdf'):
                path = Path(directory) / ('scan' + suffix)
                if suffix == '.pdf':
                    image.save(path, resolution=150, save_all=True, append_images=[image, image])
                else:
                    image.save(path)
                result = attachment.extract_attachment(path)
                self.assertEqual(len(result['pages']), 3 if suffix == '.pdf' else 1)
                for page in result['pages']:
                    self.assertEqual(page['method'], 'ocr')
                    self.assertIn('10 buffers', page['text'])
                self.assertIs(attachment.get_ocr_engine(), engine)
            for stage in (engine.text_det, engine.text_cls, engine.text_rec):
                self.assertEqual(stage.session.session.get_providers(), ['CPUExecutionProvider'])


if __name__ == '__main__':
    unittest.main()
