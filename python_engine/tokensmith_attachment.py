"""Bounded chat attachment extraction; no indexing, embeddings or LLM calls."""
from contextlib import closing
import json
import math
from pathlib import Path
import sys
import time
import unicodedata

MAX_BYTES = 10 * 1024 * 1024
MAX_PAGES = 20
MAX_CHARACTERS = 120_000
MAX_PIXELS = 25_000_000
_ocr_engine = None


def get_ocr_engine():
    """Load once per attachment worker, using only packaged models on the CPU."""
    global _ocr_engine
    if _ocr_engine is None:
        try:
            import rapidocr
            from rapidocr import RapidOCR
            import onnxruntime
        except ImportError as error:
            raise ValueError('OCR is unavailable. Reinstall the TokenSmith Python runtime.') from error
        onnxruntime.disable_telemetry_events()
        models = Path(rapidocr.__file__).parent / 'models'
        paths = {
            'Det.model_path': models / 'PP-OCRv6_det_small.onnx',
            'Cls.model_path': models / 'ch_ppocr_mobile_v2.0_cls_mobile.onnx',
            'Rec.model_path': models / 'PP-OCRv6_rec_small.onnx',
        }
        if not all(path.is_file() for path in paths.values()):
            raise ValueError('OCR models are missing. Reinstall the TokenSmith Python runtime.')
        # Explicit paths prevent RapidOCR from downloading models at inference time.
        _ocr_engine = RapidOCR(params={
            'Global.log_level': 'warning',
            **{key: str(path) for key, path in paths.items()},
        })
    return _ocr_engine


def ocr(image):
    result = get_ocr_engine()(image)
    return '\n'.join(result.txts or ()).strip()


def native_text_is_corrupt(text):
    """Detect substantial broken character mappings without assuming a language.

    Normal whitespace, Greek/math symbols, CJK and emoji are valid. Isolated bad
    glyphs do not force OCR; many controls/unmapped glyphs indicate a broken layer.
    """
    suspicious = sum(
        (ord(char) < 32 and char not in '\n\r\t') or 127 <= ord(char) <= 159
        or char == '\ufffd' or unicodedata.category(char) in ('Co', 'Cs')
        for char in text
    )
    return suspicious >= 3 and suspicious / max(1, len(text)) >= 0.01


def extract_attachment(path):
    path = Path(path)
    size = path.stat().st_size
    if not path.is_file() or not 0 < size <= MAX_BYTES:
        raise ValueError('Choose one non-empty file up to 10 MiB.')
    extension = path.suffix.lower()
    pages, warnings = [], []
    if extension in ('.txt', '.md', '.markdown'):
        text = path.read_text(encoding='utf-8-sig')
        if '\x00' in text:
            raise ValueError('Choose a UTF-8 text or Markdown file.')
        kind = 'text'
        pages.append({'page': 1, 'text': text.strip(), 'method': 'text'})
    elif extension in ('.png', '.jpg', '.jpeg'):
        from PIL import Image, ImageOps
        Image.MAX_IMAGE_PIXELS = MAX_PIXELS
        kind = 'image'
        with Image.open(path) as original:
            if original.format not in ('PNG', 'JPEG') or original.width * original.height > MAX_PIXELS:
                raise ValueError('Choose a PNG or JPEG image up to 25 megapixels.')
            with ImageOps.exif_transpose(original) as oriented:
                with oriented.convert('RGBA') as rgba, Image.new('RGBA', oriented.size, 'white') as white:
                    white.alpha_composite(rgba)
                    with white.convert('RGB') as image:
                        text = ocr(image)
        pages.append({'page': 1, 'text': text, 'method': 'ocr'})
    elif extension == '.pdf':
        import pypdfium2 as pdfium
        kind = 'pdf'
        with pdfium.PdfDocument(str(path)) as document:
            if not 0 < len(document) <= MAX_PAGES:
                raise ValueError('PDF attachments can contain up to 20 pages. Export just the pages you need.')
            for index in range(len(document)):
                with closing(document[index]) as page:
                    with closing(page.get_textpage()) as text_page:
                        text = (text_page.get_text_range() or '').strip()
                    images = sum(1 for _ in page.get_objects(filter=[pdfium.raw.FPDF_PAGEOBJ_IMAGE]))
                    method = 'native'
                    corrupt_native = native_text_is_corrupt(text)
                    if not text or corrupt_native or (images and len(text) < 80):
                        width, height = page.get_size()
                        scale = min(300 / 72, math.sqrt(MAX_PIXELS / max(1, width * height)))
                        with closing(page.render(scale=scale)) as bitmap:
                            image = bitmap.to_pil()
                            try:
                                text = ocr(image)
                            finally:
                                image.close()
                        method = 'ocr'
                        if corrupt_native:
                            if not text.strip() or native_text_is_corrupt(text):
                                raise ValueError(f'Page {index + 1} has unreadable PDF text and OCR could not recover it. Try a clearer PDF or image.')
                            warnings.append(f'Page {index + 1}: unreadable native PDF text was replaced using OCR.')
                    if images and method == 'native':
                        warnings.append(f'Page {index + 1}: text inside images or diagrams may be missing.')
                    if not text:
                        warnings.append(f'Page {index + 1}: no text was recognized.')
                    pages.append({'page': index + 1, 'text': text, 'method': method})
                    if sum(len(item['text']) for item in pages) > MAX_CHARACTERS:
                        raise ValueError('The file has too much text. Attach a shorter excerpt (up to 120,000 characters).')
    else:
        raise ValueError('Supported attachments: PDF, TXT, Markdown, PNG and JPEG.')
    if not any(page['text'].strip() for page in pages):
        raise ValueError('No text could be read. Try a clearer image or a PDF with readable text.')
    if sum(len(page['text']) for page in pages) > MAX_CHARACTERS:
        raise ValueError('The file has too much text. Attach a shorter excerpt (up to 120,000 characters).')
    if any(page['method'] == 'ocr' for page in pages):
        warnings.append('OCR was used. Check numbers and symbols; diagram relationships may not be captured.')
    return {'name': path.name[:255], 'kind': kind, 'sizeBytes': size, 'pages': pages, 'warnings': warnings}


if __name__ == '__main__':
    try:
        request = json.loads(sys.stdin.readline())
        started = time.perf_counter()
        attachment = extract_attachment(request['path'])
        attachment['extraction'] = {'durationMs': (time.perf_counter() - started) * 1000}
        try:
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            attachment['extraction']['peakMemoryMiB'] = peak / (1024 ** 2 if sys.platform == 'darwin' else 1024)
        except ImportError:
            pass  # Windows has no resource module; do not report invented memory data.
        print(json.dumps({'ok': True, 'attachment': attachment}, ensure_ascii=False))
    except Exception as error:
        print(json.dumps({'ok': False, 'error': str(error)}, ensure_ascii=False))
        sys.exit(1)
