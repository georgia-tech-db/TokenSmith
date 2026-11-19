"""PDF transcription independent of chunking, with validated per-page resume caches."""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tempfile
from threading import Event, Thread
from urllib.parse import urlparse
from contextlib import closing
from pathlib import Path

import pypdfium2 as pdfium

try:
    from tokensmith_pdf_preparation import render_page, SYSTEM
    from tokensmith_preparation import digest, read_json, write_json
except ImportError:
    from python_engine.tokensmith_pdf_preparation import render_page, SYSTEM
    from python_engine.tokensmith_preparation import digest, read_json, write_json

TRANSCRIBE = SYSTEM.split('In the same response,')[0] + '''Return JSON {"text":"transcribed Markdown"}.
Use an empty string only for a blank page. Do not reproduce blank space. Close the JSON immediately after the visible text.'''
SCHEMA = {'type': 'object', 'properties': {'text': {'type': 'string'}}, 'required': ['text'], 'additionalProperties': False}


def shared_rapid_model_dir(payload, progress=lambda message: None):
    app_root = payload.get('appUserDataPath')
    if not app_root:
        return Path(payload['userDataPath']) / 'models' / 'rapidocr'
    target = Path(app_root) / 'models' / 'rapidocr'
    target.mkdir(parents=True, exist_ok=True)
    # Only migrate files matching checksums shipped with this RapidOCR version.
    import rapidocr
    import yaml
    manifest = yaml.safe_load((Path(rapidocr.__file__).parent / 'default_models.yaml').read_text())
    hashes = {}
    def collect(node):
        if not isinstance(node, dict):
            return
        url, checksum = node.get('model_dir'), node.get('SHA256')
        if isinstance(url, str) and url.endswith('.onnx') and checksum:
            hashes.setdefault(Path(urlparse(url).path).name, set()).add(checksum)
        for value in node.values():
            collect(value)
    collect(manifest)
    candidates = [Path(payload['userDataPath']) / 'models' / 'rapidocr',
                  *map(Path, payload.get('rapidOcrLegacyPaths', [])),
                  *Path(app_root).glob('libraries/*/models/rapidocr')]
    def valid(path, checksums):
        return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() in checksums
    for directory in dict.fromkeys(candidates):
        if directory.resolve() == target.resolve():
            continue
        for source in directory.glob('*.onnx'):
            checksums = hashes.get(source.name, set())
            destination = target / source.name
            if not checksums or valid(destination, checksums) or not valid(source, checksums):
                continue
            progress(f'Reusing installed Rapid OCR model — {source.name}')
            with tempfile.NamedTemporaryFile(dir=target, suffix='.tmp', delete=False) as temporary:
                temp = Path(temporary.name)
            try:
                shutil.copyfile(source, temp)
                if valid(temp, checksums):
                    temp.replace(destination)
            finally:
                temp.unlink(missing_ok=True)
    return target


class RapidDownloadStatus(logging.Handler):
    """Forward RapidOCR's download lifecycle into the app's preparation status."""
    def __init__(self, progress):
        super().__init__()
        self.progress = progress
        self.model_name = 'recognition model'
        self.total_mb = None
        self.downloading = False
        self.last_bytes = -1

    def emit(self, record):
        message = record.getMessage()
        if message.startswith('Initiating download: '):
            self.model_name = Path(urlparse(message.split(': ', 1)[1]).path).name
            self.total_mb = None
            self.downloading = True
            self.last_bytes = -1
            self.progress(f'Downloading Rapid OCR model — {self.model_name}')
        elif message.startswith('Download size: '):
            self.total_mb = message.split(': ', 1)[1]
            self.progress(f'Downloading Rapid OCR model — {self.model_name} · {message.split(": ", 1)[1]}')
        elif message.startswith('Successfully saved to: '):
            self.downloading = False
            self.progress(f'Loading Rapid OCR model — {Path(message.split(": ", 1)[1]).name}')

    def report_bytes(self, model_dir):
        with self.lock:
            self._report_bytes(model_dir)

    def _report_bytes(self, model_dir):
        if not self.downloading:
            return
        try:
            size = (model_dir / self.model_name).stat().st_size
        except FileNotFoundError:
            return
        if size != self.last_bytes:
            self.last_bytes = size
            total = f' of {self.total_mb}' if self.total_mb else ''
            self.progress(f'Downloading Rapid OCR model — {self.model_name} · {size / 1024 / 1024:.1f} MB{total}')


def rapid_reader(model_dir: Path, progress=lambda message: None):
    progress('Loading Rapid OCR models — checking installed files')
    from rapidocr import RapidOCR
    logger = logging.getLogger('RapidOCR')
    handler = RapidDownloadStatus(progress)
    logger.addHandler(handler)
    stopped = Event()
    def monitor():
        while not stopped.wait(1):
            handler.report_bytes(model_dir)
    watcher = Thread(target=monitor, daemon=True)
    watcher.start()
    try:
        return RapidOCR(params={'Global.model_root_dir': str(model_dir)})
    finally:
        stopped.set()
        watcher.join()
        logger.removeHandler(handler)


def read_pdf(path: Path, method: str, complete, cache: Path, progress=lambda *args: None, model_dir: Path | None = None):
    if path.suffix.lower() != '.pdf' or method not in ('rapid', 'gemma'):
        raise ValueError('Choose Rapid OCR or Gemma OCR for a PDF.')
    pages, reader = [], None
    with pdfium.PdfDocument(str(path)) as pdf:
        total = len(pdf)
        progress(0, total, 'Reading PDF')
        for index in range(total):
            with closing(pdf[index]) as page:
                image = render_page(page)
            entry = cache / (digest([method, TRANSCRIBE, hashlib.sha256(image).hexdigest()]) + '.json')
            value = read_json(entry)
            cached = value is not None
            if value is None:
                if method == 'rapid':
                    if reader is None:
                        reader = rapid_reader(model_dir or cache / 'models',
                                              lambda message: progress(index, total, message))
                    progress(index, total, f'Rapid OCR — page {index + 1} of {total}')
                    result = reader(image)
                    value = {'text': '\n'.join(result.txts or ())}
                else:
                    prepare = getattr(complete, 'prepare', None)
                    if callable(prepare):
                        prepare(image=True, progress=lambda message: progress(index, total, message))
                    progress(index, total, f'Gemma OCR — page {index + 1} of {total}')
                    for attempt in range(3):
                        value = complete([{'role': 'system', 'content': TRANSCRIBE},
                                          {'role': 'user', 'content': 'Transcribe this page.'}], SCHEMA, image=image)
                        try:
                            if isinstance(value, str):
                                value = json.loads(value)
                            if not isinstance(value, dict) or set(value) != {'text'} or not isinstance(value['text'], str):
                                raise ValueError('Expected page text.')
                            break
                        except (ValueError, TypeError):
                            value = None
                    if value is None:
                        raise ValueError(f'PDF page {index + 1} could not be transcribed after three attempts.')
                write_json(entry, value)
            if cached:
                progress(index + 1, total, f'{"Rapid" if method == "rapid" else "Gemma"} OCR — page {index + 1} of {total}')
            pages.append({'page': index + 1, 'text': value['text']})
        progress(total, total, 'OCR complete')
    return pages, total
