import json
from pathlib import Path
import sqlite3
import stat
import tempfile
import unittest
import uuid
import zipfile
from unittest.mock import patch

from python_engine import tokensmith_engine as engine, tokensmith_store as store
from python_engine import tokensmith_library_archive as archive
from python_engine import tokensmith_preparation_job as preparation


class LibraryArchiveTests(unittest.TestCase):
    def fixture(self, root):
        library = root / 'library'
        sources = root / 'originals'
        sources.mkdir()
        (sources / 'notes.md').write_text('# Buffer pools\n\nBuffer pools keep database pages in memory. ' * 12)
        payload = {'userDataPath': str(library), 'path': str(sources), 'preparation': {'mode': 'basic'}}
        with patch.object(engine, 'resolve_embedding_provider_from_spec', return_value=('fixture', lambda _: [1., 0.], None)):
            material = engine.index_material(payload)['material']
        old = str(uuid.uuid4())
        manifest = {'format': 'tokensmith-collection', 'version': 1, 'id': old, 'name': 'Study library', 'indexRevision': 'r1', 'sourceBasePath': str(library)}
        (library / 'collection.json').write_text(json.dumps(manifest))
        path = str(sources / 'notes.md')
        (library / 'topic-inventory.json').write_text(json.dumps({'version': 3, 'collectionId': old, 'indexRevision': 'r1', 'modelId': 'fixture', 'complete': True, 'batches': {}, 'topics': [{'title': 'Buffer pools', 'sources': [{'materialId': old + '/' + material['id'], 'path': path}]}]}))
        (library / 'private-settings.json').write_text('{"apiKey":"DO-NOT-EXPORT"}')
        return library, sources, material, old

    def export(self, root):
        library, sources, material, old = self.fixture(root)
        output = root / 'abc.tokensmith'
        plan = archive.plan({'userDataPath': str(library)})
        archive.export_library({'userDataPath': str(library), 'destination': str(output), 'sources': {d['path']: d['path'] for d in plan['documents']}})
        return library, sources, material, old, output

    def test_portable_round_trip_preserves_content_embeddings_topics_and_originals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            library, sources, material, old, output = self.export(root)
            before = store.dump_index(str(library))['chunks']
            with zipfile.ZipFile(output) as z:
                self.assertNotIn('private-settings.json', z.namelist())
                self.assertNotIn('tokensmith.sqlite-wal', z.namelist())
            target = root / 'imported'; target.mkdir(); new = str(uuid.uuid4())
            archive.import_library({'archive': str(output), 'userDataPath': str(target), 'collectionId': new})
            sources.rename(root / 'originals-moved')
            result = store.list_materials(str(target))[0]
            self.assertTrue(Path(result['path']).is_dir())
            docs = archive.plan({'userDataPath': str(target)})['documents']
            self.assertEqual(len(docs), 1)
            self.assertTrue(Path(docs[0]['path']).is_relative_to(target))
            self.assertIn('Buffer pools', Path(docs[0]['path']).read_text())
            self.assertEqual([c['text'] for c in store.dump_index(str(target))['chunks']], [c['text'] for c in before])
            topics = json.loads((target / 'topic-inventory.json').read_text())
            self.assertEqual(topics['collectionId'], new)
            self.assertEqual(topics['topics'][0]['sources'][0]['materialId'], new + '/' + material['id'])
            self.assertEqual(topics['topics'][0]['sources'][0]['path'], docs[0]['path'])
            report = preparation.report({'userDataPath': str(target), 'path': result['path'], 'documentPath': docs[0]['path']})
            self.assertTrue(report['documents'][0]['chunks'])
            with sqlite3.connect(target / 'tokensmith.sqlite') as conn:
                self.assertGreater(conn.execute('SELECT count(*) FROM embeddings').fetchone()[0], 0)
            self.assertEqual(store.dump_index(str(library))['chunks'], before)
            second = root / 'second'; second.mkdir()
            archive.import_library({'archive': str(output), 'userDataPath': str(second), 'collectionId': str(uuid.uuid4())})
            self.assertNotEqual(json.loads((second / 'collection.json').read_text())['id'], new)

    def test_export_preserves_incomplete_topic_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            library, sources, material, old, output = self.export(root)
            topic_file = library / 'topic-inventory.json'
            topics = json.loads(topic_file.read_text())
            topics.update(complete=False, completed=1, total=4)
            topic_file.write_text(json.dumps(topics))
            listing = archive.plan({'userDataPath': str(library)})
            archive.export_library({'userDataPath': str(library), 'destination': str(output),
                'sources': {d['path']: d['path'] for d in listing['documents']}})
            target = root / 'partial'; target.mkdir()
            archive.import_library({'archive': str(output), 'userDataPath': str(target), 'collectionId': str(uuid.uuid4())})
            imported = json.loads((target / 'topic-inventory.json').read_text())
            self.assertFalse(imported['complete'])
            self.assertEqual(imported['completed'], 1)
            self.assertEqual(imported['total'], 4)
            self.assertEqual(len(imported['topics']), len(topics['topics']))
            self.assertEqual(json.loads(topic_file.read_text()), topics)

    def test_failed_export_preserves_previous_archive_and_missing_sources_are_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); library, sources, _, _, output = self.export(root)
            before = output.read_bytes()
            plan = archive.plan({'userDataPath': str(library)})
            (sources / 'notes.md').unlink()
            with self.assertRaisesRegex(ValueError, 'Original document is missing'):
                archive.export_library({'userDataPath': str(library), 'destination': str(output), 'sources': {d['path']: d['path'] for d in plan['documents']}})
            self.assertEqual(output.read_bytes(), before)

    def test_rejects_unsafe_paths_links_duplicates_and_oversize_archives(self):
        for bad in ['../escaped.txt', '/escaped.txt', 'documents/../../escaped', 'C:/escaped', 'documents\\escaped', 'private-settings.json']:
            with self.subTest(path=bad), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve(); target = root / 'target'; target.mkdir()
                file = root / 'bad.tokensmith'
                with zipfile.ZipFile(file, 'w') as z: z.writestr(bad, 'bad')
                with self.assertRaises(ValueError): archive.import_library({'archive': str(file), 'userDataPath': str(target), 'collectionId': 'new'})
                self.assertFalse((root / 'escaped.txt').exists())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); target = root / 'target'; target.mkdir(); file = root / 'bad.tokensmith'
            link = zipfile.ZipInfo('documents/link'); link.create_system = 3; link.external_attr = (stat.S_IFLNK | 0o777) << 16
            with zipfile.ZipFile(file, 'w') as z: z.writestr(link, '/etc/passwd')
            with self.assertRaisesRegex(ValueError, 'linked'): archive.import_library({'archive': str(file), 'userDataPath': str(target), 'collectionId': 'new'})
            with zipfile.ZipFile(file, 'w') as z:
                z.writestr('documents/A.txt', 'one'); z.writestr('documents/a.txt', 'two')
            with self.assertRaisesRegex(ValueError, 'duplicate'): archive.import_library({'archive': str(file), 'userDataPath': str(target), 'collectionId': 'new'})
            with patch.object(archive, 'MAX_BYTES', 1), self.assertRaisesRegex(ValueError, 'too large'):
                archive.import_library({'archive': str(file), 'userDataPath': str(target), 'collectionId': 'new'})

    def test_rejects_damaged_archive_and_does_not_overwrite_nonempty_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); target = root / 'target'; target.mkdir(); file = root / 'bad.tokensmith'; file.write_text('not a zip')
            with self.assertRaisesRegex(ValueError, 'damaged'): archive.import_library({'archive': str(file), 'userDataPath': str(target), 'collectionId': 'new'})
            (target / 'keep.txt').write_text('keep')
            with self.assertRaisesRegex(ValueError, 'empty'): archive.import_library({'archive': str(file), 'userDataPath': str(target), 'collectionId': 'new'})
            self.assertEqual((target / 'keep.txt').read_text(), 'keep')
