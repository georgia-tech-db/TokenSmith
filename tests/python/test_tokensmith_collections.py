import sqlite3
import tempfile
import unittest
from pathlib import Path
from python_engine import tokensmith_collections as collections
from python_engine import tokensmith_store as store


class CollectionValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.payload = {'userDataPath': str(self.root)}

    def test_empty_collection_creation_and_read_only_validation(self):
        collections.create(self.payload)
        database = self.root / store.DB_NAME
        before = database.read_bytes()
        self.assertEqual(collections.validate(self.payload), {'ok': True})
        self.assertEqual(database.read_bytes(), before)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            collections.create(self.payload)
        self.assertEqual(database.read_bytes(), before)

    def test_missing_and_corrupt_indexes_are_rejected_without_initializing_them(self):
        with self.assertRaisesRegex(ValueError, 'missing'):
            collections.validate(self.payload)
        self.assertFalse((self.root / store.DB_NAME).exists())
        (self.root / store.DB_NAME).write_text('not a sqlite file')
        with self.assertRaises(sqlite3.DatabaseError):
            collections.validate(self.payload)
        self.assertEqual((self.root / store.DB_NAME).read_text(), 'not a sqlite file')

    def test_unsupported_schema_is_not_upgraded_by_loading(self):
        collections.create(self.payload)
        with store.connect(str(self.root)) as conn:
            store.set_schema_value(conn, 'version', '999')
        with self.assertRaisesRegex(ValueError, 'unsupported'):
            collections.validate(self.payload)
        with sqlite3.connect(self.root / store.DB_NAME) as conn:
            self.assertEqual(conn.execute("SELECT value FROM schema_info WHERE key='version'").fetchone()[0], '999')

    def test_optional_structures_do_not_participate_in_validation(self):
        collections.create(self.payload)
        (self.root / 'graph.json').write_bytes(b'opaque future graph data')
        (self.root / 'topics.json').write_text('malformed optional topics')
        self.assertTrue(collections.validate(self.payload)['ok'])
        self.assertEqual((self.root / 'graph.json').read_bytes(), b'opaque future graph data')

    def test_incomplete_database_is_rejected(self):
        collections.create(self.payload)
        with sqlite3.connect(self.root / store.DB_NAME) as conn:
            conn.execute('DROP TABLE embeddings')
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            collections.validate(self.payload)


if __name__ == '__main__':
    unittest.main()
