"""Storage/selection contracts; model quality is tested separately on real documents."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from python_engine import tokensmith_engine as engine
from python_engine import tokensmith_preparation as prep
from python_engine import tokensmith_store as store


class SourceUnitTests(unittest.TestCase):
    def index(self, root, folder, text):
        path = Path(root) / folder / 'study.txt'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        blocks = prep.source_blocks([{'page': 1, 'text': text}])
        chunks = prep.materialize(blocks, {'title': 'Shared title', 'kind': 'source unit', 'reason': 'One unit'})
        with patch('python_engine.tokensmith_preparation_job.prepare_blocks', return_value=chunks), \
             patch.object(engine, 'resolve_embedding_provider_from_spec', return_value=('test', lambda _: [1., 0.], None)):
            return engine.index_material({'path': str(path), 'userDataPath': root, 'materialId': folder,
                'preparation': {'mode': 'ai'}, 'preparationModel': {'engine': 'ollama'}, 'model': {}})['material']

    def rows(self, root):
        with store.connect(root) as conn:
            return [dict(row) for row in conn.execute(store.source_row_select() + ' ORDER BY ch.id')]

    def test_all_document_types_retain_headers_body_and_conclusion(self):
        units = {
            'filing': ('USD millions; 2024 2025\n', 'Revenue 12 15\n', 'Amounts are consolidated.\n'),
            'poem': ('Two Windows\n', 'We hid the lights we hoped to see.\n', 'Neither saw the other.\n'),
            'calculus': ('Theorem\nAssume a > 0.\n', 'Transform both sides equally.\n', 'Therefore the identity holds.\n'),
            'paper': ('Method\nBaseline uses labeled examples.\n', 'Apply inverse transformations.\n', 'Labels follow from the transformation.\n'),
            'slides': ('Normalization\n', 'Fold constant expressions.\n', 'Preserve the query result.\n')
        }
        for genre, (header, body, end) in units.items():
            with self.subTest(genre=genre), tempfile.TemporaryDirectory() as root:
                text = header + body * 170 + end
                material = self.index(root, genre, text)
                rows = self.rows(root)
                self.assertGreater(len(rows), 1)
                self.assertEqual(len({r['parent_id'] for r in rows}), 1)
                hit = {**rows[1], 'score': .7, 'query_embedding_model': 'live-query-model'}
                result = store.expand_source_units(root, [hit, rows[0]], [material['id']])
                self.assertEqual(len(result), 1)
                self.assertEqual(result[0]['text'], text)
                self.assertTrue(result[0]['unit_complete'])
                self.assertEqual(result[0]['query_embedding_model'], 'live-query-model')
                self.assertEqual(len(result[0]['source_chunk_ids']), len(rows))
                source = engine.source_from_sqlite_chunk(result[0], ['normalization'])
                self.assertTrue(source['sourceUnitComplete'])
                self.assertEqual(source['lineTo'], rows[-1]['line_to'])

    def test_starter_sources_expand_the_same_units_as_chat_search(self):
        with tempfile.TemporaryDirectory() as root:
            material = self.index(root, 'starter', 'Opening\n' + 'Related explanation\n' * 170 + 'Conclusion\n')
            result = engine.starter_sources({'userDataPath': root, 'materials': [material], 'limit': 4})
            self.assertEqual(len(result['sources']), 1)
            self.assertTrue(result['sources'][0]['sourceUnitComplete'])
            self.assertIn('Conclusion', result['sources'][0]['context'])

    def test_unit_identity_is_scoped_to_collection_and_document(self):
        with tempfile.TemporaryDirectory() as root:
            text = 'Shared title\n' + 'identical text\n' * 250
            a = self.index(root, 'a', text)
            b = self.index(root, 'b', text)
            rows = self.rows(root)
            hits = [next(r for r in rows if r['material_id'] == m['id']) for m in [a, b]]
            result = store.expand_source_units(root, hits, [a['id'], b['id']])
            self.assertEqual(len(result), 2)
            self.assertNotEqual(result[0]['document_id'], result[1]['document_id'])
            self.assertEqual(len(store.expand_source_units(root, hits, [a['id']])), 1)

    def test_large_and_incomplete_units_do_not_claim_complete_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            material = self.index(root, 'long', 'A long unit\n' + 'body\n' * 700)
            rows = self.rows(root)
            result = store.expand_source_units(root, [rows[0]], [material['id']], max_chars=100)
            self.assertFalse(result[0]['unit_complete'])
            with store.connect(root) as conn:
                conn.execute('UPDATE chunks SET unit_parts = 999')
            self.assertFalse(store.expand_source_units(root, [rows[0]], [material['id']])[0]['unit_complete'])

    def test_three_relevant_parts_are_not_replaced_by_an_unrelated_section(self):
        rows = [{'rowid': i, 'material_id': '1', 'document_id': 1, 'section_header': 'Shared',
                 'text': 'derived evidence', 'score': 1.0} for i in range(1, 4)]
        rows.append({'rowid': 4, 'section_header': 'Other', 'text': 'unrelated'})
        selected = engine.select_source_rows(rows, ['derived', 'evidence'], ['derived'], 3)
        self.assertEqual([r['rowid'] for r in selected], [1, 2, 3])

    def test_legacy_rows_survive_migration_without_invented_parent_links(self):
        with tempfile.TemporaryDirectory() as root:
            self.index(root, 'old', 'Unchanged text\n')
            with store.connect(root) as conn:
                conn.execute('DROP INDEX idx_chunks_parent')
                for column in ['parent_id', 'unit_part', 'unit_parts']:
                    conn.execute(f'ALTER TABLE chunks DROP COLUMN {column}')
            store.init_db(root)
            rows = self.rows(root)
            self.assertEqual(rows[0]['text'], 'Unchanged text\n')
            self.assertIsNone(rows[0]['parent_id'])


if __name__ == '__main__':
    unittest.main()
