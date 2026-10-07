"""Real SQLite/FTS and FAISS scope tests with deterministic, deliberately adversarial vectors."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from python_engine import tokensmith_engine as engine
from python_engine import tokensmith_store as store


class StudyScopeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = self.temp.name
        folder = Path(self.root) / 'course'
        folder.mkdir()
        (folder / 'selected.md').write_text('# Ancestors\n\n' +
            'Selected evidence: ancestor paths record parents so insertion can propagate splits upward. ' * 30)
        (folder / 'excluded.md').write_text('\n\n'.join(
            f'# Section {index}\n\n' + 'Ancestor paths in photosynthesis are excluded from this study selection. ' * 30
            for index in range(40)))
        with patch.object(engine, 'resolve_embedding_provider_from_spec',
                          return_value=('test', lambda text: [0., 1.] if 'Selected evidence' in text else [1., 0.], None)):
            self.material = engine.index_material({'path': str(folder), 'userDataPath': self.root, 'materialId': 'course',
                'preparation': {'mode': 'basic'}, 'model': {}})['material']
        self.documents = store.study_documents(self.root)
        self.ids = [self.material['id']]
        self.selected = [next(d for d in self.documents if d['title'] == 'selected.md')]
        self.excluded = [next(d for d in self.documents if d['title'] == 'excluded.md')]

    def assert_scope(self, sources, documents):
        self.assertTrue(sources)
        allowed = {(d['materialId'], d['documentId']) for d in documents}
        self.assertTrue(all((str(s['materialId']), s['documentId']) in allowed for s in sources))

    def search(self, documents, **extra):
        with patch.object(engine, 'resolve_embedding_provider_for_key', return_value=(lambda _: [1., 0.], None)):
            return engine.search_library({'userDataPath': self.root, 'materials': [self.material],
                'documents': documents, 'query': 'ancestor paths', 'searchMode': 'hybrid', 'limit': 2, **extra})

    def test_faiss_filters_before_top_k_not_after_global_candidates(self):
        global_hits = store.vector_search(self.root, [1., 0.], self.ids, 20, 'test')
        self.assertEqual(len(global_hits), 20)
        global_rows = store.get_chunks_by_rowids(self.root, [rowid for rowid, _ in global_hits], self.ids)
        self.assertTrue(all(row['document_id'] == self.excluded[0]['documentId'] for row in global_rows))
        hits = store.vector_search(self.root, [1., 0.], self.ids, 1, 'test', self.selected)
        self.assertEqual(len(hits), 1)
        rows = store.get_chunks_by_rowids(self.root, [hits[0][0]], self.ids, self.selected)
        self.assertEqual(rows[0]['document_id'], self.selected[0]['documentId'])

    def test_hybrid_and_fts_return_only_selected_document(self):
        for mode in ['hybrid', 'vector', 'keyword']:
            self.assert_scope(self.search(self.selected, searchMode=mode)['sources'], self.selected)

    def test_explicit_scope_is_independent_of_legacy_global_activation(self):
        with store.connect(self.root) as conn:
            conn.execute('UPDATE tokensmith_collection_state SET is_active = 0')
        self.assert_scope(self.search(self.selected)['sources'], self.selected)
        self.assert_scope(engine.starter_sources({'userDataPath': self.root, 'documents': self.selected})['sources'], self.selected)
        with store.connect(self.root) as conn:
            self.assertEqual(conn.execute('SELECT SUM(is_active) FROM tokensmith_collection_state').fetchone()[0], 0)

    def test_empty_or_invalid_scope_never_falls_back_to_whole_library(self):
        self.assertEqual(self.search([]), {
            'sources': [], 'reason': 'no_selected_documents', 'latencySpans': []
        })
        self.assertEqual(engine.starter_sources({'userDataPath': self.root, 'documents': []}),
                         {'sources': [], 'reason': 'no_selected_documents'})
        for selected in [[{'materialId': 'other', 'documentId': self.selected[0]['documentId']}],
                         [{'materialId': self.ids[0], 'documentId': -1}]]:
            with self.assertRaises(ValueError):
                self.search(selected)

    def test_keyword_correction_uses_only_selected_documents_vocabulary(self):
        chosen = store.keyword_terms_for_query(self.root, 'photosynthesus', self.ids, documents=self.selected)
        excluded = store.keyword_terms_for_query(self.root, 'photosynthesus', self.ids, documents=self.excluded)
        self.assertNotIn('photosynthesis', chosen)
        self.assertIn('photosynthesis', excluded)

    def test_starters_cover_selected_documents_and_expansion_rejects_excluded_hits(self):
        sources = engine.starter_sources({'userDataPath': self.root, 'documents': self.documents, 'limit': 2})['sources']
        self.assertEqual({s['documentId'] for s in sources}, {d['documentId'] for d in self.documents})
        rows = store.practice_source_rows(self.root, self.documents)
        expanded = store.expand_source_units(self.root, rows, self.ids, documents=self.selected)
        self.assertTrue(expanded)
        self.assertTrue(all(row['document_id'] == self.selected[0]['documentId'] for row in expanded))


if __name__ == '__main__':
    unittest.main()
