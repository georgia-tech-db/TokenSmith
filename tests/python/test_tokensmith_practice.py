"""Practice scope contracts using an actual temporary SQLite index, no live model."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from python_engine import tokensmith_engine as engine
from python_engine import tokensmith_preparation as prep
from python_engine import tokensmith_store as store


class PracticeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = self.temp.name
        for collection in ['a', 'b']:
            folder = Path(self.root) / collection
            folder.mkdir()
            for filename, text in [('one.md', '# Ancestors\n\nStore the ancestor path to propagate node splits to parents.\n'),
                                   ('two.txt', 'Pin active pages so eviction cannot invalidate a page in use.\n')]:
                (folder / filename).write_text(text + '\n' + ' '.join([text.split('\n')[-2]] * 30))
            with patch.object(engine, 'resolve_embedding_provider_from_spec', return_value=('test', lambda _: [1., 0.], None)):
                engine.index_material({'path': str(folder), 'userDataPath': self.root, 'materialId': collection,
                                       'preparation': {'mode': 'basic'}, 'model': {}})
        self.documents = engine.list_study_documents({'userDataPath': self.root})['documents']

    def sources(self, documents, **extra):
        return engine.practice_sources({'userDataPath': self.root, 'documents': documents, **extra})['sources']

    def test_lists_each_document_and_keeps_same_filename_in_different_collections(self):
        self.assertEqual(len(self.documents), 4)
        self.assertEqual(len({(d['materialId'], d['documentId']) for d in self.documents}), 4)
        self.assertEqual([d['title'] for d in self.documents].count('one.md'), 2)
        self.assertTrue(all(d['chunkCount'] > 0 for d in self.documents))

    def test_selected_document_only_and_document_rotation(self):
        for index, document in enumerate(self.documents):
            for selection in [[document], self.documents]:
                sources = self.sources(selection, questionIndex=index)
                self.assertTrue(sources)
                self.assertTrue(all(str(s['materialId']) == document['materialId'] and s['documentId'] == document['documentId'] for s in sources))

    def test_practice_selection_does_not_change_chat_activation(self):
        with store.connect(self.root) as conn:
            conn.execute('UPDATE tokensmith_collection_state SET is_active = 0')
        self.assertEqual(len(engine.list_study_documents({'userDataPath': self.root})['documents']), 4)
        self.assertTrue(self.sources([self.documents[0]]))
        with store.connect(self.root) as conn:
            self.assertEqual(conn.execute('SELECT SUM(is_active) FROM tokensmith_collection_state').fetchone()[0], 0)

    def test_unknown_empty_and_cross_collection_document_scope_are_rejected(self):
        a, b = self.documents[0], self.documents[-1]
        for documents in [[], [{'materialId': 'missing', 'documentId': a['documentId']}],
                          [{'materialId': a['materialId'], 'documentId': b['documentId']}]]:
            with self.assertRaises(ValueError):
                self.sources(documents)

    def test_used_evidence_is_not_repeated_and_exhaustion_is_explicit(self):
        selected = [self.documents[0]]
        keys = []
        for _ in range(20):
            sources = self.sources(selected, usedSourceKeys=keys)
            if not sources:
                break
            fresh = [f"{s['materialId']}:{s['documentId']}:{s.get('sourceUnitId') or s.get('chunkId') or s.get('chunkRowid')}" for s in sources]
            self.assertFalse(set(keys).intersection(fresh))
            keys.extend(fresh)
        self.assertTrue(keys)
        result = engine.practice_sources({'userDataPath': self.root, 'documents': selected, 'usedSourceKeys': keys})
        self.assertEqual(result, {'sources': [], 'reason': 'no_unused_passages'})

    def test_removed_or_unready_document_cannot_start_new_questions(self):
        with store.connect(self.root) as conn:
            conn.execute("UPDATE tokensmith_collection_state SET status = 'indexing'")
        self.assertEqual(store.study_documents(self.root), [])
        with self.assertRaises(ValueError):
            self.sources([self.documents[0]])

    def test_ai_units_expand_completely_even_when_disabled_for_chat(self):
        path = Path(self.root) / 'ai.txt'
        text = 'Ancestor paths\n' + 'Store visited parents to propagate splits upward.\n' * 100 + 'End of explanation.\n'
        path.write_text(text)
        chunks = prep.materialize(prep.source_blocks([{'page': 1, 'text': text}]),
                                  {'title': 'Ancestor paths', 'kind': 'source unit', 'reason': 'One unit'})
        with patch('python_engine.tokensmith_preparation_job.prepare_blocks', return_value=chunks), \
             patch.object(engine, 'resolve_embedding_provider_from_spec', return_value=('test', lambda _: [1., 0.], None)):
            material = engine.index_material({'path': str(path), 'userDataPath': self.root, 'materialId': 'ai',
                'preparation': {'mode': 'ai'}, 'preparationModel': {'engine': 'ollama'}, 'model': {}})['material']
        with store.connect(self.root) as conn:
            conn.execute('UPDATE tokensmith_collection_state SET is_active = 0')
        document = next(d for d in store.study_documents(self.root) if d['materialId'] == material['id'])
        sources = self.sources([document])
        self.assertEqual(len(sources), 1)
        self.assertTrue(sources[0]['sourceUnitComplete'])
        self.assertEqual(sources[0]['context'], text.strip())
        key = f"{document['materialId']}:{document['documentId']}:{sources[0]['sourceUnitId']}"
        self.assertEqual(self.sources([document], usedSourceKeys=[key]), [])


if __name__ == '__main__':
    unittest.main()
