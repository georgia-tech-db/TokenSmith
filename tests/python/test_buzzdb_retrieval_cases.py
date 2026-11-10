import itertools
import unittest
from unittest.mock import patch

from tests.benchmarks import buzzdb_retrieval as benchmark


class ReviewedBuzzdbCasesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases, cls.sources = benchmark.load()

    def selected(self, chunks):
        return [{'document':'buzzdb','chunk_id':cid,'text':self.sources['buzzdb',cid]} for cid in chunks]

    def test_reviewed_inventory_and_evidence_sets(self):
        base = [c for c in self.cases.values() if not c.get('pairedWith')]
        self.assertEqual((len(base),len(self.cases)-len(base)),(36,10))
        multi = [c['evaluationProfile']['evidenceStructure'] for c in base
                 if c['evaluationProfile']['evidenceStructure']['minimumKnownChunks']>1]
        self.assertEqual(sum(s['minimumKnownChapters']==1 for s in multi),5)
        self.assertEqual(sum(s['minimumKnownChapters']>1 for s in multi),5)
        for case in self.cases.values():
            for support in benchmark.minimum_supports(case):
                with self.subTest(case=case['id'],chunks=sorted(support)):
                    self.assertTrue(benchmark.score_evidence(case,self.selected(support))['complete'])

    def test_premise_correction_and_new_codec_question(self):
        self.assertTrue(benchmark.score_evidence(self.cases['b05'],self.selected(['ch13.042']))['complete'])
        codec = self.cases['same_bitpacked_slice_alignment']
        for encoder in ['ch15.055','ch15.064']:
            self.assertTrue(benchmark.score_evidence(codec,self.selected([encoder,'ch15.056']))['complete'])
            self.assertFalse(benchmark.score_evidence(codec,self.selected([encoder]))['complete'])
        self.assertFalse(benchmark.score_evidence(codec,self.selected(['ch15.056']))['complete'])
        # Matching source IDs cannot replace evidence removed from the context.
        selected=self.selected(['ch15.055','ch15.056'])
        selected[1]['text']='The reader fetches some compressed data.'
        self.assertFalse(benchmark.score_evidence(codec,selected)['complete'])

    def test_alternative_paths_cannot_mix_incomplete_components(self):
        paths=[{'a','b'},{'c','d'},{'e','f','g'}]
        case={'answerability':'answerable','groups':[{'id':'complete-answer','required':True,
              'alternatives':[{'id':str(i),'passages':[{'sourceId':'fixture','chunkId':cid,'quotes':[cid]}
                              for cid in sorted(path)]} for i,path in enumerate(paths)]}]}
        for count in range(8):
            for chosen in itertools.combinations('abcdefg',count):
                selected=[{'document':'fixture','chunk_id':cid,'text':cid} for cid in chosen]
                self.assertEqual(benchmark.score_evidence(case,selected)['complete'],
                                 any(path<=set(chosen) for path in paths))

    def test_background_variants_share_the_base_evidence(self):
        for case in self.cases.values():
            if not case.get('pairedWith'):
                continue
            original = self.cases[case['pairedWith']]
            self.assertEqual(case['question'], case['background'] + '\n\n' + original['question'])
            self.assertIs(case['groups'], original['groups'])

    def test_review_rejects_wrong_source_and_missing_quote(self):
        import copy
        case = copy.deepcopy(self.cases['code_hash_missing_sentinel'])
        citation = case['groups'][0]['alternatives'][0]['passages'][0]
        citation['quotes'] = ['This sentence does not exist in the book.']
        with self.assertRaisesRegex(ValueError, 'Missing literal evidence'):
            benchmark.validate_review({'cases': [case]}, self.sources)
        citation['sourceId'] = 'another-book'
        with self.assertRaises(KeyError):
            benchmark.validate_review({'cases': [case]}, self.sources)

    def test_review_rejects_duplicate_ids_and_unscorable_cases(self):
        import copy
        case = copy.deepcopy(self.cases['code_hash_missing_sentinel'])
        with self.assertRaisesRegex(ValueError, 'Duplicate reviewed case'):
            benchmark.validate_review({'cases': [case, case]}, self.sources)
        for group in case['groups']:
            group['required'] = False
        self.assertFalse(benchmark.score_evidence(case, [])['complete'])
        with self.assertRaisesRegex(ValueError, 'required, answerable facts'):
            benchmark.validate_review({'cases': [case]}, self.sources)

    def test_review_rejects_stale_book(self):
        import json
        data = json.loads(benchmark.CASES.read_text())
        data['source']['sha256'] = 'changed'
        with patch.object(benchmark.Path, 'read_text', return_value=json.dumps(data)):
            with self.assertRaisesRegex(ValueError, 'Book fixture changed'):
                benchmark.load()

    def test_wrong_document_or_matching_ids_without_text_do_not_pass(self):
        case = self.cases['code_hash_missing_sentinel']
        support = next(iter(benchmark.minimum_supports(case)))
        for replacement in ({'document': 'different-book'}, {'text': ''}):
            selected = [{**source, **replacement} for source in self.selected(support)]
            self.assertFalse(benchmark.score_evidence(case, selected)['complete'])

    def test_unified_runner_fails_missing_evidence_and_keeps_other_cases(self):
        from tests.benchmarks.test_buzzdb_embeddings import BuzzDBEmbeddingBenchmarkTests as fixture
        instance = fixture()
        instance.cases = [self.cases['code_hash_missing_sentinel'],
                          self.cases['same_bitpacked_slice_alignment']]
        instance.measurements = {'fresh_query_embedding_seconds': 0.0}
        ids = next(iter(benchmark.minimum_supports(instance.cases[0])))
        sources = [{'tokensmithChunkId': cid, 'context': self.sources['buzzdb', cid]} for cid in ids]
        with patch.object(instance, 'retrieve_sources', side_effect=[{'sources': sources}, {'sources': []}]) as retrieve:
            with self.assertRaisesRegex(AssertionError, 'Hybrid evidence regression'):
                instance.test_ollama_hybrid_retrieval()
        self.assertEqual([call.args[1] for call in retrieve.call_args_list], [8, 8])
        report = fixture.report
        self.assertEqual((report['passed'], report['total']), (1, 2))
        self.assertTrue(report['cases'][1]['failures'])
        self.assertTrue(report['cases'][0]['requiredFacts'])


if __name__=='__main__':
    unittest.main()
