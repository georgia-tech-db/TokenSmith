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

    def test_runner_uses_live_adapter_and_reports_runtime_errors_separately(self):
        from tests.benchmarks.test_buzzdb_embeddings import BuzzDBEmbeddingBenchmarkTests as fixture
        case=self.cases['same_bitpacked_slice_alignment']
        sources=[{'tokensmithChunkId':cid,'context':self.sources['buzzdb',cid]}
                 for cid in ['ch15.055','ch15.056']]
        with patch.object(fixture,'setUpClass'), patch.object(fixture,'doClassCleanups') as cleanup, \
             patch.object(fixture,'measurements',{'fresh_query_embedding_seconds':0.0},create=True), \
             patch.object(fixture,'bundle',{'metadata':{'testFixture':True}},create=True), \
             patch.object(fixture,'retrieve_sources',side_effect=[RuntimeError('embedder unavailable'),
                          {'sources':sources},{'sources':sources},{'sources':sources}]) as retrieve:
            result=benchmark.run({case['id']:case},['vector','hybrid'],[4,8])
        self.assertEqual(retrieve.call_count,4)
        self.assertEqual([call.args[1:] for call in retrieve.call_args_list],
                         [(4,'vector'),(8,'vector'),(4,'hybrid'),(8,'hybrid')])
        cleanup.assert_called_once()
        self.assertEqual(len(result['runtimeErrors']),1)
        self.assertEqual(result['runtimeErrors'][0]['error'],'embedder unavailable')
        self.assertEqual(sum(row['metrics']['complete'] for row in result['cases']),3)


if __name__=='__main__':
    unittest.main()
