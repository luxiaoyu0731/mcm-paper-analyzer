import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mcm-analyzer"))
from core.retriever import Retriever

class Store:
    def __init__(self):
        self.rows = [{"chunk_id":"one", "paper_id":"2025_C_1234567", "distance":0.8, "text":"model"}]
    def search(self, *args, **kwargs): return self.rows
class Embedder:
    def encode_single(self, text): return [1.0]

class RetrievalTests(unittest.TestCase):
    def test_repeated_retrieval_does_not_reweight_cache(self):
        store = Store()
        retriever = Retriever(store, Embedder())
        first = retriever.retrieve_for_section("query", section_type="modeling", problem_letter="C")
        second = retriever.retrieve_for_section("query", problem_letter="C")
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0]["distance"], 0.4)
        self.assertEqual(second[0]["distance"], 0.4)
        self.assertEqual(store.rows[0]["distance"], 0.8)
        self.assertNotIn("_same_letter", store.rows[0])
    def test_other_letter_penalty(self):
        result = Retriever(Store(), Embedder()).retrieve_for_section("query", problem_letter="B")
        self.assertAlmostEqual(result[0]["distance"], 0.96)

class RetrievalFailureTests(unittest.TestCase):
    def test_empty_store_has_no_fake_references(self):
        store=Store(); store.rows=[]
        retriever=Retriever(store, Embedder())
        result=retriever.retrieve_for_section('unknown', section_type='modeling')
        self.assertEqual(result, [])
        self.assertEqual(retriever.format_retrieval_context(result), '（无相关检索结果）')

    def test_optional_filter_failure_retains_primary_evidence(self):
        class Partial(Store):
            def search(self, *args, **kwargs):
                if kwargs.get('where'): raise RuntimeError('unsupported filter')
                return self.rows
        result=Retriever(Partial(), Embedder()).retrieve_for_section('query', section_type='modeling', problem_type='forecast')
        self.assertEqual([r['chunk_id'] for r in result], ['one'])

    def test_primary_failure_is_not_reported_as_empty_success(self):
        class Broken(Store):
            def search(self, *args, **kwargs): raise ConnectionError('unavailable')
        with self.assertRaises(ConnectionError):
            Retriever(Broken(), Embedder()).retrieve_for_section('query')
