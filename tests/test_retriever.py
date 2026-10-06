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
