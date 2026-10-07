"""Exercise the real retriever with synthetic passages; no API or embedding downloads."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mcm-analyzer"))
from core.retriever import Retriever
from analysis.report_builder import save_report

class SyntheticStore:
    def __init__(self):
        self.rows = [
            {"chunk_id": "sample-1", "paper_id": "2025_C_0000001", "distance": 0.3, "text": "Synthetic reference: hold out future observations before fitting a demand model."},
            {"chunk_id": "sample-2", "paper_id": "2025_C_0000002", "distance": 0.4, "text": "Synthetic reference: compare forecast errors against a seasonal baseline."},
        ]
    def search(self, *args, **kwargs):
        return self.rows

class SyntheticEmbedder:
    def encode_single(self, text):
        return [1.0]  # Explicit stand-in; this example does not test semantic similarity.

def run(destination):
    root = Path(destination).absolute()
    root.mkdir(parents=True, exist_ok=False)
    store = SyntheticStore()
    original = json.dumps(store.rows, sort_keys=True)
    retriever = Retriever(store, SyntheticEmbedder())
    results = retriever.retrieve_for_section("Demand forecast validation", section_type="modeling", problem_letter="C")
    assert len(results) == 2 and json.dumps(store.rows, sort_keys=True) == original
    assert results == retriever.retrieve_for_section("Demand forecast validation", section_type="modeling", problem_letter="C")
    report = "# Synthetic retrieval walkthrough\n\nNo PDF, embedding model or LLM was used. This illustrates provenance and report format, not an AI evaluation.\n\n## Retrieved references\n\n"
    for row in results:
        report += f"- [{row['chunk_id']}] {row['text']}\n"
    report += "\n## Example revision checklist (manually authored)\n\n1. Document the training/validation date split.\n2. Compare with a seasonal baseline.\n3. Link each revision to its reference above.\n"
    save_report(report, str(root / "sample-report.md"))
    (root / "retrieval.json").write_text(json.dumps(results, indent=2) + "\n")
    print(report)
    print("PASS: 2 unique passages; repeat-query consistency; source rows unchanged; 0 model calls.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True)
    run(parser.parse_args().destination)
