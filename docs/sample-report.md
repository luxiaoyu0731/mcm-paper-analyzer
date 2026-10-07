# Synthetic retrieval walkthrough

No PDF, embedding model or LLM was used. This illustrates provenance and report format, not an AI evaluation.

## Input

Section: demand forecast validation. Two self-authored reference passages are returned by a synthetic store and fixed vector.

## Retrieved references

- [sample-1] Synthetic reference: hold out future observations before fitting a demand model.
- [sample-2] Synthetic reference: compare forecast errors against a seasonal baseline.

## Review notes (manually authored)

| Observation | Suggested change | Reference |
| --- | --- | --- |
| A random split can mix future observations into training | State the cutoff date and hold out later observations | sample-1 |
| A complex model without a baseline lacks a useful comparison | Add seasonal baseline errors on the same holdout | sample-2 |

## Revision checklist (manually authored)

1. Document the training/validation date split.
2. Compare with a seasonal baseline.
3. Link each revision to its reference above.

## Acceptance and limits

The real retriever merged two unique passages consistently without changing stored rows. Check each citation against retrieval.json. Mathematical validity, PDF parsing, semantic relevance and model diagnosis are not evaluated by this synthetic example.
