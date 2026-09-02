# Dataset — what the demo uses vs. what the thesis uses

There are **two layers**: (1) a tiny built-in toy dataset that makes the code run
offline in seconds, and (2) the real public benchmarks the thesis actually
targets. The toy layer has the *same shape* as the real one, so swapping in the
real data is a loader change, not a rewrite.

---

## 1. The demo (toy) dataset — `common/data.py`

A self-contained, textbook-level medical-QA set shipped inside the repo. No
download, no API, no GPU.

**Corpus:** 18 short biomedical passages (`Passage(doc_id, title, text)`), e.g.
metformin, aspirin/COX, scurvy/vitamin C, insulin/beta cells, smoking/lung
cancer, plus deliberate *distractor* passages (hypertension, vitamin D, anaemia)
so retrieval has to discriminate.

**Questions:** 6 factoid questions (`Question(qid, text, gold_answer,
gold_doc_ids, answerable)`):

| QID | Question | Gold answer | Answerable? |
|-----|----------|-------------|-------------|
| Q01 | first-line therapy for type 2 diabetes | metformin | yes |
| Q02 | enzyme aspirin irreversibly inhibits | cyclooxygenase | yes |
| Q03 | vitamin deficiency causing scurvy | vitamin c | yes |
| Q04 | pancreatic cells that secrete insulin | beta cells | yes |
| Q05 | dose of experimental drug "XYZ-999" | insufficient evidence | **NO** |
| Q06 | leading cause of lung cancer | smoking | yes (held out as OOD) |

Two design choices carry the whole demo:
- **Q05 is deliberately unanswerable** — no passage supports it. This is what the
  answerability gate (month 5) and abstention logic (month 4) must catch.
- **Q06 is held out** (`OOD_QIDS`) as an unseen "MedQA-style" probe for the
  month-6 generalization test; the rest are `IN_DISTRIBUTION_QIDS`.

Every fact is general, textbook-level knowledge — nothing clinically actionable.

**Why toy?** So each month's script is verifiable end-to-end in ~1 second with
zero dependencies beyond numpy. The numbers therefore *saturate* (faithfulness
hits 1.0); they prove the pipeline logic is correct, not that the method beats
MedRAG at scale.

---

## 2. The real datasets the thesis targets

These replace the toy layer for the actual experiments. All are public.

### Evaluation benchmarks
- **MIRAGE** — the primary benchmark: **7,663 questions across 5 medical-QA
  datasets**, with published MedRAG scores to beat. This is the main target.
- **PubMedQA** — grounded **yes / no / maybe** biomedical QA; short answers with
  attached context. Good for answerability-gate supervision.
- **BioASQ** — ships **gold documents**, so it's used to evaluate the *retriever*
  directly (Recall@k, nDCG, MRR) — the month-3 metrics.
- **MedQA (USMLE)** — used as the **out-of-distribution** stress test (the real
  version of the month-6 Q06 probe).

### Retrieval corpus
- **PubMed** — **23.9M biomedical abstracts** (plus StatPearls and medical
  textbooks). MedRAG ships **pre-computed MedCPT / Contriever / SPECTER
  embeddings**, so the index does **not** need rebuilding — this is what makes the
  thesis feasible on modest compute.

### Models (drop-in replacements for the stand-ins)
| Toy stand-in (in `common/`) | Real model |
|------|------|
| `DenseRetriever` | MedCPT biomedical dense retriever |
| `BM25Retriever` | BM25 (real; or `rank_bm25`) |
| `CrossEncoderReranker` | BGE cross-encoder reranker |
| `nli_entailment` (in `verify.py`) | DeBERTa-NLI / LLM judge |
| `VanillaGenerator` / `GroundedGenerator` | Qwen2.5-7B / Llama-3.1-8B via vLLM |
| `AnswerabilityGate` training data | features from PubMedQA labels + BioASQ gold docs |

### Where to get them
- MedRAG toolkit, MIRAGE benchmark, and the pre-computed index:
  `https://github.com/Teddy-XiongGZ/MedRAG`
- MIRAGE / PubMedQA / BioASQ / MedQA are all available via the HuggingFace
  `datasets` hub.

---

## 3. How to swap toy → real

`common/data.py` exposes a single entry point:

```python
def load_dataset():
    return QUESTIONS, CORPUS      # toy
```

To go real, reimplement `load_dataset()` to return the same
`(List[Question], List[Passage])` shape from MIRAGE + the PubMed index, and point
`DenseRetriever` / `nli_entailment` / the generator at the real models. **Nothing
downstream changes** — the retrieval, gate, repair, ablation, and metric code all
consume the same dataclasses (`Question`, `Passage`, `Scored`, `Claim`,
`Answer`), which is exactly why the demo is a faithful rehearsal of the real run.
