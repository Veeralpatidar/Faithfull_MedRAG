# Faithfull_MedRAG
Faithfull_MedRAG is an AI-powered medical question-answering system built using Retrieval-Augmented Generation (RAG). Its goal is to provide reliable, evidence-based answers to medical queries by grounding the AI's responses in trusted medical documents rather than relying only on the model's internal knowledge.
# Faithful-MedRAG — Monthly Implementation Log

Runnable, month-by-month code for the thesis **Faithful-MedRAG: Reducing
Hallucination in Biomedical Question Answering via Learned Answerability Gating
and Claim-Level Repair.**

Each folder is one month of work. It contains:
- a **demo script** you can run in front of your mentor (`python3 <script>.py`),
- a **`READING_SCRIPT.md`** — a talking script of exactly what to say about that
  month's work, the numbers to point at, and answers to likely questions, and
- a **`RESULTS_<script>.md`** — the captured console output from an actual run.

See also **`RESULTS_SUMMARY.md`** (all months' numbers in one place) and
**`DATASET.md`** (the toy demo dataset vs. the real MIRAGE / PubMed data).

Everything runs **offline in seconds on numpy alone**. The heavy real components
(MedCPT, BGE-reranker, DeBERTa-NLI, Qwen/Llama via vLLM, the 23.9M-doc PubMed
index, the MIRAGE benchmark) are represented by small, deterministic,
same-interface stand-ins so the *pipeline logic* is exercised end-to-end without
a GPU. Swapping in the real models is a one-line change at each call site.

## Quick start
```bash
pip install -r requirements.txt          # just numpy
python3 run_all.py                        # run all 7 months in order
# or run one month:
python3 month_04_november_grounded_gen_repair/experiment_e3.py
```

## Month map

| Month | Folder | Experiment | What it demonstrates |
|------|--------|-----------|----------------------|
| **Aug** | `month_01_august_baseline` | — | Reproduce MedRAG; accuracy 0.83, no faithfulness check |
| **Sep** | `month_02_september_eval_harness` | — | Faithfulness harness: accuracy 0.83 **but** faithfulness 0.67 |
| **Oct** | `month_03_october_hybrid_retrieval` | E1, E2 | Retriever bake-off + reranker lift (strong baseline) |
| **Nov** | `month_04_november_grounded_gen_repair` | E3 | ★ Grounded gen + **claim-level repair loop** (headline) |
| **Dec** | `month_05_december_answerability_gate` | E4 | Learned **answerability gate**; AUROC 1.0 / ECE 0.01 |
| **Jan** | `month_06_january_ablation_ood` | E5, E6 | Full ablation (attribution) + OOD generalization |
| **Feb** | `month_07_february_human_eval` | — | Human-eval sheet + faithfulness–helpfulness trade-off |

## The shared toolkit (`common/`)
| Module | Contents |
|--------|----------|
| `data.py` | Core dataclasses + the toy MIRAGE-style dataset (→ real MIRAGE/PubMed) |
| `retrieval.py` | `BM25Retriever`, `DenseRetriever` (MedCPT), RRF, `CrossEncoderReranker` (BGE) |
| `generate.py` | `VanillaGenerator` (MedRAG-style), `GroundedGenerator` (strict-citation) |
| `verify.py` | Claim decomposition + `nli_entailment` (DeBERTa-NLI stand-in) |
| `repair.py` | `repair_answer` — the claim-level surgical re-retrieval loop (Contribution A) |
| `gate.py` | `AnswerabilityGate` + feature extraction (Contribution B) |
| `pipeline.py` | `FaithfulMedRAG` — the full pipeline with per-component toggles |
| `metrics.py` | accuracy, faithfulness, retrieval (Recall/nDCG/MRR), ECE, AUROC |

## The through-line (what each meeting adds)
1. **Aug** — baseline number to beat (accuracy 0.83).
2. **Sep** — prove the problem: accuracy ≠ faithfulness (0.83 vs 0.67).
3. **Oct** — build strong, honestly-labelled retrieval scaffolding.
4. **Nov** — the headline fix: verify each claim, re-retrieve, repair or abstain.
5. **Dec** — decide *before* generating with a calibrated learned gate.
6. **Jan** — attribute every gain (ablation) and show it generalises (OOD).
7. **Feb** — validate with humans; characterise the faithfulness–helpfulness curve.

> Novelty vs scaffolding is called out explicitly in each reading script — the
> repair loop and the answerability gate are the contributions; hybrid retrieval,
> reranking, and the metrics are engineering. Keeping that line clear is
> deliberate.
