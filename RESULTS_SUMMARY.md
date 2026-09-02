# Faithful-MedRAG — Results Summary (all months)

Key numbers from an actual run of each month's demo. Per-month full console
output is saved in each folder as `RESULTS_<script>.md`. Regenerate everything
with `python3 run_all.py`.

> These are demo numbers on the toy dataset (see `DATASET.md`). They validate the
> *pipeline logic*; at real MIRAGE/PubMed scale the gains are partial, not
> saturated.

## Headline story in one table

| Month | Experiment | Metric | Result |
|------|-----------|--------|--------|
| Aug | MedRAG baseline | answer accuracy | **0.833** (faithfulness never measured) |
| Sep | Faithfulness harness | accuracy / faithfulness / hallucination | **0.833 / 0.667 / 0.333** — accuracy ≠ faithfulness |
| Oct | E1 retriever bake-off | nDCG@k (hybrid ≥ dense) | BM25 0.645 · Dense 0.568 · **Hybrid 0.645** |
| Oct | E2 reranking | nDCG@k lift | **+0.077** (recall 0.500 → 0.600) |
| Nov | E3 grounded + repair | faithfulness / accuracy | **0.667 → 1.000** faithful · accuracy **0.833 → 1.000** |
| Nov | E3 repair walkthrough | per-claim | **repaired = 1, abstained = 1** |
| Dec | E4 gate vs LLM conf. | AUROC / ECE | gate **1.000 / 0.011** vs LLM **0.894 / 0.332** |
| Jan | E5 ablation (base→full) | faithfulness / hallucination / abstain-recall | **0.667→1.000 / 0.333→0.000 / 0.000→1.000** |
| Jan | E6 OOD probe (unseen Q) | faithfulness | **1.000** (holds on unseen question) |
| Feb | 7a human-eval sheet | claim-level rows written | 16 rows (target ~300 at scale) |
| Feb | 7b trade-off sweep | sweet spot | **τ ≈ 0.2–0.4** → faithfulness 1.0, helpfulness 1.0 |

## E5 ablation table (each row adds one stage)

```text
configuration                             acc   faith  halluc   abs-P   abs-R   %abs
A. MedRAG baseline (dense+vanilla)      0.833   0.667   0.333   1.000   0.000   0.00
B. + hybrid + reranker                  0.833   0.667   0.333   1.000   0.000   0.00
C. + grounded generation                0.833   1.000   0.000   1.000   0.000   0.00
D. + claim repair loop                  1.000   1.000   0.000   1.000   0.000   0.00
E. + answerability gate (FULL)          1.000   1.000   0.000   1.000   1.000   0.17
```

## 7b trade-off curve (sweep abstention threshold τ)

```text
tau   %answered   faithfulness   helpfulness
0.00     1.00         0.917          1.000
0.20     0.83         1.000          1.000   <- sweet spot
0.40     0.83         1.000          1.000
0.50     0.67         1.000          0.800
0.95     0.33         1.000          0.400
1.00     0.00         1.000          0.000
```

**Reading:** grounded generation kills hallucination, the repair loop recovers
true claims and lifts accuracy, the gate adds calibrated pre-generation
abstention, and the trade-off sweep shows there is no single best τ — you choose
where to sit on the faithfulness–helpfulness curve.
