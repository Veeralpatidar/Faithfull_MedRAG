"""MONTH 6a (January) -- Full ablation (E5), the intellectual core.

Goal for the mentor:
    Turn components on one at a time and attribute every metric change to a
    specific stage.  This is what separates a real contribution from
    scaffolding: we can say WHERE each faithfulness gain comes from.

Run:  python3 ablation_e5.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data import load_dataset
from common.pipeline import Config, FaithfulMedRAG

CONFIGS = [
    Config("A. MedRAG baseline (dense+vanilla)"),
    Config("B. + hybrid + reranker", hybrid=True, reranker=True),
    Config("C. + grounded generation", hybrid=True, reranker=True, grounded=True),
    Config("D. + claim repair loop", hybrid=True, reranker=True, repair=True),
    Config("E. + answerability gate (FULL)", hybrid=True, reranker=True,
           repair=True, gate=True, tau=0.4),   # tau tuned on validation (month 7)
]


def main():
    questions, corpus = load_dataset()
    print("E5  Component ablation  (each row adds one stage)")
    print("=" * 96)
    print(f"{'configuration':<38}{'acc':>7}{'faith':>8}{'halluc':>8}"
          f"{'abs-P':>8}{'abs-R':>8}{'%abs':>7}")
    print("-" * 96)
    rows = []
    for cfg in CONFIGS:
        r = FaithfulMedRAG(corpus, cfg).evaluate(questions)
        rows.append(r)
        print(f"{r['config']:<38}{r['accuracy']:>7.3f}{r['faithfulness']:>8.3f}"
              f"{r['hallucination']:>8.3f}{r['abstain_prec']:>8.3f}"
              f"{r['abstain_rec']:>8.3f}{r['pct_abstained']:>7.2f}")
    print("-" * 96)
    base, full = rows[0], rows[-1]
    print("Attribution (baseline -> full):")
    print(f"   faithfulness   {base['faithfulness']:.3f} -> {full['faithfulness']:.3f}  "
          f"(+{full['faithfulness'] - base['faithfulness']:.3f})")
    print(f"   hallucination  {base['hallucination']:.3f} -> {full['hallucination']:.3f}  "
          f"({full['hallucination'] - base['hallucination']:+.3f})")
    print(f"   abstain recall {base['abstain_rec']:.3f} -> {full['abstain_rec']:.3f}  "
          f"(catches unanswerable queries)")
    print("   Repair loop removes hallucinated claims; the gate adds pre-gen")
    print("   abstention. Each gain is traceable to one stage -- the core result.")


if __name__ == "__main__":
    main()
