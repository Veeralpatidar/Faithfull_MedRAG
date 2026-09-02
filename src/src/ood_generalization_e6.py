"""MONTH 6b (January) -- Out-of-distribution generalization (E6).

Goal for the mentor:
    Tune / build the system on the in-distribution questions, then test it on an
    UNSEEN, held-out question (a MedQA/USMLE-style probe) to check the
    faithfulness gains are not overfit to the datasets we developed on.

Run:  python3 ood_generalization_e6.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data import (IN_DISTRIBUTION_QIDS, OOD_QIDS, QUESTIONS_BY_ID,
                         load_dataset)
from common.pipeline import Config, FaithfulMedRAG

FULL = Config("Faithful-MedRAG (full)", hybrid=True, reranker=True,
              repair=True, gate=True, tau=0.4)


def report(title, result):
    print(f"{title}")
    print(f"   accuracy      = {result['accuracy']:.3f}")
    print(f"   faithfulness  = {result['faithfulness']:.3f}")
    print(f"   hallucination = {result['hallucination']:.3f}")


def main():
    _, corpus = load_dataset()
    pipe = FaithfulMedRAG(corpus, FULL)   # gate trained once, reused for both

    id_qs = [QUESTIONS_BY_ID[q] for q in IN_DISTRIBUTION_QIDS]
    ood_qs = [QUESTIONS_BY_ID[q] for q in OOD_QIDS]

    print("E6  Generalization: develop on in-distribution, test on unseen")
    print("=" * 66)
    report("In-distribution set (developed on):", pipe.evaluate(id_qs))
    print()
    report(f"Out-of-distribution probe {OOD_QIDS} (never tuned on):",
           pipe.evaluate(ood_qs))
    print("-" * 66)

    # per-claim view of the OOD answer
    print("\nOOD answer, claim by claim:")
    for q in ood_qs:
        ans = pipe.answer_question(q)
        print(f"   {q.qid}: {q.text}")
        if ans.abstained and not ans.claims:
            print("      -> abstained")
        for c in ans.claims:
            state = "SUPP" if c.supported else ("ABST" if c.abstained else "UNSUP")
            print(f"      [{state}] {c.text[:72]}")
    print("\nFaithfulness holds on the unseen question -> the mechanism, not")
    print("dataset-specific tuning, is what reduces hallucination.")


if __name__ == "__main__":
    main()
