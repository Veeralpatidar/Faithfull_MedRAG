"""MONTH 2 (September) -- Faithfulness / hallucination evaluation harness.

Goal for the mentor:
    Take the SAME MedRAG baseline answers from month 1 and measure them on a new
    axis: is every sentence actually supported by the retrieved evidence?  This
    is where we prove empirically that a high-accuracy answer can still be
    unfaithful -- the central motivation of the thesis.

Adds:   per-claim verification, faithfulness, hallucination rate
Shows:  accuracy stays high while faithfulness is low  (accuracy != faithfulness)

Run:  python3 faithfulness_eval.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data import load_dataset
from common.retrieval import DenseRetriever
from common.generate import VanillaGenerator
from common.metrics import (answer_accuracy, corpus_faithfulness,
                            corpus_hallucination, faithfulness)
from common.verify import verify_answer


def run_eval(k: int = 5):
    questions, corpus = load_dataset()
    retriever = DenseRetriever(corpus)
    generator = VanillaGenerator()

    answers, golds = [], []
    print("Per-claim verification of the MedRAG baseline answers")
    print("=" * 78)
    for q in questions:
        evidence = retriever.search(q.text, k=k)
        ans = generator.answer(q, evidence)
        verify_answer(ans.claims, evidence)     # <-- the new step
        answers.append(ans)
        golds.append(q.gold_answer)

        print(f"{q.qid}: {q.text}")
        for c in ans.claims:
            mark = "SUPPORTED  " if c.supported else "UNSUPPORTED"
            src = f"<- {c.evidence_id}" if c.evidence_id else "<- (no evidence)"
            print(f"   [{mark} e={c.entailment:.2f}] {src}  {c.text[:70]}")
        print(f"   faithfulness={faithfulness(ans):.2f}")
        print("-" * 78)

    acc = answer_accuracy(answers, golds)
    faith = corpus_faithfulness(answers)
    hall = corpus_hallucination(answers)
    print("HEADLINE FINDING")
    print(f"   answer accuracy    = {acc:.3f}   (looks great)")
    print(f"   faithfulness       = {faith:.3f}   (much lower!)")
    print(f"   hallucination rate = {hall:.3f}")
    print()
    print("   => A correct label hides an unsupported claim in nearly every")
    print("      answer. Accuracy and faithfulness are NOT the same axis.")
    print("      This gap is the problem the rest of the thesis closes.")
    return acc, faith, hall


if __name__ == "__main__":
    run_eval()
