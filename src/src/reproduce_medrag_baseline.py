"""MONTH 1 (August) -- Reproduce the MedRAG baseline on a MIRAGE-style set.

Goal for the mentor:
    Stand up the plain MedRAG pipeline -- dense retrieve, then read -- and report
    the ONE number MedRAG reports: answer accuracy.  This is the point we must
    match before we are allowed to claim any improvement.

Pipeline (MedRAG):   query --> dense retrieval (MedCPT) --> vanilla read
Metric reported:     answer accuracy only  (no faithfulness yet -- that's month 2)

Run:  python3 reproduce_medrag_baseline.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data import load_dataset
from common.retrieval import DenseRetriever
from common.generate import VanillaGenerator
from common.metrics import answer_accuracy, answer_correct


def run_baseline(k: int = 5):
    questions, corpus = load_dataset()
    retriever = DenseRetriever(corpus)     # stand-in for MedCPT dense index
    generator = VanillaGenerator()         # MedRAG-style "retrieve then read"

    answers, golds = [], []
    print(f"MedRAG baseline  |  retriever={retriever.name}  generator={generator.name}")
    print("-" * 78)
    for q in questions:
        evidence = retriever.search(q.text, k=k)
        ans = generator.answer(q, evidence)
        answers.append(ans)
        golds.append(q.gold_answer)
        top_ids = ", ".join(s.passage.doc_id for s in evidence[:3])
        ok = "OK " if answer_correct(ans, q.gold_answer) else "MISS"
        print(f"[{ok}] {q.qid}: retrieved[{top_ids}]")
        print(f"       answer: {ans.text[:96]}...")

    acc = answer_accuracy(answers, golds)
    print("-" * 78)
    print(f"BASELINE ANSWER ACCURACY = {acc:.3f}  ({int(acc * len(answers))}/{len(answers)})")
    print("Note: MedRAG stops here. It never checks whether each sentence is")
    print("      actually supported -- that blind spot is what month 2 exposes.")
    return acc


if __name__ == "__main__":
    run_baseline()
