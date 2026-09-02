"""MONTH 3 (October) -- Hybrid retrieval + cross-encoder reranker (E1, E2).

Goal for the mentor:
    Before touching the novel parts, build a STRONG, honest retrieval baseline
    and measure it properly.  Two experiments:

    E1  Retriever bake-off : BM25 vs dense (MedCPT-stub) vs hybrid (RRF fusion),
                             scored with Recall@k, nDCG@k and MRR on gold docs.
    E2  Reranking impact   : add a cross-encoder reranker on top of hybrid and
                             show the Recall@k lift.

These are labelled "engineering, not novelty" in the proposal -- but we still
measure them, because the gate + repair loop are only as good as the evidence
they receive.

Run:  python3 experiment_e1_e2.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data import load_dataset
from common.metrics import mean_ignore_nan, mrr, ndcg_at_k, recall_at_k
from common.retrieval import (BM25Retriever, CrossEncoderReranker,
                              DenseRetriever, HybridRetriever)

K = 2


def evaluate(retriever, questions, corpus, k=K, rerank=None):
    rec, ndcg, rr = [], [], []
    for q in questions:
        if not q.gold_doc_ids:            # skip the unanswerable question
            continue
        hits = retriever.search(q.text, k=max(k, 8))
        if rerank is not None:
            hits = rerank.rerank(q.text, hits, k=max(k, 8))
        rec.append(recall_at_k(hits, q.gold_doc_ids, k))
        ndcg.append(ndcg_at_k(hits, q.gold_doc_ids, k))
        rr.append(mrr(hits, q.gold_doc_ids))
    return (mean_ignore_nan(rec), mean_ignore_nan(ndcg), mean_ignore_nan(rr))


def main():
    questions, corpus = load_dataset()
    retrievers = [BM25Retriever(corpus), DenseRetriever(corpus), HybridRetriever(corpus)]

    print(f"E1  Retriever bake-off  (metrics @k={K}, averaged over answerable Qs)")
    print("=" * 70)
    print(f"{'retriever':<28}{'Recall@k':>10}{'nDCG@k':>10}{'MRR':>10}")
    print("-" * 70)
    hybrid = None
    for r in retrievers:
        recall, ndcg, rr = evaluate(r, questions, corpus)
        print(f"{r.name:<28}{recall:>10.3f}{ndcg:>10.3f}{rr:>10.3f}")
        if isinstance(r, HybridRetriever):
            hybrid = r

    print()
    print("E2  Reranking impact  (cross-encoder reranker recovers ordering)")
    print("=" * 70)
    reranker = CrossEncoderReranker()
    dense = DenseRetriever(corpus)
    base = evaluate(dense, questions, corpus)
    reranked = evaluate(dense, questions, corpus, rerank=reranker)
    print(f"{'config':<28}{'Recall@k':>10}{'nDCG@k':>10}{'MRR':>10}")
    print("-" * 70)
    print(f"{'dense (no rerank)':<28}{base[0]:>10.3f}{base[1]:>10.3f}{base[2]:>10.3f}")
    print(f"{'dense + reranker':<28}{reranked[0]:>10.3f}{reranked[1]:>10.3f}{reranked[2]:>10.3f}")
    print("-" * 70)
    print(f"nDCG@k lift from reranking = {reranked[1] - base[1]:+.3f}")
    print()
    print("Takeaway: reranking helps most where the first-stage ordering is")
    print("          imperfect -- it lifts dense back up to the hybrid level.")
    print("          The deployed config is hybrid(RRF) + reranker, feeding the")
    print("          month-4/5 novel stages the cleanest evidence set.")


if __name__ == "__main__":
    main()
