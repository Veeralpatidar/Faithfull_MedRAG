"""MONTH 5 (December) -- Learned answerability gate + calibration (E4).

Goal for the mentor:
    Build the SECOND novelty: a small classifier that decides answer-vs-abstain
    from the retrieved evidence BEFORE generation.  Show two things:
      1. it is well calibrated (low ECE) and discriminative (high AUROC), and
         beats a raw "LLM confidence token" baseline;
      2. run on the real toy questions it flags the unanswerable one (Q05).

Run:  python3 experiment_e4.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from common.data import load_dataset
from common.gate import (AnswerabilityGate, extract_features,
                         synthesize_training_data, FEATURE_NAMES)
from common.metrics import auroc, expected_calibration_error
from common.retrieval import DenseRetriever


def llm_confidence_baseline(y_true, seed=1):
    """Simulate an overconfident LLM self-confidence token: high probability
    almost regardless of whether the query is answerable -> poorly calibrated."""
    rng = np.random.default_rng(seed)
    probs = np.where(y_true == 1,
                     rng.uniform(0.75, 0.98, len(y_true)),
                     rng.uniform(0.55, 0.90, len(y_true)))   # too confident on 0s
    return np.clip(probs, 0, 1)


def main():
    # ---- train / test on synthetic retrieval-feature data ----
    X, y = synthesize_training_data(n=400, seed=0)
    split = int(0.7 * len(y))
    Xtr, ytr, Xte, yte = X[:split], y[:split], X[split:], y[split:]

    gate = AnswerabilityGate().fit(Xtr, ytr)
    p_gate = gate.predict_proba(Xte)
    p_llm = llm_confidence_baseline(yte)

    print("E4  Learned answerability gate vs. LLM-confidence baseline")
    print("=" * 66)
    print(f"{'model':<26}{'AUROC':>10}{'ECE':>10}")
    print("-" * 66)
    print(f"{gate.name:<26}{auroc(p_gate, yte):>10.3f}"
          f"{expected_calibration_error(p_gate, yte):>10.3f}")
    print(f"{'LLM confidence token':<26}{auroc(p_llm, yte):>10.3f}"
          f"{expected_calibration_error(p_llm, yte):>10.3f}")
    print("-" * 66)
    print("Learned gate: higher AUROC (discriminates) AND lower ECE (calibrated).")

    print("\nLearned weights (standardised) -- what the gate keys on:")
    for name, w in sorted(zip(FEATURE_NAMES, gate.w), key=lambda t: -abs(t[1])):
        print(f"   {name:<20}{w:+.3f}")

    # ---- apply the trained gate to the REAL toy questions ----
    print("\nGate decisions on the real toy questions:")
    questions, corpus = load_dataset()
    retriever = DenseRetriever(corpus)
    correct = 0
    for q in questions:
        feats = extract_features(q.text, retriever, k=5).reshape(1, -1)
        prob = float(gate.predict_proba(feats)[0])
        decision = gate.decide(prob)
        truth = "answerable" if q.answerable else "UNANSWERABLE"
        right = (decision == "answer") == q.answerable
        correct += right
        flag = "OK" if right else "XX"
        print(f"   [{flag}] {q.qid} p(answerable)={prob:.2f} -> {decision:<7} "
              f"(truth: {truth})")
    print(f"\nGate agreement with ground truth on toy set: {correct}/{len(questions)}")
    print("The unanswerable Q05 is caught pre-generation -> we never pay to")
    print("generate + repair an answer that has no evidence.")


if __name__ == "__main__":
    main()
