"""MONTH 7b (February) -- Faithfulness-helpfulness trade-off curve.

Goal for the mentor:
    The defining "thesis-grade" analysis.  As we raise the gate's abstention
    threshold tau, the system abstains on its least-confident queries first.
    Those weak-evidence queries are exactly where an LLM hallucinates most, so
    the faithfulness of what we DO answer rises -- while helpfulness (how many
    answerable questions we actually answer) falls.  No single tau is best; we
    characterise the curve and pick an operating point.

Modelling note: a real LLM hallucinates more when its evidence is weak.  We
reproduce that here -- the answer builder adds more UNSUPPORTED claims when the
query-evidence grounding is weak -- so the trade-off is visible on the toy set.

Run:  python3 tradeoff_analysis.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data import Answer, Claim, load_dataset
from common.gate import (AnswerabilityGate, extract_features,
                         synthesize_training_data)
from common.retrieval import HybridRetriever, DenseRetriever
from common.verify import nli_entailment, verify_answer

HALLUCINATIONS = [
    "This treatment is universally effective in all patients.",
    "No further testing or specialist review is ever required.",
]


def evidence_strength(question, evidence):
    """Max query-evidence entailment in [0,1]: how well the top evidence
    actually grounds THIS question."""
    return max((nli_entailment(s.passage.text, question) for s in evidence),
              default=0.0)


def build_answer(q, evidence):
    """Confidence-aware answer: strong evidence -> clean; weak evidence -> the
    model pads with unsupported claims (the hallucination behaviour we model)."""
    strength = evidence_strength(q.text, evidence)
    n_halluc = 0 if strength >= 0.6 else (1 if strength >= 0.3 else 2)
    claims = [Claim(text=s.passage.text.split(". ")[0] + ".") for s in evidence[:2]]
    claims += [Claim(text=HALLUCINATIONS[i]) for i in range(n_halluc)]
    ans = Answer(qid=q.qid, text="", claims=claims)
    verify_answer(ans.claims, evidence)
    return ans


def sweep(corpus, questions, taus):
    retriever = HybridRetriever(corpus)
    gate_retriever = DenseRetriever(corpus)
    X, y = synthesize_training_data(400, 0)
    gate = AnswerabilityGate().fit(X, y)

    # cache each question's gate prob + answer once (tau only changes the cutoff)
    cache = []
    for q in questions:
        prob = float(gate.predict_proba(extract_features(q.text, gate_retriever, 5).reshape(1, -1))[0])
        ev = retriever.search(q.text, k=6)
        ans = build_answer(q, ev)
        faith = sum(1 for c in ans.claims if c.supported) / len(ans.claims)
        correct = (q.answerable and any(q.gold_answer in c.text.lower()
                                        for c in ans.claims if c.supported))
        cache.append((q, prob, faith, correct))

    n_answerable = sum(1 for q in questions if q.answerable)
    rows = []
    for tau in taus:
        answered = [(q, f, ok) for q, p, f, ok in cache if p >= tau]
        faith = sum(f for _, f, _ in answered) / len(answered) if answered else 1.0
        helped = sum(1 for q, _, ok in answered if ok) / n_answerable
        rows.append((tau, len(answered) / len(questions), faith, helped))
    return rows


def bar(x, width=20):
    n = int(round(x * width))
    return "#" * n + "." * (width - n)


def main():
    questions, corpus = load_dataset()
    taus = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.95, 1.0]
    rows = sweep(corpus, questions, taus)

    print("7b  Faithfulness-helpfulness trade-off (sweep abstention threshold)")
    print("=" * 78)
    print(f"{'tau':>6}{'%answered':>11}{'faithfulness':>14}{'helpfulness':>13}")
    print("-" * 78)
    for tau, pct, faith, help_ in rows:
        print(f"{tau:>6.2f}{pct:>11.2f}{faith:>14.3f}{help_:>13.3f}")
    print("-" * 78)

    print("\nTrade-off curve (F = faithfulness of answers, H = helpfulness):")
    for tau, pct, faith, help_ in rows:
        print(f"  tau={tau:<4} F |{bar(faith)}| {faith:.2f}   "
              f"H |{bar(help_)}| {help_:.2f}")

    scored = [(2 * f * h / (f + h) if (f + h) else 0, tau, f, h)
              for tau, _, f, h in rows]
    best = max(scored)
    print(f"\nBalanced operating point (max harmonic mean of F,H): "
          f"tau={best[1]:.2f}  (F={best[2]:.2f}, H={best[3]:.2f})")
    print("Reading: low tau answers everything -> weak-evidence hallucinations")
    print("drag faithfulness down. Raising tau abstains those first, lifting")
    print("faithfulness; push too far and helpfulness collapses. Choosing where")
    print("to sit on this curve -- not one accuracy number -- is the contribution.")


if __name__ == "__main__":
    main()
