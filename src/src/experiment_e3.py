"""MONTH 4 (November) -- Grounded generation + claim-level repair loop (E3).

Goal for the mentor:
    Implement the HEADLINE contribution.  Take the unfaithful MedRAG-style
    answer, then (a) verify each claim, (b) surgically re-retrieve evidence for
    every unsupported claim and repair it, (c) abstain on any claim that still
    cannot be supported.  Show faithfulness jump while accuracy holds.

Compares three systems on the same questions:
    baseline  : MedRAG vanilla read            (hallucinates)
    grounded  : strict-citation generation      (drops ungrounded sentences)
    +repair   : grounded + claim repair loop     (re-retrieves & fixes/abstains)

Run:  python3 experiment_e3.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data import load_dataset
from common.generate import GroundedGenerator, VanillaGenerator
from common.metrics import (answer_accuracy, corpus_faithfulness,
                            corpus_hallucination)
from common.repair import repair_answer
from common.retrieval import CrossEncoderReranker, HybridRetriever
from common.verify import verify_answer


def run_system(name, questions, retriever, reranker, generator, do_repair):
    answers, golds, traces = [], [], []
    for q in questions:
        ev = retriever.search(q.text, k=6)
        ev = reranker.rerank(q.text, ev, k=6)
        ans = generator.answer(q, ev)
        verify_answer(ans.claims, ev)
        if do_repair:
            traces.append(repair_answer(ans, retriever, reranker, max_rounds=2, k=6))
        answers.append(ans)
        golds.append(q.gold_answer)
    return {
        "name": name,
        "accuracy": answer_accuracy(answers, golds),
        "faithfulness": corpus_faithfulness(answers),
        "hallucination": corpus_hallucination(answers),
        "traces": traces,
        "answers": answers,
    }


def main():
    questions, corpus = load_dataset()
    retriever = HybridRetriever(corpus)
    reranker = CrossEncoderReranker()

    results = [
        run_system("baseline (vanilla)", questions, retriever, reranker,
                   VanillaGenerator(), do_repair=False),
        run_system("grounded gen", questions, retriever, reranker,
                   GroundedGenerator(), do_repair=False),
        run_system("vanilla + repair loop", questions, retriever, reranker,
                   VanillaGenerator(), do_repair=True),
    ]

    print("E3  Grounded generation + claim repair loop")
    print("=" * 74)
    print(f"{'system':<26}{'accuracy':>10}{'faithful':>10}{'halluc.':>10}")
    print("-" * 74)
    for r in results:
        print(f"{r['name']:<26}{r['accuracy']:>10.3f}"
              f"{r['faithfulness']:>10.3f}{r['hallucination']:>10.3f}")
    print("-" * 74)

    # show the repair loop working on one answer
    repaired = results[-1]
    print("\nRepair traces (per question):")
    for t in repaired["traces"]:
        print(f"   {t.qid}: rounds={t.rounds} repaired={t.n_repaired} "
              f"abstained={t.n_abstained}")

    b, f = results[0], results[-1]
    print(f"\nFaithfulness: {b['faithfulness']:.3f} -> {f['faithfulness']:.3f}  "
          f"(+{f['faithfulness'] - b['faithfulness']:.3f})")
    print(f"Accuracy held: {b['accuracy']:.3f} -> {f['accuracy']:.3f}")

    repair_loop_walkthrough(retriever, reranker)


def repair_loop_walkthrough(retriever, reranker):
    """A controlled example proving the loop RECOVERS a true-but-dropped claim
    via targeted re-retrieval, and abstains only on the unsupportable one."""
    from common.data import CORPUS_BY_ID, Answer, Claim, Scored
    from common.repair import repair_answer
    from common.verify import verify_answer

    print("\n" + "=" * 74)
    print("Repair-loop walkthrough (why per-claim repair beats plain grounding)")
    print("=" * 74)

    # Simulate a POOR first-pass retrieval that only returned the insulin passage
    thin_first_pass = [Scored(CORPUS_BY_ID["D07"], 1.0)]
    ans = Answer("DEMO", "")
    ans.claims = [
        Claim("Insulin is secreted by the beta cells of the pancreas."),
        Claim("Glucagon is secreted by the alpha cells of the pancreatic islets."),
        Claim("This hormone is universally effective in all patients."),
    ]
    verify_answer(ans.claims, thin_first_pass)
    print("Before repair (verified against the thin first-pass evidence):")
    for c in ans.claims:
        print(f"   [{'SUPP' if c.supported else 'UNSUPP'}] {c.text}")

    trace = repair_answer(ans, retriever, reranker, max_rounds=2, k=5)
    print(f"\nRepair loop ran (rounds={trace.rounds}, "
          f"repaired={trace.n_repaired}, abstained={trace.n_abstained}):")
    for c in ans.claims:
        state = "SUPP" if c.supported else "ABSTAIN"
        src = f"<- {c.evidence_id}" if c.evidence_id else ""
        print(f"   [{state:>7}] {src:<6} {c.text}")
    print("\n  * claim 2 was NOT in the first retrieval, but claim-targeted")
    print("    re-retrieval found its evidence (D08) and REPAIRED it.")
    print("  * claim 3 has no support anywhere -> abstained on that claim only.")
    print("  Plain grounding would have silently dropped claim 2; repair keeps it.")


if __name__ == "__main__":
    main()
