"""MONTH 7a (February) -- Human-evaluation sampler.

Goal for the mentor:
    Automatic faithfulness metrics (NLI / LLM-judge) are noisy, so the thesis
    validates them with a ~300-item human spot-check.  This script builds the
    annotation sheet: it runs the full pipeline, explodes every answer into
    per-claim rows, and writes a JSONL file with the model's judgement plus
    BLANK fields for the human annotator.  Agreement between the human labels
    and the model labels is what validates the automatic metric.

Run:  python3 human_eval_sampler.py   (writes human_eval_sheet.jsonl)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data import load_dataset
from common.pipeline import Config, FaithfulMedRAG

FULL = Config("Faithful-MedRAG (full)", hybrid=True, reranker=True,
              repair=True, gate=True, tau=0.4)

# In the thesis: SAMPLE_SIZE = 300, stratified across the 5 MIRAGE datasets.
SAMPLE_SIZE = 300
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "human_eval_sheet.jsonl")


def main():
    questions, corpus = load_dataset()
    pipe = FaithfulMedRAG(corpus, FULL)

    rows = []
    for q in questions:
        ans = pipe.answer_question(q)
        if ans.abstained and not ans.claims:
            rows.append({
                "qid": q.qid, "question": q.text, "claim_id": 0,
                "claim_text": "[whole answer abstained]",
                "model_label": "abstained", "model_evidence_id": None,
                "human_supported": None, "human_notes": ""})
            continue
        for i, c in enumerate(ans.claims):
            model_label = ("supported" if c.supported
                           else "abstained" if c.abstained else "unsupported")
            rows.append({
                "qid": q.qid, "question": q.text, "claim_id": i,
                "claim_text": c.text,
                "model_label": model_label,
                "model_evidence_id": c.evidence_id,
                "human_supported": None,        # <- annotator fills yes/no
                "human_notes": ""})             # <- annotator fills free text

    with open(OUT, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    print("Human-evaluation annotation sheet")
    print("=" * 60)
    print(f"claim-level rows written : {len(rows)}")
    print(f"target sample at scale   : {SAMPLE_SIZE} items (stratified by dataset)")
    print(f"output file              : {OUT}")
    print("\nEach row carries the model's label + blank human fields:")
    for r in rows[:4]:
        print(f"   {r['qid']} [{r['model_label']:<11}] {r['claim_text'][:52]}")
    print("   ...")
    print("\nNext: annotators fill `human_supported`; we then report human-vs-model")
    print("agreement (Cohen's kappa) to validate the automatic faithfulness metric.")


if __name__ == "__main__":
    main()
