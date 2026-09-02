"""Core data structures and a tiny, self-contained medical-QA dataset.

In the real thesis this is replaced by the MIRAGE benchmark (7,663 questions,
5 datasets) over the 23.9M-document PubMed corpus.  Here we ship a ~6-question
toy set over a ~18-passage corpus so the *entire* pipeline runs offline in a
second and produces believable numbers a mentor can inspect.

Every fact below is textbook-level general knowledge -- nothing actionable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# --------------------------------------------------------------------------- #
# Dataclasses used across the whole project
# --------------------------------------------------------------------------- #
@dataclass
class Passage:
    doc_id: str
    title: str
    text: str


@dataclass
class Question:
    qid: str
    text: str
    gold_answer: str            # short reference answer (yes/no or a term)
    gold_doc_ids: List[str]     # passages that truly support the answer
    answerable: bool            # False => the corpus has no supporting evidence


@dataclass
class Scored:
    """A passage paired with a retrieval score."""
    passage: Passage
    score: float


@dataclass
class Claim:
    """One atomic, independently-verifiable statement from an answer."""
    text: str
    supported: Optional[bool] = None
    evidence_id: Optional[str] = None
    entailment: float = 0.0
    abstained: bool = False       # withdrawn by the repair loop (not asserted)


@dataclass
class Answer:
    qid: str
    text: str
    claims: List[Claim] = field(default_factory=list)
    abstained: bool = False
    repair_rounds: int = 0


# --------------------------------------------------------------------------- #
# Toy corpus  (biomedical, textbook-level)
# --------------------------------------------------------------------------- #
CORPUS: List[Passage] = [
    Passage("D01", "Metformin in type 2 diabetes",
            "Metformin is recommended as the first-line pharmacological therapy "
            "for type 2 diabetes mellitus in most treatment guidelines because it "
            "lowers hepatic glucose production and improves insulin sensitivity."),
    Passage("D02", "Biguanides overview",
            "Biguanides such as metformin reduce blood glucose without causing "
            "weight gain and carry a low risk of hypoglycaemia when used alone."),
    Passage("D03", "Aspirin mechanism",
            "Aspirin irreversibly inhibits the cyclooxygenase (COX) enzyme by "
            "acetylating a serine residue, reducing the synthesis of prostaglandins "
            "and thromboxane."),
    Passage("D04", "Antiplatelet agents",
            "Because aspirin's inhibition of COX in platelets is irreversible, its "
            "antiplatelet effect lasts for the lifespan of the platelet."),
    Passage("D05", "Scurvy and vitamin C",
            "Scurvy is caused by a deficiency of vitamin C (ascorbic acid), which "
            "is required for collagen synthesis; symptoms include bleeding gums and "
            "poor wound healing."),
    Passage("D06", "Ascorbic acid functions",
            "Vitamin C acts as a cofactor for prolyl hydroxylase, an enzyme "
            "essential for stabilising the collagen triple helix."),
    Passage("D07", "Pancreatic islet cells",
            "Insulin is secreted by the beta cells of the islets of Langerhans in "
            "the pancreas in response to elevated blood glucose."),
    Passage("D08", "Glucagon and alpha cells",
            "Alpha cells of the pancreatic islets secrete glucagon, which raises "
            "blood glucose, opposing the action of insulin."),
    Passage("D09", "Smoking and lung cancer",
            "Cigarette smoking is the leading cause of lung cancer and is "
            "responsible for the majority of lung-cancer deaths worldwide."),
    Passage("D10", "Tobacco carcinogens",
            "Tobacco smoke contains numerous carcinogens that damage the "
            "respiratory epithelium and increase cancer risk in a dose-dependent "
            "manner."),
    # ---- distractor / loosely-related passages (retrieval noise) -------------
    Passage("D11", "Hypertension basics",
            "Hypertension is a chronic elevation of arterial blood pressure and a "
            "major risk factor for stroke and cardiovascular disease."),
    Passage("D12", "Type 1 vs type 2 diabetes",
            "Type 1 diabetes results from autoimmune destruction of beta cells, "
            "whereas type 2 diabetes is characterised by insulin resistance."),
    Passage("D13", "Prostaglandins",
            "Prostaglandins are lipid mediators involved in inflammation, pain and "
            "fever; NSAIDs reduce their production."),
    Passage("D14", "Collagen structure",
            "Collagen is the most abundant protein in the human body and provides "
            "tensile strength to connective tissue."),
    Passage("D15", "Pancreatic anatomy",
            "The pancreas is both an exocrine gland, secreting digestive enzymes, "
            "and an endocrine gland, secreting hormones into the bloodstream."),
    Passage("D16", "Respiratory system",
            "The lungs are the primary organs of respiration, exchanging oxygen "
            "and carbon dioxide across the alveolar membrane."),
    Passage("D17", "Vitamin D",
            "Vitamin D regulates calcium homeostasis; deficiency can cause rickets "
            "in children and osteomalacia in adults."),
    Passage("D18", "Anaemia overview",
            "Anaemia is a reduction in haemoglobin concentration and can result "
            "from iron, vitamin B12 or folate deficiency."),
]

CORPUS_BY_ID = {p.doc_id: p for p in CORPUS}


# --------------------------------------------------------------------------- #
# Toy questions.  Q05 is deliberately UNANSWERABLE (no supporting evidence in
# the corpus) so the answerability gate and abstention logic have something to
# catch.  Q06 is held out as an "out-of-distribution" probe for month 6.
# --------------------------------------------------------------------------- #
QUESTIONS: List[Question] = [
    Question("Q01", "What is the recommended first-line pharmacological therapy "
                    "for type 2 diabetes?",
             "metformin", ["D01", "D02"], answerable=True),
    Question("Q02", "Which enzyme does aspirin irreversibly inhibit?",
             "cyclooxygenase", ["D03", "D04"], answerable=True),
    Question("Q03", "Which vitamin deficiency causes scurvy?",
             "vitamin c", ["D05", "D06"], answerable=True),
    Question("Q04", "Which pancreatic cells secrete insulin?",
             "beta cells", ["D07", "D15"], answerable=True),
    Question("Q05", "What is the recommended dose of the experimental drug XYZ-999 "
                    "for treating condition ABC-syndrome?",
             "insufficient evidence", [], answerable=False),
    Question("Q06", "What is the leading cause of lung cancer?",
             "smoking", ["D09", "D10"], answerable=True),
]

QUESTIONS_BY_ID = {q.qid: q for q in QUESTIONS}

# Split used from month 6 onward: train/tune on the first datasets, keep Q06 as
# an unseen "MedQA-style" out-of-distribution test question.
IN_DISTRIBUTION_QIDS = ["Q01", "Q02", "Q03", "Q04", "Q05"]
OOD_QIDS = ["Q06"]


def load_dataset():
    """Return (questions, corpus) -- the single entry point every month calls."""
    return QUESTIONS, CORPUS


if __name__ == "__main__":
    qs, corpus = load_dataset()
    print(f"Loaded {len(qs)} questions over {len(corpus)} passages.")
    for q in qs:
        tag = "answerable" if q.answerable else "UNANSWERABLE"
        print(f"  {q.qid} [{tag}] {q.text}")
