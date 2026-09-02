"""The full Faithful-MedRAG pipeline with per-component toggles.

Assembling every month's piece behind flags is what makes the month-6 ablation
possible: turn one component on at a time and attribute each metric change to it.

    query
      -> retrieve (dense  OR  hybrid=BM25+MedCPT via RRF)
      -> [reranker]                     cross-encoder
      -> [answerability gate]           answer / abstain, pre-generation
      -> generate (vanilla OR grounded)
      -> [claim repair loop]            verify -> re-retrieve -> repair/abstain
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .data import Answer, Question
from .gate import (AnswerabilityGate, extract_features,
                   synthesize_training_data)
from .generate import GroundedGenerator, VanillaGenerator
from .metrics import (abstention_precision_recall, answer_accuracy,
                      answer_correct, corpus_faithfulness, corpus_hallucination)
from .repair import repair_answer
from .retrieval import CrossEncoderReranker, DenseRetriever, HybridRetriever
from .verify import verify_answer


@dataclass
class Config:
    name: str
    hybrid: bool = False
    reranker: bool = False
    gate: bool = False
    repair: bool = False
    grounded: bool = False
    tau: float = 0.5


def _train_gate() -> AnswerabilityGate:
    X, y = synthesize_training_data(n=400, seed=0)
    return AnswerabilityGate().fit(X, y)


class FaithfulMedRAG:
    def __init__(self, corpus, config: Config, gate: Optional[AnswerabilityGate] = None):
        self.cfg = config
        self.retriever = HybridRetriever(corpus) if config.hybrid else DenseRetriever(corpus)
        self.reranker = CrossEncoderReranker() if config.reranker else None
        self.generator = GroundedGenerator() if config.grounded else VanillaGenerator()
        self.gate = gate if gate is not None else (_train_gate() if config.gate else None)
        # The gate is a cheap, standalone pre-check: it always reads dense-cosine
        # features (0-1, absolute-meaningful) regardless of the downstream
        # retriever, so its inputs match the scale it was trained on.
        self.gate_retriever = DenseRetriever(corpus)

    def answer_question(self, q: Question, k: int = 6) -> Answer:
        ev = self.retriever.search(q.text, k=k)
        if self.reranker is not None:
            ev = self.reranker.rerank(q.text, ev, k=k)

        # pre-generation answerability gate
        if self.cfg.gate and self.gate is not None:
            # fixed feature-k (matches the gate's training), independent of gen-k
            feats = extract_features(q.text, self.gate_retriever, k=5).reshape(1, -1)
            prob = float(self.gate.predict_proba(feats)[0])
            if self.gate.decide(prob, tau=self.cfg.tau) == "abstain":
                return Answer(qid=q.qid, text="[abstained: gate]", abstained=True)

        ans = self.generator.answer(q, ev)
        verify_answer(ans.claims, ev)
        if self.cfg.repair:
            repair_answer(ans, self.retriever, self.reranker, max_rounds=2, k=k)
        return ans

    def evaluate(self, questions: List[Question]) -> dict:
        answers = [self.answer_question(q) for q in questions]
        golds = [q.gold_answer for q in questions]
        decided_abstain = [a.abstained for a in answers]
        truly_unans = [not q.answerable for q in questions]
        ap, ar = abstention_precision_recall(decided_abstain, truly_unans)
        return {
            "config": self.cfg.name,
            "accuracy": answer_accuracy(answers, golds),
            "faithfulness": corpus_faithfulness(answers),
            "hallucination": corpus_hallucination(answers),
            "abstain_prec": ap,
            "abstain_rec": ar,
            "pct_abstained": sum(decided_abstain) / len(answers),
            "answers": answers,
        }
