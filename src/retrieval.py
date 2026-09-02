"""Retrieval stack: lexical (BM25), dense (stand-in for MedCPT), RRF fusion,
and a cross-encoder reranker (stand-in for BGE-reranker).

BM25 is implemented for real -- it is cheap and exact.  The "dense" retriever
uses a deterministic TF-IDF cosine as a stand-in for MedCPT's learned biomedical
embeddings: same `.search(query, k)` interface, so month 3 can swap in the real
MedCPT encoder without touching the fusion / rerank code.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List

from .data import Passage, Scored

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


# --------------------------------------------------------------------------- #
# BM25 (real implementation)
# --------------------------------------------------------------------------- #
class BM25Retriever:
    name = "BM25"

    def __init__(self, corpus: List[Passage], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1, self.b = k1, b
        self.docs = [tokenize(p.title + " " + p.text) for p in corpus]
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = sum(self.doc_len) / len(self.docs)
        self.tf = [Counter(d) for d in self.docs]
        df: Counter = Counter()
        for d in self.docs:
            df.update(set(d))
        N = len(self.docs)
        self.idf = {t: math.log(1 + (N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def score(self, query_tokens: List[str], i: int) -> float:
        s = 0.0
        tf, dl = self.tf[i], self.doc_len[i]
        for t in query_tokens:
            if t not in tf:
                continue
            freq = tf[t]
            denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            s += self.idf.get(t, 0.0) * (freq * (self.k1 + 1)) / denom
        return s

    def search(self, query: str, k: int = 5) -> List[Scored]:
        qt = tokenize(query)
        scored = [Scored(p, self.score(qt, i)) for i, p in enumerate(self.corpus)]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:k]


# --------------------------------------------------------------------------- #
# Dense retriever  (deterministic TF-IDF cosine == stand-in for MedCPT)
# --------------------------------------------------------------------------- #
class DenseRetriever:
    """Placeholder for MedCPT.  In the thesis, `_embed` calls the MedCPT encoder
    and vectors come from the pre-computed PubMed index shipped with MedRAG."""
    name = "Dense(MedCPT-stub)"

    def __init__(self, corpus: List[Passage]):
        self.corpus = corpus
        docs = [tokenize(p.title + " " + p.text) for p in corpus]
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        N = len(docs)
        self.idf = {t: math.log((N + 1) / (n + 1)) + 1 for t, n in df.items()}
        self.doc_vecs = [self._embed_tokens(d) for d in docs]

    def _embed_tokens(self, tokens: List[str]) -> Dict[str, float]:
        tf = Counter(tokens)
        vec = {t: (c / len(tokens)) * self.idf.get(t, 1.0) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def _embed(self, text: str) -> Dict[str, float]:
        return self._embed_tokens(tokenize(text))

    @staticmethod
    def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
        small, large = (a, b) if len(a) < len(b) else (b, a)
        return sum(v * large.get(t, 0.0) for t, v in small.items())

    def search(self, query: str, k: int = 5) -> List[Scored]:
        qv = self._embed(query)
        scored = [Scored(p, self._cosine(qv, dv))
                  for p, dv in zip(self.corpus, self.doc_vecs)]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:k]


# --------------------------------------------------------------------------- #
# Reciprocal Rank Fusion  (hybrid = BM25 + dense)
# --------------------------------------------------------------------------- #
def reciprocal_rank_fusion(rankings: List[List[Scored]], k: int = 5,
                           rrf_k: int = 60) -> List[Scored]:
    """Fuse several ranked lists into one.  Score = sum 1/(rrf_k + rank)."""
    fused: Dict[str, float] = {}
    passages: Dict[str, Passage] = {}
    for ranking in rankings:
        for rank, s in enumerate(ranking):
            did = s.passage.doc_id
            fused[did] = fused.get(did, 0.0) + 1.0 / (rrf_k + rank + 1)
            passages[did] = s.passage
    out = [Scored(passages[d], sc) for d, sc in fused.items()]
    out.sort(key=lambda s: s.score, reverse=True)
    return out[:k]


class HybridRetriever:
    name = "Hybrid(BM25+MedCPT, RRF)"

    def __init__(self, corpus: List[Passage]):
        self.bm25 = BM25Retriever(corpus)
        self.dense = DenseRetriever(corpus)

    def search(self, query: str, k: int = 5, pool: int = 10) -> List[Scored]:
        return reciprocal_rank_fusion(
            [self.bm25.search(query, pool), self.dense.search(query, pool)], k=k)


# --------------------------------------------------------------------------- #
# Cross-encoder reranker  (stand-in for BGE-reranker)
# --------------------------------------------------------------------------- #
def _stem(tok: str) -> str:
    """Very small suffix stripper so 'inhibit'/'inhibits'/'inhibiting' match."""
    for suf in ("ingly", "edly", "ing", "ies", "ied", "es", "ed", "s"):
        if len(tok) > len(suf) + 2 and tok.endswith(suf):
            return tok[: -len(suf)] + ("y" if suf in ("ies", "ied") else "")
    return tok


class CrossEncoderReranker:
    """Placeholder for a BGE cross-encoder.  The real model scores (query,
    passage) jointly with a transformer.  Here we approximate joint relevance
    with a stemmed content-word overlap, blended with the candidate's incoming
    rank as a prior so a strong upstream ordering is refined, not discarded --
    which is exactly how a reranker behaves in practice.
    """
    name = "CrossEncoder(BGE-stub)"

    def rerank(self, query: str, candidates: List[Scored], k: int = 5,
               alpha: float = 0.25) -> List[Scored]:
        q = {_stem(t) for t in tokenize(query) if len(t) > 2}
        n = len(candidates) or 1
        out = []
        for rank, s in enumerate(candidates):
            toks = {_stem(t) for t in tokenize(s.passage.title + " " + s.passage.text)}
            joint = len(q & toks) / (len(q) or 1)          # query-passage relevance
            prior = 1.0 - rank / n                          # trust upstream order
            out.append(Scored(s.passage, alpha * prior + (1 - alpha) * joint))
        out.sort(key=lambda s: s.score, reverse=True)
        return out[:k]


def build_retriever(kind: str, corpus: List[Passage]):
    kind = kind.lower()
    if kind == "bm25":
        return BM25Retriever(corpus)
    if kind in ("dense", "medcpt"):
        return DenseRetriever(corpus)
    if kind == "hybrid":
        return HybridRetriever(corpus)
    raise ValueError(f"unknown retriever kind: {kind}")
