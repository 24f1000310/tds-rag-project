import os
import re
import threading
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi

from passages import PASSAGES

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ROOT = Path(__file__).resolve().parent
MODEL_CACHE = ROOT / "model_cache"

# Initial thresholds: tune against the actual 20-question evaluator.
MIN_HYBRID_SCORE = float(os.getenv("MIN_HYBRID_SCORE", "0.30"))
MIN_LEXICAL_SCORE = float(os.getenv("MIN_LEXICAL_SCORE", "0.015"))

TOKEN_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_+#.-]*")

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "what", "which", "who", "when", "where", "why",
    "how", "many", "much", "of", "for", "to", "from", "in", "on", "at",
    "by", "with", "and", "or", "that", "this", "it", "its", "use", "uses",
    "used", "using", "about", "explain", "tell", "me", "please", "give",
    "can", "could", "would", "should", "does", "their", "they", "them",
}

# Explicitly reject requests that attempt to change the assistant's rules,
# extract hidden instructions, or obtain secrets. Retrieved text is never
# executed: it is treated only as data for evidence extraction.
INJECTION_PATTERNS = [
    r"\b(ignore|disregard|forget|override|bypass)\b.{0,70}\b(previous|prior|above|system|developer|all)\b",
    r"\b(reveal|print|show|repeat|dump|expose)\b.{0,60}\b(system prompt|developer prompt|hidden prompt|secret|api key|credentials|token)\b",
    r"\b(system prompt|developer message|hidden instructions|internal instructions)\b",
    r"\b(jailbreak|prompt injection|DAN mode)\b",
    r"\b(act as|pretend to be)\b.{0,50}\b(unrestricted|no rules|without restrictions)\b",
]

def tokenize(text):
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if token.lower() not in STOP_WORDS
    ]

class RAGEngine:
    def __init__(self):
        if not MODEL_CACHE.exists() or not any(MODEL_CACHE.iterdir()):
            raise RuntimeError(
                "model_cache is missing. Run `python setup_model.py` "
                "and deploy the generated model_cache directory."
            )

        # FastEmbed's ONNX inference session is shared by requests.
        # Serialize access to keep concurrent calls safe.
        self.model_lock = threading.RLock()
        self.model = TextEmbedding(
            model_name=MODEL_NAME,
            cache_dir=str(MODEL_CACHE),
            threads=1,
            local_files_only=True,
        )

        self.passages = PASSAGES
        self.texts = [p["text"] for p in PASSAGES]
        self.passage_tokens = [tokenize(text) for text in self.texts]
        self.bm25 = BM25Okapi(self.passage_tokens)

        self.sentences = []
        self.sentence_passage_ids = []
        for passage in PASSAGES:
            for sentence in re.split(r"(?<=[.!?])\s+", passage["text"]):
                sentence = sentence.strip()
                if sentence:
                    self.sentences.append(sentence)
                    self.sentence_passage_ids.append(passage["id"])

        self.sentence_tokens = [tokenize(s) for s in self.sentences]
        self.sentence_bm25 = BM25Okapi(self.sentence_tokens)

        passage_cache = ROOT / "passage_vectors.npy"
        sentence_cache = ROOT / "sentence_vectors.npy"
        if passage_cache.exists() and sentence_cache.exists():
            # Precomputed offline to reduce cold-start work on serverless hosts.
            self.passage_vectors = np.load(passage_cache)
            self.sentence_vectors = np.load(sentence_cache)
        else:
            self.passage_vectors = self._embed(self.texts)
            self.sentence_vectors = self._embed(self.sentences)

    def _embed(self, texts):
        with self.model_lock:
            vectors = np.asarray(
                list(self.model.embed(texts)),
                dtype=np.float32,
            )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)

    @staticmethod
    def _is_adversarial(question):
        q = question.lower()
        return any(re.search(pattern, q, flags=re.I | re.S)
                   for pattern in INJECTION_PATTERNS)

    @staticmethod
    def _abstain():
        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.1,
        }

    def query(self, question: str, top_k: int = 3):
        question = question.strip()
        if not question or self._is_adversarial(question):
            return self._abstain()

        query_tokens = tokenize(question)
        if not query_tokens:
            return self._abstain()

        query_vector = self._embed([question])[0]

        # Dense cosine similarity stays in its natural range. Do not shift
        # unrelated cosine scores upward by 0.5: that makes abstention weak.
        dense = self.passage_vectors @ query_vector
        dense01 = np.clip(dense, 0.0, 1.0)

        # BM25 is normalized, then discounted by the fraction of meaningful
        # query terms actually present in each passage. This avoids the common
        # bug where the best irrelevant passage receives lexical score 1.0
        # merely because it is the best among otherwise irrelevant passages.
        lexical = np.asarray(self.bm25.get_scores(query_tokens), dtype=np.float32)
        lexical_max = float(lexical.max()) if lexical.size else 0.0
        lexical_rank = (
            lexical / lexical_max if lexical_max > 1e-9
            else np.zeros_like(lexical)
        )

        query_set = set(query_tokens)
        passage_coverage = np.asarray([
            len(query_set.intersection(set(tokens))) / max(1, len(query_set))
            for tokens in self.passage_tokens
        ], dtype=np.float32)
        lexical01 = lexical_rank * passage_coverage

        hybrid = 0.62 * dense01 + 0.23 * lexical01 + 0.15 * passage_coverage

        order = np.argsort(-hybrid)
        best_idx = int(order[0])
        best_score = float(hybrid[best_idx])
        best_dense = float(dense01[best_idx])
        best_coverage = float(passage_coverage[best_idx])

        # Require meaningful evidence. A strong semantic paraphrase may pass
        # with less overlap, but a weakly related query cannot pass on rank
        # normalization alone.
        if best_score < MIN_HYBRID_SCORE:
            return self._abstain()
        if best_coverage < 0.08 and best_dense < 0.58:
            return self._abstain()

        selected_indices = [int(i) for i in order[:min(top_k, len(order))]]
        selected_scores = [float(hybrid[i]) for i in selected_indices]

        # Sentence-level hybrid reranking restricted to the top passages.
        allowed = set(selected_indices)
        sentence_dense = self.sentence_vectors @ query_vector
        sentence_lex = np.asarray(
            self.sentence_bm25.get_scores(query_tokens), dtype=np.float32
        )
        max_sentence_lex = float(sentence_lex.max()) if sentence_lex.size else 0.0
        sentence_lex_rank = (
            sentence_lex / max_sentence_lex if max_sentence_lex > 1e-9
            else np.zeros_like(sentence_lex)
        )

        ranked_sentences = []
        for i, sentence in enumerate(self.sentences):
            pid = self.sentence_passage_ids[i]
            if pid not in allowed:
                continue
            sentence_tokens = set(self.sentence_tokens[i])
            overlap = len(query_set.intersection(sentence_tokens)) / max(1, len(query_set))
            dense_score = float(np.clip(sentence_dense[i], 0.0, 1.0))
            lexical_score = float(sentence_lex_rank[i] * overlap)
            score = 0.62 * dense_score + 0.23 * lexical_score + 0.15 * overlap
            ranked_sentences.append((score, sentence, pid))

        ranked_sentences.sort(key=lambda item: item[0], reverse=True)

        answer_sentences = []
        seen = set()
        top_sentence_score = ranked_sentences[0][0] if ranked_sentences else 0.0
        for score, sentence, pid in ranked_sentences:
            if len(answer_sentences) >= 3:
                break
            if score < max(0.30, top_sentence_score - 0.20):
                continue
            if sentence not in seen:
                answer_sentences.append(sentence)
                seen.add(sentence)

        if not answer_sentences:
            return self._abstain()

        # Scores must be descending and in [0, 1], as required by the API.
        confidence = float(np.clip(0.62 + 0.30 * best_score, 0.62, 0.95))
        return {
            "answer": " ".join(answer_sentences),
            "sources": selected_indices,
            "scores": [round(float(np.clip(s, 0.0, 1.0)), 4) for s in selected_scores],
            "confidence": round(confidence, 4),
        }
