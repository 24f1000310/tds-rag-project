import os
import re
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding
from passages import PASSAGES

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ROOT = Path(__file__).resolve().parent
MODEL_CACHE = ROOT / "model_cache"

# Tune this using the assignment's known-answer and unanswerable tests.
ABSTAIN_THRESHOLD = float(os.getenv("ABSTAIN_THRESHOLD", "0.30"))

class RAGEngine:
    def __init__(self):
        if not MODEL_CACHE.exists() or not any(MODEL_CACHE.iterdir()):
            raise RuntimeError(
                "Model cache is missing. Run `python setup_model.py` "
                "and include model_cache/ in the deployment."
            )

        self.model = TextEmbedding(
            model_name=MODEL_NAME,
            cache_dir=str(MODEL_CACHE),
            threads=1,
            local_files_only=True,
        )
        self.passages = PASSAGES
        self.texts = [p["text"] for p in self.passages]

        self.passage_vectors = self._embed(self.texts)
        self.sentences = []
        self.sentence_passage_ids = []

        for passage in self.passages:
            parts = re.split(r"(?<=[.!?])\s+", passage["text"])
            for sentence in parts:
                sentence = sentence.strip()
                if sentence:
                    self.sentences.append(sentence)
                    self.sentence_passage_ids.append(passage["id"])

        self.sentence_vectors = self._embed(self.sentences)

    def _embed(self, texts):
        vectors = np.asarray(
            list(self.model.embed(texts)),
            dtype=np.float32,
        )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)

    def query(self, question: str, top_k: int = 3):
        question = question.strip()
        if not question:
            return self._abstain()

        # The application does not execute user instructions or call an LLM.
        # It only embeds the question and retrieves fixed source text.
        query_vector = self._embed([question])[0]

        passage_scores = self.passage_vectors @ query_vector
        order = np.argsort(-passage_scores)
        top_indices = order[:min(top_k, len(order))]
        best_score = float(passage_scores[top_indices[0]])

        if best_score < ABSTAIN_THRESHOLD:
            return self._abstain()

        ranked_sources = [int(i) for i in top_indices]
        ranked_scores = [
            float(np.clip(passage_scores[i], 0.0, 1.0))
            for i in top_indices
        ]

        allowed_ids = set(ranked_sources)
        sentence_scores = self.sentence_vectors @ query_vector
        candidates = []

        for i, score in enumerate(sentence_scores):
            passage_id = self.sentence_passage_ids[i]
            if passage_id in allowed_ids:
                candidates.append(
                    (float(score), self.sentences[i], passage_id)
                )

        candidates.sort(key=lambda item: item[0], reverse=True)

        selected = []
        seen = set()
        best_sentence_score = candidates[0][0] if candidates else 0.0

        for score, sentence, passage_id in candidates:
            if len(selected) >= 3:
                break
            if score < max(0.20, best_sentence_score - 0.18):
                continue
            if sentence not in seen:
                selected.append(sentence)
                seen.add(sentence)

        if not selected:
            return self._abstain()

        confidence = float(
            np.clip((best_score - 0.20) / 0.55, 0.0, 0.98)
        )

        return {
            "answer": " ".join(selected),
            "sources": ranked_sources,
            "scores": [round(s, 4) for s in ranked_scores],
            "confidence": round(confidence, 4),
        }

    @staticmethod
    def _abstain():
        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.1,
        }
