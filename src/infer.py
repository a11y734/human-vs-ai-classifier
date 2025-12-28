from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import joblib

from src.artifacts import resolve_latest


class InferenceError(Exception):
    pass


class Predictor:
    def __init__(self, model_dir: Path = Path("models"), backend: str = "baseline"):
        self.model_dir = Path(model_dir)
        self.backend = backend
        try:
            paths = resolve_latest(self.model_dir, backend)
        except FileNotFoundError as exc:
            raise InferenceError(str(exc)) from exc

        self.paths = paths
        self.vectorizer = joblib.load(paths["vectorizer"])
        self.clf = joblib.load(paths["model"])
        self.metadata = {}
        meta_path = paths.get("metadata")
        if meta_path and meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        self.class_names = self.metadata.get("class_names", {"0": "Human", "1": "AI"})
        self.class_order = getattr(self.clf, "classes_", [0, 1])
        self._cc = None
        if self.metadata.get("normalize_zh", False):
            try:
                from opencc import OpenCC  # type: ignore

                self._cc = OpenCC("t2s")
            except Exception:
                self._cc = None

    def predict(self, text: str) -> Dict[str, float]:
        if self._cc is not None:
            try:
                text = self._cc.convert(str(text))
            except Exception:
                pass
        X = self.vectorizer.transform([text])
        proba = self.clf.predict_proba(X)[0]
        prob_map = {}
        for idx, cls in enumerate(self.class_order):
            name = self.class_names.get(str(cls), str(cls))
            prob_map[name] = float(proba[idx])

        ai_prob = float(prob_map.get("AI", proba[-1]))
        human_prob = float(prob_map.get("Human", proba[0]))
        label = "AI" if ai_prob >= human_prob else "Human"
        confidence = max(ai_prob, human_prob)
        return {
            "label": label,
            "ai_prob": ai_prob,
            "human_prob": human_prob,
            "confidence": float(confidence),
        }

    def explain(self, text: str, top_k: int = 8) -> List[Dict[str, float]]:
        if not hasattr(self.clf, "coef_"):
            return []
        if self._cc is not None:
            try:
                text = self._cc.convert(str(text))
            except Exception:
                pass
        X = self.vectorizer.transform([text])
        coef = self.clf.coef_[0]
        row = X.tocoo()
        if row.nnz == 0:
            return []

        contributions = row.data * coef[row.col]
        pred = self.predict(text)["label"]
        if pred == "Human":
            contributions = -contributions

        if contributions.size == 0:
            return []
        top_k = min(top_k, contributions.size)
        top_idx = contributions.argsort()[-top_k:][::-1]
        feature_names = self.vectorizer.get_feature_names_out()
        results = []
        for idx in top_idx:
            col = row.col[idx]
            results.append(
                {"feature": feature_names[col], "score": float(contributions[idx])}
            )
        return results
