## Why
Adapt the reference project structure (`refs/human-vs-ai-text-classifier/`) to a Chinese-focused workflow that trains from Excel files under `Chinesedata/` and provides a Streamlit UI for instant inference (AI% / Human%). Start with a simple, fast scikit-learn baseline; treat any transformer backend as a separate, second-phase enhancement.

## What Changes
- Data ingestion from `Chinesedata/` Excel files (`.xlsx`) with normalized columns: `text`, `label` (map to {Human: 0, AI: 1}).
- Baseline model: TF-IDF (character n-grams, e.g., 2–5) + Logistic Regression with stratified split and fixed random seed.
- Artifacts saved under `model/` (aligned to the reference):
  - Vectorizer (e.g., `model/tfidf_char.joblib`)
  - Classifier (e.g., `model/logreg_model.joblib`)
  - Metrics and plots under `model/metrics/` (confusion matrix, ROC/PR) and a compact `metrics.json`.
- Streamlit UI for single-text inference:
  - Input text box, Predict button, show AI%/Human% and predicted label
  - Display available evaluation visuals from `model/metrics/`
  - Clear messaging when artifacts are missing (how to run training)
- Minimal commands documented in README for Windows PowerShell.

## Impact
- Delivers a fast, local Chinese classifier and an interactive demo UI.
- Establishes an artifact contract (`model/` layout) compatible with the reference repo’s conventions.
- Creates a clean baseline that can be upgraded later without changing the UI surface.

## Out of Scope (Phase 1)
- Transformer-based models (Phase 2)
- Multi-language support beyond Chinese
- Online learning or continuous ingestion

## Risks and Mitigations
- Chinese tokenization complexity → Use character n-grams TF-IDF to avoid external tokenizers initially.
- Dataset imbalance → Use stratified split, consider class_weight="balanced"; report metrics transparently.
- Inconsistent labels/columns → Normalize labels and validate required columns during ingestion.

