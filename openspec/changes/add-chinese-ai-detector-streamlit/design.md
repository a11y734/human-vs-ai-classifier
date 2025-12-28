## Overview
A minimal, fast, and explainable Chinese AI-vs-Human text classifier built with TF‑IDF features and a linear classifier, surfaced via Streamlit. The solution favors simplicity, low latency, and easy iteration.

## Data and Labels
- Source: HC3-Chinese CSV files in `refs/chatgpt-comparison-detection/china-data/` plus the optional reference dataset `refs/human-vs-ai-text-classifier/data/your_dataset_5000.csv`.
- HC3 columns: `question`, `human_answers`, `chatgpt_answers` (answers stored as JSON arrays). The loader flattens answers into `text` with labels `Human=0`, `AI=1`.
- English reference columns: `text`, `label` (categorical with values like `AI`/`Human` or `1`/`0`).
- Preprocessing: use raw text with TF‑IDF character and/or word n‑grams (character n‑grams avoid heavy tokenizers for Chinese, still works for English).

## Baseline Model
- Vectorizer: TF‑IDF with character n‑grams (e.g., n=2..5). Optionally include word n‑grams as a follow-up.
- Classifier: Logistic Regression (class_weight="balanced" if needed), stratified train/validation split.
- Metrics: accuracy, F1, ROC‑AUC, PR‑AUC; plots for confusion matrix, ROC, and PR; dataset stats (class counts, text length distribution).

## Artifacts
- Persisted files in `models/` under a versioned run directory:
  - Vectorizer (e.g., `vectorizer.joblib`)
  - Classifier (e.g., `model.joblib`)
  - Metrics JSON and plots under the run's `metrics/` folder
- A `models/baseline/latest.json` pointer selects the most recent run for inference.

## Streamlit UI
- Single-text input; on submit, display AI% and Human% (probabilities sum to ~100%).
- Display confidence (max probability) and optional explanation/feature importance when available.
- Secondary section shows basic metrics/visualizations if available (e.g., confusion matrix image, ROC/PR curves, dataset stats).
- Clear status messages if artifacts are missing; guide to run training first.

## Extensibility
- Define a small inference interface so future backends (custom features) can plug in behind the same UI.
- Keep baseline the default to minimize dependencies and startup time.

## Risks
- Dataset imbalance or limited size → use stratification and report metrics transparently.
- Chinese tokenization complexity → character n‑grams reduce dependency burden and still perform reasonably.
