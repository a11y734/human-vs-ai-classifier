## 1. Implementation
- [x] 1.1 Create minimal project structure (`src/`, `models/`, `ui/`).
- [x] 1.2 Load HC3-Chinese dataset files (`.csv`) from `--input-dir`; flatten `human_answers` and `chatgpt_answers` into labeled samples.
- [x] 1.3 Build baseline: TF‑IDF (char/word n‑grams) + Logistic Regression/SVM with stratified split and fixed random seed.
- [x] 1.4 Persist artifacts (vectorizer + model) to `models/` with versioned filenames.
- [x] 1.5 Add evaluation: accuracy, F1, confusion matrix, ROC‑AUC, PR‑AUC; save plots under the run's `metrics/` folder, plus dataset stats (class counts, text length).
- [x] 1.6 Implement inference module: load artifacts, return probabilities + confidence, optional feature‑importance explanation.
- [x] 1.7 Implement Streamlit app (`app.py`): single input, AI% / Human% + confidence, optional explanation, and render plots/stats when available.
- [x] 1.8 Add simple CLI: `python -m src.train --input-dir refs/chatgpt-comparison-detection/china-data`.
- [x] 1.9 Document usage in `README.md` (train + run UI).
- [x] 1.10 Localize UI text to Chinese and apply a polished visual style.
- [x] 1.11 Make explanation always visible, present metrics/stats in readable cards, and flag low-confidence results.
- [x] 1.12 Merge the English reference dataset with HC3-Chinese for bilingual training (controlled by `--include-english`).
- [x] 1.13 Add a Streamlit title and strengthen block color contrast in the UI.

- [x] 1.14 Update training defaults and docs to remove `Chinesedata/*.xlsx` assumptions.
- [x] 1.15 Apply language/class balancing for bilingual runs (sampling or class_weight) and keep stratified split by class.
- [x] 1.16 Treat Traditional=Simplified Chinese; add optional script normalization to Simplified during preprocessing, fallback to char n-grams if unavailable.
- [x] 1.17 Update README usage to document bilingual flag and script handling.

## 2. Validation
- [ ] 2.1 `openspec validate add-chinese-ai-detector-streamlit --strict` passes.
- [ ] 2.2 Manual test: paste multiple Chinese texts and verify plausible outputs.
- [ ] 2.3 Confirm visualizations render (confusion matrix + ROC/PR) and model artifacts load without errors.

- [ ] 2.4 Validate bilingual run: ensure English inference works and report per-language metrics.

## 3. Dependencies & Env
- [x] 3.1 Add `requirements.txt` (streamlit, scikit‑learn, pandas, numpy, matplotlib, seaborn, openpyxl); note optional transformer deps if used.
- [x] 3.2 Ensure local run instructions for Windows (PowerShell) are clear.
