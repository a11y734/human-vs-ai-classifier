## 1. Implementation (Baseline)
- [x] 1.1 Create folders aligned with reference: `src/`, `model/`, `ui/`.
- [x] 1.2 Ingest all `Chinesedata/*.xlsx`; normalize to columns `text` (str) and `label` ({Human, AI} → {0, 1}); drop nulls/empties.
- [x] 1.3 Train TF-IDF (char n-grams 2–5) + Logistic Regression with stratified split and fixed seed; compute accuracy, F1, ROC-AUC, PR-AUC.
- [x] 1.4 Persist artifacts to `model/`: `tfidf_char.joblib`, `logreg_model.joblib`; write `metrics.json`; save plots under `model/metrics/`.
- [x] 1.5 Implement inference module (`src/infer.py`): load artifacts and return AI%/Human% + predicted label.
- [x] 1.6 Implement Streamlit app (`ui/app.py`): text input → predict; show probabilities/label; display available plots; warn if artifacts missing.
- [x] 1.7 Add `requirements.txt` with minimal deps (streamlit, scikit-learn, pandas, numpy, matplotlib, seaborn, joblib, openpyxl).
- [x] 1.8 Document run commands in `README.md` (PowerShell): train, then `streamlit run ui/app.py`.

## 2. Validation
- [x] 2.1 `openspec validate add-chinese-streamlit-ref-structure --strict` passes.
- [ ] 2.2 Manual sanity test with multiple Chinese texts to confirm plausible outputs.
- [ ] 2.3 Confirm plots exist and UI shows them when present.

## 3. Phase 2 (Transformers, not now)
- [ ] 3.1 Design optional transformer backend with same inference interface; toggle in UI.
- [ ] 3.2 Dependencies isolated; fallback to baseline when transformer artifacts absent.
