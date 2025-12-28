# 中英文 AI vs Human 分類器（Baseline）

Demo: https://healinganimal-mc55cew8quapyj4gwt3fbg.streamlit.app/

本專案為中英文文字的基線分類器，並提供 Streamlit 介面即時推論。繁體與簡體中文一律視為同一類中文文本處理。訓練時可選擇合併英文參考資料集，形成中英混合的分類模型。

## 安裝

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## 訓練

中文資料使用 HC3-Chinese CSV，預設路徑為 `refs/chatgpt-comparison-detection/china-data/`。
欄位需求：
- `question`
- `human_answers`（JSON 陣列字串）
- `chatgpt_answers`（JSON 陣列字串）
訓練時會將 human/chatgpt answers 展平成 `text` 並標記 `Human=0`、`AI=1`。

```powershell
python -m src.train --input-dir refs/chatgpt-comparison-detection/china-data --model-dir models
```

若要合併英文參考資料集，使用 `--include-english`：

```powershell
python -m src.train --input-dir refs/chatgpt-comparison-detection/china-data --include-english `
  --ref-csv refs/human-vs-ai-text-classifier/data/your_dataset_5000.csv
```

`--ref-csv` 可接受檔案、資料夾或 glob（例如 `refs/human-vs-ai-text-classifier/data/*.csv`）。

產出會存於 `models/baseline/<run_id>/`，並在
`models/baseline/latest.json` 指向最新模型：
- `vectorizer.joblib`, `model.joblib`
- `metrics/metrics.json` 與圖表（confusion_matrix.png, roc_curve.png, pr_curve.png）
- `metrics/dataset_stats.json` 與 `text_length_hist.png`

可選模型參數：
```powershell
python -m src.train --input-dir refs/chatgpt-comparison-detection/china-data --algorithm svm
python -m src.train --input-dir refs/chatgpt-comparison-detection/china-data --analyzer word --ngram-min 1 --ngram-max 2
```

若不想做繁體→簡體正規化，可加上：
```powershell
python -m src.train --input-dir refs/chatgpt-comparison-detection/china-data --no-normalize-zh
```

## 執行介面

```powershell
streamlit run app.py
```

貼上中/英文文字並點擊 Predict，即可看到 AI% / Human%、信心度與可選的特徵重要性說明。若有評估圖表，UI 會一併顯示。

## 參考來源

- 中文辨識：https://github.com/Hello-SimpleAI/chatgpt-comparison-detection
- 英文辨識：https://github.com/AnastasiyaKotelnikova/human-vs-ai-text-classifier
