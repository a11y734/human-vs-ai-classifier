import json
from pathlib import Path

import streamlit as st

from src.infer import InferenceError, Predictor


THEME_CSS = """
<style>
:root {
  --bg-1: #d9ecff;
  --bg-2: #eef6ff;
  --card: rgba(255, 255, 255, 0.92);
  --ink: #1b1b1f;
  --muted: #5f6168;
  --accent: #f97316;
  --accent-2: #38bdf8;
  --accent-3: #22c55e;
  --accent-4: #fbbf24;
  --shadow: 0 18px 40px rgba(15, 23, 42, 0.35);
  --radius: 18px;
  --block-input: linear-gradient(135deg, #e7f1ff 0%, #f5f9ff 55%, #ffffff 100%);
  --block-result: linear-gradient(135deg, #fff1e1 0%, #fff9f0 55%, #ffffff 100%);
  --block-explain: linear-gradient(135deg, #eafaf1 0%, #f6fffb 55%, #ffffff 100%);
  --block-metrics: linear-gradient(135deg, #fef6df 0%, #fffdf3 55%, #ffffff 100%);
}

html, body, [class*="css"]  {
  font-family: "Noto Serif SC", "Source Han Serif SC", "PingFang SC", "Microsoft YaHei",
    "Heiti SC", serif;
}

.stApp {
  background:
    radial-gradient(900px circle at 10% -10%, rgba(59, 130, 246, 0.18) 0, transparent 55%),
    radial-gradient(900px circle at 90% 10%, rgba(14, 116, 144, 0.14) 0, transparent 55%),
    linear-gradient(160deg, var(--bg-1), var(--bg-2));
  color: var(--ink);
}

.block-container {
  padding-top: 2rem;
  max-width: 1080px;
}

.hero {
  background: var(--card);
  color: #0f172a;
  padding: 24px 28px;
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  border: 1px solid rgba(0, 0, 0, 0.06);
  animation: fadeUp 0.6s ease-out both;
}

.hero-badge {
  display: inline-block;
  padding: 6px 12px;
  border-radius: 999px;
  background: rgba(59, 130, 246, 0.18);
  color: #1d4ed8;
  font-size: 12px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 10px;
}

.hero h1 {
  margin: 0 0 6px 0;
  font-size: 32px;
  letter-spacing: 0.5px;
}

.hero p {
  margin: 0;
  color: #334155;
  font-size: 16px;
}

.section-title {
  font-size: 20px;
  margin: 0 0 6px 0;
  font-weight: 700;
  letter-spacing: 0.4px;
}

.section-subtitle {
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.18em;
  color: var(--muted);
  margin-bottom: 14px;
}

.section-shell {
  padding: 18px 22px;
  border-radius: var(--radius);
  border: 1px solid rgba(0, 0, 0, 0.08);
  box-shadow: var(--shadow);
  margin-top: 18px;
  color: #0f172a;
}

.section-shell.block-input {
  background: var(--block-input);
  border-left: 6px solid #3b82f6;
}

.section-shell.block-result {
  background: var(--block-result);
  border-left: 6px solid #f97316;
}

.section-shell.block-explain {
  background: var(--block-explain);
  border-left: 6px solid #10b981;
}

.section-shell.block-metrics {
  background: var(--block-metrics);
  border-left: 6px solid #f59e0b;
}

.stat-card {
  background: rgba(255, 255, 255, 0.92);
  border: 1px solid rgba(0, 0, 0, 0.08);
  border-radius: 14px;
  padding: 14px 16px;
  box-shadow: var(--shadow);
  color: #0f172a;
}

.stat-card.card-label {
  background: linear-gradient(135deg, #1d3557, #457b9d);
  color: #fff;
}

.stat-card.card-confidence {
  background: linear-gradient(135deg, #e63946, #f28482);
  color: #fff;
}

.stat-card.card-prob {
  background: linear-gradient(135deg, rgba(42, 157, 143, 0.2), rgba(244, 162, 97, 0.22));
  border: 2px solid rgba(42, 157, 143, 0.5);
}

.stat-label {
  font-size: 12px;
  color: var(--muted);
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.stat-card.card-label .stat-label,
.stat-card.card-confidence .stat-label {
  color: rgba(255, 255, 255, 0.7);
}

.stat-card .stat-label {
  color: #475569;
}

.stat-value {
  font-size: 22px;
  font-weight: 600;
  margin-top: 6px;
}

.dual-bar {
  margin-top: 6px;
  height: 12px;
  border-radius: 999px;
  background: rgba(0, 0, 0, 0.08);
  overflow: hidden;
  display: flex;
}

.dual-bar .ai {
  background: linear-gradient(135deg, var(--accent), #f4a261);
}

.dual-bar .human {
  background: linear-gradient(135deg, var(--accent-2), #52b788);
}

.dual-bar span {
  display: block;
  height: 100%;
}

.stButton > button {
  background: linear-gradient(135deg, var(--accent), var(--accent-4));
  color: #fff;
  border: none;
  border-radius: 999px;
  padding: 0.6rem 1.8rem;
  box-shadow: var(--shadow);
}

.stButton > button:hover {
  filter: brightness(0.96);
}

div[data-testid="stTextArea"] textarea {
  border-radius: 14px !important;
  background: rgba(255, 255, 255, 0.95) !important;
}

@keyframes fadeUp {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>
"""


st.set_page_config(page_title="中文 AI / 人類 文本判別器", layout="wide")
st.markdown(THEME_CSS, unsafe_allow_html=True)
st.title("中文 AI / 人類 文本判別器")
st.caption("中英雙語版本 · TF-IDF + Logistic Regression / SVM")


@st.cache_resource
def get_predictor(model_dir: str = "models", backend: str = "baseline"):
    return Predictor(Path(model_dir), backend=backend)


def metrics_section(paths: dict):
    metrics_path = paths.get("metrics")
    if metrics_path and metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        st.subheader("評估指標")
        metric_cols = st.columns(4)
        metric_cols[0].metric("準確率", _format_metric(metrics.get("accuracy")))
        metric_cols[1].metric("F1 分數", _format_metric(metrics.get("f1")))
        metric_cols[2].metric("ROC AUC", _format_metric(metrics.get("roc_auc")))
        metric_cols[3].metric("PR AUC", _format_metric(metrics.get("pr_auc")))

        cm = metrics.get("confusion_matrix")
        if cm and len(cm) == 2:
            st.caption("混淆矩陣（列：真實，欄：預測）")
            st.table(
                {
                    "人類": {"人類": cm[0][0], "AI": cm[0][1]},
                    "AI": {"人類": cm[1][0], "AI": cm[1][1]},
                }
            )

    stats_path = paths.get("dataset_stats")
    if stats_path and stats_path.exists():
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        st.subheader("資料統計")
        total = stats.get("total_samples")
        class_counts = stats.get("class_counts", {})
        length_stats = stats.get("text_length", {})

        stat_cols = st.columns(3)
        stat_cols[0].metric("樣本數", total or "-")
        stat_cols[1].metric("人類", class_counts.get("Human", "-"))
        stat_cols[2].metric("AI", class_counts.get("AI", "-"))

        length_cols = st.columns(5)
        length_cols[0].metric("最短", _format_metric(length_stats.get("min")))
        length_cols[1].metric("平均", _format_metric(length_stats.get("mean")))
        length_cols[2].metric("中位數", _format_metric(length_stats.get("median")))
        length_cols[3].metric("P95", _format_metric(length_stats.get("p95")))
        length_cols[4].metric("最長", _format_metric(length_stats.get("max")))

    run_dir = paths.get("run_dir")
    if not run_dir:
        return
    metrics_dir = run_dir / "metrics"
    plots = [
        "confusion_matrix.png",
        "roc_curve.png",
        "pr_curve.png",
        "text_length_hist.png",
    ]
    cols = st.columns(2)
    col_idx = 0
    for name in plots:
        img_path = metrics_dir / name
        if img_path.exists():
            caption = {
                "confusion_matrix.png": "混淆矩陣",
                "roc_curve.png": "ROC 曲線",
                "pr_curve.png": "PR 曲線",
                "text_length_hist.png": "文本長度分布",
            }.get(name, name)
            cols[col_idx].image(str(img_path), caption=caption, use_container_width=True)
            col_idx = (col_idx + 1) % 2


def main():
    repo_root = Path(__file__).resolve().parents[1]
    model_dir = repo_root / "models"
    with st.sidebar:
        st.subheader("模型資訊")
        st.write("基線模型（TF-IDF + Logistic Regression / SVM）")
        st.subheader("推論設定")
        margin_threshold = st.slider("低信心門檻", 0.05, 0.3, 0.12, 0.01)
    try:
        predictor = get_predictor(str(model_dir))
    except InferenceError:
        st.warning("找不到模型 artifacts。")
        st.info(
            "請先訓練模型，例如：`python -m src.train --input-dir "
            "refs/chatgpt-comparison-detection/china-data`。"
        )
        st.stop()

    st.markdown(
        "請貼上要判別的文字（中文 / 英文皆可），立即顯示 AI% / 人類% 與信心值，"
        "並提供特徵解釋與評估視覺化。"
    )
    text = st.text_area("", height=160, label_visibility="collapsed")
    submitted = st.button("開始判別")
    if submitted and not text.strip():
        st.error("請先輸入文字。")

    if submitted and text.strip():
        out = predictor.predict(text)
        st.markdown(
            """
            <div class="section-shell block-result">
              <div class="section-title">預測結果</div>
              <div class="section-subtitle">即時機率輸出</div>
            """,
            unsafe_allow_html=True,
        )
        label_col, conf_col, prob_col = st.columns([1.1, 1.1, 1.8])
        margin = abs(out["ai_prob"] - out["human_prob"])
        label_text = "人類" if out["label"] == "Human" else out["label"]
        if margin < margin_threshold:
            label_text = "不確定"
        label_col.markdown(
            f"""
            <div class="stat-card card-label">
              <div class="stat-label">預測類別</div>
              <div class="stat-value">{label_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        conf_col.markdown(
            f"""
            <div class="stat-card card-confidence">
              <div class="stat-label">信心值</div>
              <div class="stat-value">{out["confidence"]:.2%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        ai_pct = out["ai_prob"] * 100
        human_pct = out["human_prob"] * 100
        margin_pct = margin * 100
        prob_col.markdown(
            f"""
            <div class="stat-card card-prob">
              <div class="stat-label">機率分布</div>
              <div class="stat-value">AI {ai_pct:.1f}% · 人類 {human_pct:.1f}%</div>
              <div class="stat-label" style="margin-top: 6px;">差距 {margin_pct:.1f}%</div>
              <div class="dual-bar">
                <span class="ai" style="width: {ai_pct:.1f}%"></span>
                <span class="human" style="width: {human_pct:.1f}%"></span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if margin < margin_threshold:
            st.info("目前判別差距較小，建議提供更長或更具體的文本。")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="section-shell block-explain">
              <div class="section-title">特徵解釋</div>
              <div class="section-subtitle">重要特徵摘要</div>
            """,
            unsafe_allow_html=True,
        )
        explanation = predictor.explain(text)
        if explanation:
            st.table(explanation)
        else:
            st.caption("此模型不支援解釋，或輸入文字無足夠特徵。")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="section-shell block-metrics">
          <div class="section-title">評估與視覺化</div>
          <div class="section-subtitle">模型表現追蹤</div>
        """,
        unsafe_allow_html=True,
    )
    metrics_section(predictor.paths)
    st.markdown("</div>", unsafe_allow_html=True)


def _format_metric(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return value


if __name__ == "__main__":
    main()
