import argparse
import glob
import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from src.artifacts import create_run_dir, write_latest
from src.data import CLASS_NAMES, load_labeled_data, summarize_dataset
from src.metrics import (
    compute_metrics,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
    plot_text_length_hist,
)


def _build_classifier(algorithm: str, seed: int):
    if algorithm == "svm":
        base = LinearSVC(random_state=seed, class_weight="balanced")
        return CalibratedClassifierCV(base, cv=3)
    return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)


def _resolve_ref_paths(ref_csv: str | Path | None) -> list[Path]:
    if not ref_csv:
        return []
    if isinstance(ref_csv, Path):
        ref_csv = str(ref_csv)
    path = Path(ref_csv)
    if path.exists():
        if path.is_dir():
            return sorted(path.glob("*.csv"))
        return [path]
    matches = [Path(p) for p in glob.glob(ref_csv)]
    return sorted([p for p in matches if p.is_file()])


def train(
    input_path: Path,
    model_dir: Path,
    seed: int = 42,
    algorithm: str = "logreg",
    analyzer: str = "char",
    ngram_min: int = 2,
    ngram_max: int = 5,
    test_size: float = 0.2,
    ref_csv: str | Path | None = None,
    include_english: bool = False,
    normalize_zh: bool = True,
) -> Dict[str, float]:
    backend = "baseline"

    np.random.seed(seed)
    ref_paths = []
    if include_english:
        ref_paths = _resolve_ref_paths(ref_csv)
        if not ref_paths:
            print(f"Warning: English reference dataset not found at {ref_csv}.")
    df, label_map = load_labeled_data(input_path, extra_csv=ref_paths)

    # Optional Traditional->Simplified normalization for Chinese text
    if normalize_zh:
        try:
            from opencc import OpenCC  # type: ignore

            cc = OpenCC("t2s")
            if "lang" in df.columns:
                is_zh = df["lang"].astype(str) == "zh"
                df.loc[is_zh, "text"] = df.loc[is_zh, "text"].astype(str).map(cc.convert)
            else:
                df["text"] = df["text"].astype(str).map(cc.convert)
        except Exception:
            # If OpenCC not available, skip normalization; char n-grams will mitigate variance
            pass

    # Language balancing when bilingual to avoid skew
    if include_english and "lang" in df.columns:
        import pandas as pd

        def _balance(df_in: pd.DataFrame) -> pd.DataFrame:
            groups = df_in.groupby(["lang", "label"])  # type: ignore[arg-type]
            # target per (lang,label) is the minimum observed
            sizes = groups.size()
            target = int(sizes.min())
            parts = []
            for _, g in groups:
                n = min(len(g), target)
                parts.append(g.sample(n=n, random_state=seed))
            return pd.concat(parts, ignore_index=True)

        df = _balance(df)

    # Recompute dataset stats after preprocessing/balancing
    dataset_stats = summarize_dataset(df)

    X = df["text"].values
    y = df["label"].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=(ngram_min, ngram_max))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    clf = _build_classifier(algorithm, seed)
    clf.fit(X_train_vec, y_train)

    y_prob = clf.predict_proba(X_val_vec)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = compute_metrics(y_val, y_prob)

    bundle = create_run_dir(model_dir, backend)
    plot_confusion_matrix(y_val, y_pred, bundle.metrics_dir / "confusion_matrix.png")
    plot_roc_curve(y_val, y_prob, bundle.metrics_dir / "roc_curve.png")
    plot_pr_curve(y_val, y_prob, bundle.metrics_dir / "pr_curve.png")
    plot_text_length_hist(df["text"].str.len(), bundle.metrics_dir / "text_length_hist.png")

    with open(bundle.metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(bundle.dataset_stats_path, "w", encoding="utf-8") as f:
        json.dump(dataset_stats, f, ensure_ascii=False, indent=2)

    metadata = {
        "algorithm": algorithm,
        "analyzer": analyzer,
        "ngram_range": [ngram_min, ngram_max],
        "seed": seed,
        "include_english": bool(ref_paths),
        "ref_dataset_path": str(ref_csv) if ref_paths else None,
        "normalize_zh": bool(normalize_zh),
        "label_map": label_map,
        "class_names": {str(k): v for k, v in CLASS_NAMES.items()},
    }
    with open(bundle.metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    joblib.dump(vectorizer, bundle.vectorizer_path)
    joblib.dump(clf, bundle.model_path)
    write_latest(model_dir, backend, bundle)

    print(f"Saved artifacts to: {bundle.run_dir}")
    print(json.dumps(metrics, indent=2))
    summary = {"class_counts": dataset_stats.get("class_counts", {})}
    if "language_counts" in dataset_stats:
        summary["language_counts"] = dataset_stats["language_counts"]
    print("Dataset split summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Chinese + English AI-vs-Human baseline classifier"
    )
    parser.add_argument(
        "--input-dir",
        "--input",
        dest="input_dir",
        type=str,
        default="refs/chatgpt-comparison-detection/china-data",
        help="Path to HC3-Chinese .csv folder or a single .csv file",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Output directory for artifacts",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["logreg", "svm"],
        default="logreg",
        help="Classifier type",
    )
    parser.add_argument(
        "--analyzer",
        type=str,
        choices=["char", "word"],
        default="char",
        help="TF-IDF analyzer",
    )
    parser.add_argument("--ngram-min", type=int, default=2, help="Min n-gram size")
    parser.add_argument("--ngram-max", type=int, default=5, help="Max n-gram size")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--ref-csv",
        type=str,
        default="refs/human-vs-ai-text-classifier/data/your_dataset_5000.csv",
        help="Optional English reference dataset (file, directory, or glob)",
    )
    parser.add_argument(
        "--include-english",
        action="store_true",
        help="Include the English reference dataset for bilingual training",
    )
    parser.add_argument(
        "--no-normalize-zh",
        action="store_true",
        help="Disable Traditional->Simplified normalization for Chinese",
    )
    args = parser.parse_args()

    train(
        Path(args.input_dir),
        Path(args.model_dir),
        seed=args.seed,
        algorithm=args.algorithm,
        analyzer=args.analyzer,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
        test_size=args.test_size,
        ref_csv=args.ref_csv,
        include_english=bool(args.include_english),
        normalize_zh=not bool(args.no_normalize_zh),
    )


if __name__ == "__main__":
    main()
