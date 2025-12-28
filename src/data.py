from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


LABEL_ALIASES = {
    "human": 0,
    "ai": 1,
    "0": 0,
    "1": 1,
}

CLASS_NAMES = {0: "Human", 1: "AI"}


def find_hc3_csv_files(input_path: Path) -> List[Path]:
    if input_path.is_file() and input_path.suffix.lower() == ".csv":
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.glob("*.csv"))
    return []


def _load_csv_frames(files: List[Path]) -> pd.DataFrame:
    frames = []
    for path in files:
        if path.suffix.lower() != ".csv":
            continue
        try:
            df = pd.read_csv(path)
            df["__source"] = str(path)
            frames.append(df)
        except Exception as exc:
            print(f"Warning: failed to read {path}: {exc}")
    if not frames:
        raise FileNotFoundError("No readable .csv files found.")
    return pd.concat(frames, ignore_index=True)


def _parse_answer_list(value: object) -> List[str]:
    if isinstance(value, list):
        answers = value
    elif isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
        answers = parsed if isinstance(parsed, list) else [parsed]
    else:
        try:
            if pd.isna(value):
                return []
        except Exception:
            pass
        answers = [value]
    cleaned = [str(item).strip() for item in answers if str(item).strip()]
    return cleaned


def _load_hc3_frames(files: List[Path]) -> pd.DataFrame:
    rows = []
    for path in files:
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"Warning: failed to read {path}: {exc}")
            continue
        if not {"human_answers", "chatgpt_answers"}.issubset(df.columns):
            print(
                f"Warning: {path} missing required columns "
                "(human_answers, chatgpt_answers)."
            )
            continue
        for _, row in df.iterrows():
            for answer in _parse_answer_list(row.get("human_answers")):
                rows.append({"text": answer, "label": 0, "lang": "zh"})
            for answer in _parse_answer_list(row.get("chatgpt_answers")):
                rows.append({"text": answer, "label": 1, "lang": "zh"})
    if not rows:
        raise FileNotFoundError("No readable HC3 .csv files found.")
    return pd.DataFrame(rows)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    text_col = cols.get("text")
    label_col = cols.get("label")
    if text_col is None or label_col is None:
        raise ValueError("Expected columns 'text' and 'label' in data files.")

    keep_cols = [text_col, label_col]
    lang_col = "lang" if "lang" in df.columns else None
    if lang_col:
        keep_cols.append(lang_col)

    out = df[keep_cols].rename(columns={text_col: "text", label_col: "label"})
    out["text"] = out["text"].astype(str).str.strip()
    out = out.dropna(subset=["text", "label"]).reset_index(drop=True)
    out = out[out["text"].str.len() > 0].copy()
    return out


def _map_labels(series: pd.Series) -> Tuple[np.ndarray, Dict[str, int]]:
    if pd.api.types.is_numeric_dtype(series):
        values = series.astype(int)
        unique = set(values.unique().tolist())
        if not unique.issubset({0, 1}):
            raise ValueError(f"Numeric labels must be 0/1; got: {sorted(unique)}")
        return values.to_numpy(), {"Human": 0, "AI": 1}

    normalized = series.astype(str).str.strip().str.lower()
    mapped = normalized.map(LABEL_ALIASES)
    if mapped.isna().any():
        unknown = sorted(set(normalized[mapped.isna()].tolist()))
        raise ValueError(f"Unknown label values: {unknown}. Expected AI/Human or 1/0.")
    return mapped.astype(int).to_numpy(), {"Human": 0, "AI": 1}


def load_labeled_data(
    input_path: Path, extra_csv: List[Path] | None = None
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    frames = []
    hc3_files = find_hc3_csv_files(input_path)
    if not hc3_files and extra_csv:
        print(
            f"Warning: no HC3 .csv files found in {input_path}; "
            "training will use only the English reference dataset."
        )
    if hc3_files:
        zh_df = _load_hc3_frames(hc3_files)
        frames.append(zh_df)
    if extra_csv:
        en_df = _load_csv_frames(extra_csv)
        en_df["lang"] = "en"
        frames.append(en_df)
    if not frames:
        raise FileNotFoundError(f"No readable HC3 .csv files found in: {input_path}")
    df = pd.concat(frames, ignore_index=True)
    df = _normalize_columns(df)
    labels, label_map = _map_labels(df["label"])
    df = df.assign(label=labels)
    return df, label_map


def summarize_dataset(df: pd.DataFrame) -> Dict[str, object]:
    lengths = df["text"].astype(str).str.len()
    class_counts = df["label"].value_counts().to_dict()
    class_counts_named = {
        CLASS_NAMES.get(int(label), str(label)): int(count)
        for label, count in class_counts.items()
    }
    stats = {
        "total_samples": int(len(df)),
        "class_counts": class_counts_named,
        "text_length": {
            "min": int(lengths.min()),
            "mean": float(lengths.mean()),
            "median": float(lengths.median()),
            "p95": float(np.percentile(lengths, 95)),
            "max": int(lengths.max()),
        },
    }
    if "lang" in df.columns:
        lang_counts = df["lang"].value_counts().to_dict()
        stats["language_counts"] = {str(k): int(v) for k, v in lang_counts.items()}
    return stats
