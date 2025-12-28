from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict


@dataclass
class ArtifactBundle:
    run_dir: Path
    vectorizer_path: Path
    model_path: Path
    metadata_path: Path
    metrics_path: Path
    dataset_stats_path: Path
    metrics_dir: Path


def create_run_dir(model_dir: Path, backend: str) -> ArtifactBundle:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = model_dir / backend / timestamp
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    return ArtifactBundle(
        run_dir=run_dir,
        vectorizer_path=run_dir / "vectorizer.joblib",
        model_path=run_dir / "model.joblib",
        metadata_path=run_dir / "metadata.json",
        metrics_path=metrics_dir / "metrics.json",
        dataset_stats_path=metrics_dir / "dataset_stats.json",
        metrics_dir=metrics_dir,
    )


def write_latest(model_dir: Path, backend: str, bundle: ArtifactBundle) -> None:
    backend_dir = model_dir / backend
    backend_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "run_id": bundle.run_dir.name,
        "paths": {
            "vectorizer": bundle.vectorizer_path.relative_to(backend_dir).as_posix(),
            "model": bundle.model_path.relative_to(backend_dir).as_posix(),
            "metadata": bundle.metadata_path.relative_to(backend_dir).as_posix(),
            "metrics": bundle.metrics_path.relative_to(backend_dir).as_posix(),
            "dataset_stats": bundle.dataset_stats_path.relative_to(backend_dir).as_posix(),
        },
    }
    with open(backend_dir / "latest.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def resolve_latest(model_dir: Path, backend: str) -> Dict[str, Path]:
    backend_dir = model_dir / backend
    latest_path = backend_dir / "latest.json"
    if latest_path.exists():
        with open(latest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        normalized = {
            k: str(v).replace("\\", "/") for k, v in data.get("paths", {}).items()
        }
        paths = {k: backend_dir / v for k, v in normalized.items()}
        paths["run_dir"] = backend_dir / str(data.get("run_id", "")).strip()
        required = ("vectorizer", "model")
        if all(paths.get(key) and paths[key].exists() for key in required):
            return paths

        if backend_dir.exists():
            candidates = sorted(
                [path for path in backend_dir.iterdir() if path.is_dir()],
                reverse=True,
            )
            for run_dir in candidates:
                vectorizer = run_dir / "vectorizer.joblib"
                model = run_dir / "model.joblib"
                if vectorizer.exists() and model.exists():
                    metrics_dir = run_dir / "metrics"
                    return {
                        "vectorizer": vectorizer,
                        "model": model,
                        "metadata": run_dir / "metadata.json",
                        "metrics": metrics_dir / "metrics.json",
                        "dataset_stats": metrics_dir / "dataset_stats.json",
                        "run_dir": run_dir,
                    }

    legacy_vectorizer = model_dir / "tfidf_char.joblib"
    legacy_model = model_dir / "logreg_model.joblib"
    legacy_metrics = model_dir / "metrics" / "metrics.json"
    if legacy_vectorizer.exists() and legacy_model.exists():
        return {
            "vectorizer": legacy_vectorizer,
            "model": legacy_model,
            "metadata": model_dir / "metadata.json",
            "metrics": legacy_metrics,
            "dataset_stats": model_dir / "metrics" / "dataset_stats.json",
            "run_dir": model_dir,
        }
    raise FileNotFoundError(
        f"Artifacts not found under {model_dir}. Run training first."
    )
