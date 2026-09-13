from __future__ import annotations

import json
import math
import subprocess
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import InputExample, SentenceTransformer, evaluation, losses
from torch.utils.data import DataLoader


PRESETS = {
    "rtx4060": {
        "batch_size": 2,
        "max_seq_length": 256,
        "checkpoint_save_steps": 100,
        "checkpoint_save_total_limit": 3,
        "monitor_seconds": 60,
    },
    "rtx4090": {
        "batch_size": 8,
        "max_seq_length": 512,
        "checkpoint_save_steps": 1000,
        "checkpoint_save_total_limit": 5,
        "monitor_seconds": 60,
    },
    "bgem3_rtx4060": {
        "batch_size": 2,
        "max_seq_length": 256,
        "checkpoint_save_steps": 50,
        "checkpoint_save_total_limit": 3,
        "monitor_seconds": 30,
        "use_amp": True,
    },
    "bgem3_triplet_rtx4060": {
        "batch_size": 1,
        "max_seq_length": 256,
        "checkpoint_save_steps": 50,
        "checkpoint_save_total_limit": 3,
        "monitor_seconds": 30,
        "use_amp": True,
    },
    "bgem3_conservative_rtx4060": {
        "batch_size": 2,
        "max_seq_length": 256,
        "checkpoint_save_steps": 1000,
        "checkpoint_save_total_limit": 2,
        "monitor_seconds": 0,
        "use_amp": True,
        "learning_rate": 1e-6,
    },
    "bgem3_margin_rtx4060": {
        "batch_size": 1,
        "max_seq_length": 256,
        "checkpoint_save_steps": 1000,
        "checkpoint_save_total_limit": 2,
        "monitor_seconds": 0,
        "use_amp": True,
        "learning_rate": 1e-6,
    },
    "bgem3_labeled_rtx4060": {
        "batch_size": 4,
        "max_seq_length": 384,
        "checkpoint_save_steps": 1000,
        "checkpoint_save_total_limit": 2,
        "monitor_seconds": 0,
        "use_amp": True,
        "learning_rate": 5e-7,
    },
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    if not checkpoint_dir.exists():
        return None
    checkpoints = [p for p in checkpoint_dir.glob("checkpoint-*") if p.is_dir()]
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def load_examples(path: Path, limit: int | None = None) -> tuple[list[InputExample], dict[str, bool]]:
    examples: list[InputExample] = []
    has_negatives = False
    has_labels = False
    has_margin_labels = False
    has_positive_labels = False
    has_negative_labels = False
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if "margin_label" in row:
                negative = str(row.get("negative", "") or "").strip()
                if not negative:
                    continue
                examples.append(
                    InputExample(
                        texts=[row["query"], row["positive"], negative],
                        label=float(row.get("margin_label", 0.0)),
                    )
                )
                has_margin_labels = True
                has_negatives = True
            elif "label" in row:
                candidate = str(row.get("candidate") or row.get("positive") or "").strip()
                label = float(row.get("label", 0.0))
                if not candidate:
                    continue
                examples.append(InputExample(texts=[row["query"], candidate], label=label))
                has_labels = True
                if label > 0:
                    has_positive_labels = True
                else:
                    has_negative_labels = True
            else:
                texts = [row["query"], row["positive"]]
                negative = str(row.get("negative", "") or "").strip()
                if negative:
                    texts.append(negative)
                    has_negatives = True
                examples.append(InputExample(texts=texts))
            negative = str(row.get("negative", "") or "").strip()
            if limit is not None and len(examples) >= limit:
                break
    return examples, {
        "has_negatives": has_negatives,
        "has_labels": has_labels,
        "has_margin_labels": has_margin_labels,
        "has_positive_labels": has_positive_labels,
        "has_negative_labels": has_negative_labels,
    }


def build_evaluator(path: Path | None, *, batch_size: int, limit: int | None = None) -> tuple[evaluation.SentenceEvaluator | None, str | None, int]:
    if path is None or not path.exists():
        return None, None, 0

    rows = read_jsonl(path)
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        return None, None, 0

    if any("label" in row for row in rows):
        sentences1: list[str] = []
        sentences2: list[str] = []
        labels: list[int] = []
        for row in rows:
            candidate = str(row.get("candidate") or row.get("positive") or "").strip()
            query = str(row.get("query", "")).strip()
            if not query or not candidate or "label" not in row:
                continue
            sentences1.append(query)
            sentences2.append(candidate)
            labels.append(1 if float(row.get("label", 0.0)) > 0 else 0)
        if sentences1 and len(set(labels)) > 1:
            return (
                evaluation.BinaryClassificationEvaluator(
                    sentences1,
                    sentences2,
                    labels,
                    name=path.stem,
                    batch_size=batch_size,
                    show_progress_bar=False,
                ),
                "BinaryClassificationEvaluator",
                len(sentences1),
            )

    if any("negative" in row for row in rows):
        anchors: list[str] = []
        positives: list[str] = []
        negatives: list[str] = []
        for row in rows:
            query = str(row.get("query", "")).strip()
            positive = str(row.get("positive", "")).strip()
            negative = str(row.get("negative", "")).strip()
            if not query or not positive or not negative:
                continue
            anchors.append(query)
            positives.append(positive)
            negatives.append(negative)
        if anchors:
            return (
                evaluation.TripletEvaluator(
                    anchors,
                    positives,
                    negatives,
                    name=path.stem,
                    batch_size=batch_size,
                    show_progress_bar=False,
                ),
                "TripletEvaluator",
                len(anchors),
            )

    return None, None, 0


def _read_gpu_snapshot() -> dict[str, str] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None

    first_line = result.stdout.strip().splitlines()
    if not first_line:
        return None
    name, mem_used, mem_total, util, temp, power = [part.strip() for part in first_line[0].split(",")]
    return {
        "name": name,
        "memory_used_mib": mem_used,
        "memory_total_mib": mem_total,
        "utilization_percent": util,
        "temperature_c": temp,
        "power_w": power,
    }


def start_gpu_monitor(interval_seconds: int, sink: list[dict[str, str]]) -> tuple[threading.Event, threading.Thread | None]:
    stop_event = threading.Event()
    if interval_seconds <= 0:
        return stop_event, None

    def monitor() -> None:
        while not stop_event.wait(interval_seconds):
            snapshot = _read_gpu_snapshot()
            if snapshot is None:
                return
            snapshot["timestamp"] = time.strftime("%H:%M:%S")
            sink.append(snapshot)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    return stop_event, thread


def train_biencoder(
    *,
    train_pairs: Path,
    dev_pairs: Path | None = None,
    base_model: str,
    output_dir: Path,
    checkpoint_dir: Path,
    checkpoint_save_steps: int,
    checkpoint_save_total_limit: int,
    resume: bool,
    monitor_seconds: int,
    epochs: int,
    batch_size: int,
    limit: int | None,
    max_seq_length: int,
    use_amp: bool,
    learning_rate: float | None = None,
    evaluation_steps: int | None = None,
) -> dict[str, object]:
    model_source = base_model
    resumed_from = None
    if resume:
        latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint is not None:
            model_source = str(latest_checkpoint)
            resumed_from = str(latest_checkpoint)

    model_device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_source, device=model_device)
    model.max_seq_length = max_seq_length
    train_examples, example_flags = load_examples(train_pairs, limit)
    train_loader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    has_negatives = bool(example_flags["has_negatives"])
    has_labels = bool(example_flags["has_labels"])
    has_margin_labels = bool(example_flags["has_margin_labels"])
    if has_margin_labels:
        train_loss = losses.MarginMSELoss(model)
        loss_name = "MarginMSELoss"
    elif has_labels and example_flags["has_positive_labels"] and example_flags["has_negative_labels"]:
        train_loss = losses.ContrastiveLoss(
            model,
            distance_metric=losses.SiameseDistanceMetric.COSINE_DISTANCE,
            margin=0.3,
        )
        loss_name = "ContrastiveLoss"
    elif has_negatives:
        train_loss = losses.TripletLoss(
            model,
            distance_metric=losses.TripletDistanceMetric.COSINE,
            triplet_margin=0.2,
        )
        loss_name = "TripletLoss"
    else:
        train_loss = losses.MultipleNegativesRankingLoss(model)
        loss_name = "MultipleNegativesRankingLoss"
    steps_per_epoch = len(train_loader)
    warmup_steps = max(10, steps_per_epoch // 10)
    evaluator, evaluator_name, dev_examples_loaded = build_evaluator(
        dev_pairs,
        batch_size=max(1, min(batch_size * 4, 32)),
        limit=None,
    )
    resolved_evaluation_steps = 0 if evaluator is None else (evaluation_steps or max(10, steps_per_epoch // 2))

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    gpu_logs: list[dict[str, str]] = []
    stop_monitor, monitor_thread = start_gpu_monitor(monitor_seconds, gpu_logs)
    try:
        fit_kwargs = {
            "train_objectives": [(train_loader, train_loss)],
            "epochs": epochs,
            "warmup_steps": warmup_steps,
            "output_path": str(output_dir),
            "show_progress_bar": True,
            "checkpoint_path": str(checkpoint_dir),
            "checkpoint_save_steps": checkpoint_save_steps,
            "checkpoint_save_total_limit": checkpoint_save_total_limit,
            "use_amp": use_amp and torch.cuda.is_available(),
            "evaluator": evaluator,
            "evaluation_steps": resolved_evaluation_steps,
            "save_best_model": evaluator is not None,
        }
        if learning_rate is not None:
            fit_kwargs["optimizer_params"] = {"lr": float(learning_rate)}
        model.fit(**fit_kwargs)
    finally:
        stop_monitor.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=2)

    device_info: dict[str, object] = {
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        device_info["cuda_device"] = torch.cuda.get_device_name(0)
        device_info["cuda_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)

    return {
        "model_source": model_source,
        "resumed_from": resumed_from,
        "train_pairs": str(train_pairs),
        "dev_pairs": str(dev_pairs) if dev_pairs is not None else None,
        "examples_loaded": len(train_examples),
        "dev_examples_loaded": dev_examples_loaded,
        "has_negatives": has_negatives,
        "has_labels": has_labels,
        "has_margin_labels": has_margin_labels,
        "loss_name": loss_name,
        "evaluator_name": evaluator_name,
        "evaluation_steps": resolved_evaluation_steps,
        "epochs": epochs,
        "batch_size": batch_size,
        "max_seq_length": max_seq_length,
        "model_device": str(model.device),
        "steps_per_epoch": steps_per_epoch,
        "warmup_steps": warmup_steps,
        "output_dir": str(output_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_save_steps": checkpoint_save_steps,
        "checkpoint_save_total_limit": checkpoint_save_total_limit,
        "use_amp": bool(use_amp and torch.cuda.is_available()),
        "learning_rate": learning_rate,
        "device_info": device_info,
        "gpu_logs": gpu_logs,
    }


def ndcg(scores: list[float], k: int) -> float:
    dcg = sum(score / math.log2(i + 2) for i, score in enumerate(scores[:k]))
    ideal = sorted(scores, reverse=True)
    idcg = sum(score / math.log2(i + 2) for i, score in enumerate(ideal[:k]))
    return 0.0 if idcg == 0 else dcg / idcg


def evaluate_biencoder(
    *,
    model_path: Path,
    data_dir: Path,
    batch_size: int,
) -> dict[str, object]:
    queries = read_jsonl(data_dir / "queries.jsonl")
    corpus = read_jsonl(data_dir / "corpus.jsonl")
    qrels = read_jsonl(data_dir / "qrels" / "test.jsonl")

    query_text = {str(row["_id"]): row["text"] for row in queries}
    corpus_text = {}
    for row in corpus:
        title = str(row.get("title", "")).strip()
        text = str(row["text"]).strip()
        corpus_text[str(row["_id"])] = f"{title}\n{text}" if title else text

    relevance: dict[str, dict[str, float]] = defaultdict(dict)
    for row in qrels:
        relevance[str(row["query-id"])][str(row["corpus-id"])] = float(row["score"])

    model = SentenceTransformer(str(model_path))
    corpus_ids = list(corpus_text)
    corpus_embeddings = model.encode(
        [corpus_text[cid] for cid in corpus_ids],
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    recalls = {100: [], 200: [], 500: [], 1000: []}
    ndcg10 = []
    average_precisions = []

    for qid in relevance:
        q_emb = model.encode([query_text[qid]], normalize_embeddings=True, convert_to_numpy=True)
        sims = np.matmul(corpus_embeddings, q_emb[0])
        order = np.argsort(-sims)
        ranked_ids = [corpus_ids[i] for i in order]
        relevant_ids = {cid for cid, score in relevance[qid].items() if score > 0}

        for k in recalls:
            recalls[k].append(len(set(ranked_ids[:k]) & relevant_ids) / max(1, len(relevant_ids)))

        graded = [relevance[qid].get(cid, 0.0) for cid in ranked_ids[:1000]]
        ndcg10.append(ndcg(graded, 10))

        hits = 0
        ap = 0.0
        for rank, cid in enumerate(ranked_ids, start=1):
            if cid in relevant_ids:
                hits += 1
                ap += hits / rank
        average_precisions.append(ap / max(1, len(relevant_ids)))

    metrics = {f"Recall@{k}": round(float(np.mean(v)), 4) for k, v in recalls.items()}
    metrics["NDCG@10"] = round(float(np.mean(ndcg10)), 4)
    metrics["MAP"] = round(float(np.mean(average_precisions)), 4)
    return {
        "model_path": str(model_path),
        "data_dir": str(data_dir),
        "query_count": len(queries),
        "corpus_count": len(corpus),
        "qrels_count": len(qrels),
        "metrics": metrics,
    }

