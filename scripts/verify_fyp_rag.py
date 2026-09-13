from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.config import settings
from backend.app.rag import load_manifest, search_index
from backend.app.rag_benchmark import evaluate_retrieval_rankings, load_queries_and_qrels


BASELINE = {
    "doc_recall@5": 0.1409,
    "doc_mrr@5": 0.8324,
    "doc_ndcg@5": 0.6917,
    "doc_hit_rate@5": 0.9245,
}


def assert_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} is missing: {path}")


def main() -> None:
    default_model = Path(settings.rag_default_model)
    default_index = settings.rag_index_dir
    fusion_base_model = Path(settings.rag_fusion_base_model)
    fusion_base_index = settings.rag_fusion_base_index_dir

    assert settings.rag_default_retrieval_mode == "vector_fusion", (
        "Expected RAG_RETRIEVAL_MODE=vector_fusion, "
        f"got {settings.rag_default_retrieval_mode!r}"
    )
    assert_exists(default_model, "Default fine-tuned model")
    assert_exists(default_index / "manifest.json", "Default FAISS index manifest")
    assert_exists(fusion_base_model, "Fusion base bge-m3 model")
    assert_exists(fusion_base_index / "manifest.json", "Fusion base FAISS index manifest")

    default_manifest = load_manifest(default_index)
    base_manifest = load_manifest(fusion_base_index)
    queries = load_queries_and_qrels(settings.lecardv2_dir, max_queries=20)
    if not queries:
        raise RuntimeError(f"No LeCaRDv2 benchmark queries found under {settings.lecardv2_dir}")

    retrieval = search_index(
        query=queries[0]["query"],
        model_name_or_path=settings.rag_default_model,
        index_dir=settings.rag_index_dir,
        top_k=5,
        retrieval_mode=settings.rag_default_retrieval_mode,
        vector_weight=settings.rag_default_vector_weight,
        bm25_weight=settings.rag_default_bm25_weight,
        fusion_base_model_name_or_path=settings.rag_fusion_base_model,
        fusion_base_index_dir=settings.rag_fusion_base_index_dir,
        fusion_base_weight=settings.rag_fusion_base_weight,
    )
    if retrieval["backend"] != "vector_fusion" or not retrieval["hits"]:
        raise RuntimeError("Default vector_fusion retrieval did not return hits.")

    metrics, _, latency_ms = evaluate_retrieval_rankings(
        queries,
        model_name_or_path=settings.rag_default_model,
        index_dir=settings.rag_index_dir,
        retrieval_mode=settings.rag_default_retrieval_mode,
        vector_weight=settings.rag_default_vector_weight,
        bm25_weight=settings.rag_default_bm25_weight,
        retrieve_k=settings.rag_default_retrieve_k,
        top_k=5,
        fusion_base_model_name_or_path=settings.rag_fusion_base_model,
        fusion_base_index_dir=settings.rag_fusion_base_index_dir,
        fusion_base_weight=settings.rag_fusion_base_weight,
    )

    result = {
        "status": "ok",
        "retrieval_mode": settings.rag_default_retrieval_mode,
        "default_model": str(default_model),
        "default_index": str(default_index),
        "default_index_chunks": default_manifest.get("chunk_count"),
        "fusion_base_model": str(fusion_base_model),
        "fusion_base_index": str(fusion_base_index),
        "fusion_base_index_chunks": base_manifest.get("chunk_count"),
        "fusion_base_weight": settings.rag_fusion_base_weight,
        "smoke_hit_count": len(retrieval["hits"]),
        "benchmark_query_count": len(queries),
        "benchmark_latency_ms": round(latency_ms, 2),
        "benchmark_metrics_sample20": metrics,
        "full_benchmark_baseline_reference": BASELINE,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

