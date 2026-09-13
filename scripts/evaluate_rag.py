from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.config import settings
from backend.app.rag_benchmark import (
    evaluate_generation_outputs,
    evaluate_reranker_rankings,
    evaluate_retrieval_rankings,
    load_queries_and_qrels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline in retrieval, rerank, and generation layers.")
    parser.add_argument("--dataset", choices=["lecardv1", "lecardv2"], default="lecardv2")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--index-dir", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--retrieval-mode", choices=["hybrid", "vector", "bm25", "vector_fusion"], default=settings.rag_default_retrieval_mode)
    parser.add_argument("--vector-weight", type=float, default=settings.rag_default_vector_weight)
    parser.add_argument("--bm25-weight", type=float, default=settings.rag_default_bm25_weight)
    parser.add_argument("--retrieve-k", type=int, default=settings.rag_default_retrieve_k)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-queries", type=int, default=20)
    parser.add_argument("--use-reranker", action="store_true")
    parser.add_argument("--reranker-model", default=settings.rag_reranker_model)
    parser.add_argument("--evaluate-generation", action="store_true")
    parser.add_argument("--generation-sample-size", type=int, default=10)
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[str, str, str]:
    if args.dataset == "lecardv1":
        data_dir = args.data_dir or str(settings.lecardv1_rag_dir)
        index_dir = args.index_dir or str(settings.lecardv1_rag_index_dir)
        model_path = args.model_path or str(settings.lead_official_checkpoint)
        return data_dir, index_dir, model_path

    data_dir = args.data_dir or str(settings.lecardv2_dir)
    index_dir = args.index_dir or str(settings.rag_index_dir)
    model_path = args.model_path or settings.rag_default_model
    return data_dir, index_dir, model_path


def main() -> None:
    args = parse_args()
    data_dir, index_dir, model_path = resolve_paths(args)
    queries = load_queries_and_qrels(Path(data_dir), max_queries=args.max_queries)
    retrieval_metrics, retrieval_rows, retrieval_latency_ms = evaluate_retrieval_rankings(
        queries,
        model_name_or_path=model_path,
        index_dir=Path(index_dir),
        retrieval_mode=args.retrieval_mode,
        vector_weight=args.vector_weight,
        bm25_weight=args.bm25_weight,
        retrieve_k=args.retrieve_k,
        top_k=args.top_k,
        fusion_base_model_name_or_path=settings.rag_fusion_base_model,
        fusion_base_index_dir=settings.rag_fusion_base_index_dir,
        fusion_base_weight=settings.rag_fusion_base_weight,
    )

    result: dict[str, object] = {
        "dataset": args.dataset,
        "data_dir": data_dir,
        "index_dir": index_dir,
        "retrieval_model": model_path,
        "retrieval_mode": args.retrieval_mode,
        "query_count": len(queries),
        "retrieval_layer": {
            "query_count": len(retrieval_rows),
            "avg_latency_ms": round(retrieval_latency_ms, 2),
            "metrics": retrieval_metrics,
        },
    }

    generation_rows = retrieval_rows
    if args.use_reranker and args.reranker_model:
        rerank_metrics, generation_rows, rerank_latency_ms = evaluate_reranker_rankings(
            retrieval_rows,
            reranker_model=args.reranker_model,
            top_k=args.top_k,
        )
        result["rerank_layer"] = {
            "query_count": len(generation_rows),
            "avg_latency_ms": round(rerank_latency_ms, 2),
            "metrics": rerank_metrics,
            "reranker_model": args.reranker_model,
        }

    if args.evaluate_generation:
        generation_metrics, generation_latency_ms = evaluate_generation_outputs(
            generation_rows,
            api_url=settings.llm_api_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            fallback_model=settings.llm_fallback_model,
            primary_timeout=settings.llm_primary_timeout_seconds,
            fallback_timeout=settings.llm_fallback_timeout_seconds,
            sample_size=min(args.generation_sample_size, len(generation_rows)),
        )
        result["generation_layer"] = {
            "query_count": min(args.generation_sample_size, len(generation_rows)),
            "avg_latency_ms": round(generation_latency_ms, 2),
            "metrics": generation_metrics,
        }

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

