from __future__ import annotations

import math
import re
import time
from pathlib import Path
from typing import Any

from .llm import answer_with_context
from .rag import (
    aggregate_hits_by_doc,
    build_retrieval_query,
    read_jsonl,
    rerank_hits,
    rerank_hits_with_biencoder_fusion,
    search_index,
)


SELECTED_DOC_PATTERN = re.compile(r"最相似案例是[:：]\s*([A-Za-z0-9_#-]+)")
CITATION_PATTERN = re.compile(r"\[([A-Za-z0-9_#-]+)\]")


def load_queries_and_qrels(data_dir: Path, *, max_queries: int | None) -> list[dict[str, Any]]:
    queries = read_jsonl(data_dir / "queries.jsonl")
    qrel_rows = read_jsonl(data_dir / "qrels" / "test.jsonl")

    qrels_by_query: dict[str, dict[str, float]] = {}
    for row in qrel_rows:
        query_id = str(row.get("query-id", ""))
        corpus_id = str(row.get("corpus-id", ""))
        score = float(row.get("score", 0.0))
        if not query_id or not corpus_id:
            continue
        qrels_by_query.setdefault(query_id, {})[corpus_id] = score

    items: list[dict[str, Any]] = []
    for row in queries:
        query_id = str(row.get("_id", ""))
        if query_id not in qrels_by_query:
            continue
        items.append(
            {
                "query_id": query_id,
                "query": str(row.get("text", "")),
                "qrels": qrels_by_query[query_id],
            }
        )
        if max_queries is not None and len(items) >= max_queries:
            break
    return items


def unique_doc_ids(hits: list[dict[str, Any]], *, top_k: int) -> list[str]:
    doc_ids: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        doc_id = str(hit.get("doc_id", ""))
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        doc_ids.append(doc_id)
        if len(doc_ids) >= top_k:
            break
    return doc_ids


def merge_case_hits(hits: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    if not hits:
        return []
    merged = aggregate_hits_by_doc(hits, top_k=top_k)
    for hit in merged:
        hit["text"] = str(hit.get("text", "")).strip()
    return merged


def reciprocal_rank(ranked_docs: list[str], relevant_docs: set[str]) -> float:
    for idx, doc_id in enumerate(ranked_docs, start=1):
        if doc_id in relevant_docs:
            return 1.0 / idx
    return 0.0


def dcg_at_k(ranked_docs: list[str], qrels: dict[str, float], k: int) -> float:
    score = 0.0
    for idx, doc_id in enumerate(ranked_docs[:k], start=1):
        rel = float(qrels.get(doc_id, 0.0))
        if rel <= 0:
            continue
        score += (2.0**rel - 1.0) / math.log2(idx + 1.0)
    return score


def ndcg_at_k(ranked_docs: list[str], qrels: dict[str, float], k: int) -> float:
    ideal_docs = [doc_id for doc_id, _ in sorted(qrels.items(), key=lambda item: item[1], reverse=True)]
    ideal = dcg_at_k(ideal_docs, qrels, k)
    if ideal <= 0:
        return 0.0
    return dcg_at_k(ranked_docs, qrels, k) / ideal


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate_retrieval_metrics(per_query: list[dict[str, float]], *, top_k: int) -> dict[str, float]:
    recalls = [row["recall"] for row in per_query]
    mrrs = [row["mrr"] for row in per_query]
    ndcgs = [row["ndcg"] for row in per_query]
    hit_rates = [row["hit"] for row in per_query]
    uniques = [row["unique_docs"] for row in per_query]
    duplicates = [row["duplicate_ratio"] for row in per_query]
    return {
        f"doc_recall@{top_k}": round(mean(recalls), 4),
        f"doc_mrr@{top_k}": round(mean(mrrs), 4),
        f"doc_ndcg@{top_k}": round(mean(ndcgs), 4),
        f"doc_hit_rate@{top_k}": round(mean(hit_rates), 4),
        f"avg_unique_docs@{top_k}": round(mean(uniques), 2),
        f"avg_duplicate_ratio@{top_k}": round(mean(duplicates), 4),
    }


def evaluate_retrieval_rankings(
    queries: list[dict[str, Any]],
    *,
    model_name_or_path: str,
    index_dir: Path,
    retrieval_mode: str,
    vector_weight: float,
    bm25_weight: float,
    retrieve_k: int,
    top_k: int,
    fusion_base_model_name_or_path: str | None = None,
    fusion_base_index_dir: Path | None = None,
    fusion_base_weight: float = 0.7,
) -> tuple[dict[str, float], list[dict[str, Any]], float]:
    per_query_metrics: list[dict[str, float]] = []
    retrieval_rows: list[dict[str, Any]] = []
    latencies_ms: list[float] = []

    for row in queries:
        started = time.perf_counter()
        retrieval = search_index(
            query=row["query"],
            model_name_or_path=model_name_or_path,
            index_dir=index_dir,
            top_k=retrieve_k,
            retrieval_mode=retrieval_mode,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            fusion_base_model_name_or_path=fusion_base_model_name_or_path,
            fusion_base_index_dir=fusion_base_index_dir,
            fusion_base_weight=fusion_base_weight,
        )
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
        hits = merge_case_hits(retrieval["hits"], top_k=top_k)
        ranked_docs = unique_doc_ids(hits, top_k=top_k)
        relevant_docs = {doc_id for doc_id, rel in row["qrels"].items() if rel > 0}
        relevant_found = sum(1 for doc_id in ranked_docs if doc_id in relevant_docs)
        total_returned = min(len(hits), top_k)

        per_query_metrics.append(
            {
                "recall": relevant_found / max(len(relevant_docs), 1),
                "mrr": reciprocal_rank(ranked_docs, relevant_docs),
                "ndcg": ndcg_at_k(ranked_docs, row["qrels"], top_k),
                "hit": 1.0 if relevant_found > 0 else 0.0,
                "unique_docs": float(len(ranked_docs)),
                "duplicate_ratio": 0.0 if total_returned == 0 else max(total_returned - len(ranked_docs), 0) / total_returned,
            }
        )
        retrieval_rows.append(
            {
                "query_id": row["query_id"],
                "query": row["query"],
                "qrels": row["qrels"],
                "candidate_hits": retrieval["hits"],
                "hits": hits,
            }
        )

    return aggregate_retrieval_metrics(per_query_metrics, top_k=top_k), retrieval_rows, mean(latencies_ms)


def evaluate_reranker_rankings(
    retrieval_rows: list[dict[str, Any]],
    *,
    reranker_model: str,
    top_k: int,
    reranker_type: str = "cross_encoder",
    fusion_weight: float = 0.15,
    rerank_window: int = 10,
) -> tuple[dict[str, float], list[dict[str, Any]], float]:
    per_query_metrics: list[dict[str, float]] = []
    reranked_rows: list[dict[str, Any]] = []
    latencies_ms: list[float] = []

    for row in retrieval_rows:
        started = time.perf_counter()
        retrieval_query = build_retrieval_query(row["query"])
        candidate_hits = row.get("candidate_hits") or row["hits"]
        rerank_limit = max(len(candidate_hits), top_k)
        if reranker_type == "bi_encoder_fusion":
            reranked_hits = rerank_hits_with_biencoder_fusion(
                query=row["query"],
                hits=candidate_hits,
                reranker_model=reranker_model,
                top_k=rerank_limit,
                fusion_weight=fusion_weight,
                rerank_window=rerank_window,
            )
        else:
            reranked_hits = rerank_hits(
                query=retrieval_query,
                hits=candidate_hits,
                reranker_model=reranker_model,
                top_k=rerank_limit,
            )
        reranked_hits = merge_case_hits(reranked_hits, top_k=top_k)
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
        ranked_docs = unique_doc_ids(reranked_hits, top_k=top_k)
        relevant_docs = {doc_id for doc_id, rel in row["qrels"].items() if rel > 0}
        relevant_found = sum(1 for doc_id in ranked_docs if doc_id in relevant_docs)
        total_returned = min(len(reranked_hits), top_k)

        per_query_metrics.append(
            {
                "recall": relevant_found / max(len(relevant_docs), 1),
                "mrr": reciprocal_rank(ranked_docs, relevant_docs),
                "ndcg": ndcg_at_k(ranked_docs, row["qrels"], top_k),
                "hit": 1.0 if relevant_found > 0 else 0.0,
                "unique_docs": float(len(ranked_docs)),
                "duplicate_ratio": 0.0 if total_returned == 0 else max(total_returned - len(ranked_docs), 0) / total_returned,
            }
        )
        reranked_rows.append(
            {
                "query_id": row["query_id"],
                "query": row["query"],
                "qrels": row["qrels"],
                "hits": reranked_hits,
            }
        )

    return aggregate_retrieval_metrics(per_query_metrics, top_k=top_k), reranked_rows, mean(latencies_ms)


def extract_selected_doc_id(answer: str) -> str:
    match = SELECTED_DOC_PATTERN.search(answer)
    return match.group(1).strip() if match else ""


def extract_reference_citation(answer: str) -> str:
    matches = CITATION_PATTERN.findall(answer)
    return matches[-1] if matches else ""


def evaluate_generation_outputs(
    rows: list[dict[str, Any]],
    *,
    api_url: str,
    api_key: str,
    model: str,
    fallback_model: str | None,
    primary_timeout: int,
    fallback_timeout: int,
    sample_size: int,
) -> tuple[dict[str, float], float]:
    scored_rows = rows[:sample_size]
    latencies_ms: list[float] = []
    selected_doc_relevant: list[float] = []
    selected_doc_in_hits: list[float] = []
    citation_matches_doc: list[float] = []
    citation_relevant: list[float] = []
    similar_points_present: list[float] = []
    answer_nonempty: list[float] = []

    for row in scored_rows:
        started = time.perf_counter()
        _, answer = answer_with_context(
            query=row["query"],
            hits=row["hits"][: max(1, min(len(row["hits"]), 5))],
            api_url=api_url,
            api_key=api_key,
            model=model,
            fallback_model=fallback_model,
            primary_timeout=primary_timeout,
            fallback_timeout=fallback_timeout,
        )
        latencies_ms.append((time.perf_counter() - started) * 1000.0)

        relevant_docs = {doc_id for doc_id, rel in row["qrels"].items() if rel > 0}
        selected_doc_id = extract_selected_doc_id(answer)
        reference = extract_reference_citation(answer)
        reference_doc_id = reference.split("#", 1)[0] if reference else ""
        hit_doc_ids = {str(hit["doc_id"]) for hit in row["hits"]}

        selected_doc_relevant.append(1.0 if selected_doc_id in relevant_docs else 0.0)
        selected_doc_in_hits.append(1.0 if selected_doc_id in hit_doc_ids else 0.0)
        citation_matches_doc.append(1.0 if reference_doc_id and reference_doc_id == selected_doc_id else 0.0)
        citation_relevant.append(1.0 if reference_doc_id in relevant_docs else 0.0)
        similar_points_present.append(1.0 if "相似点" in answer else 0.0)
        answer_nonempty.append(1.0 if answer.strip() else 0.0)

    metrics = {
        "selected_doc_relevant_rate": round(mean(selected_doc_relevant), 4),
        "selected_doc_in_candidates_rate": round(mean(selected_doc_in_hits), 4),
        "citation_matches_selected_doc_rate": round(mean(citation_matches_doc), 4),
        "citation_relevant_rate": round(mean(citation_relevant), 4),
        "similar_points_present_rate": round(mean(similar_points_present), 4),
        "answer_nonempty_rate": round(mean(answer_nonempty), 4),
    }
    return metrics, mean(latencies_ms)

