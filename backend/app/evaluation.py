from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List


def graded_ndcg_at_k(ranks: List[int], ideal_ranks: List[int], k: int) -> float:
    padded = ranks[:k] + [0] * max(0, k - len(ranks))
    ideal = sorted(ideal_ranks, reverse=True)[:k]
    ideal = ideal + [0] * max(0, k - len(ideal))
    dcg = sum(score / math.log2(i + 2) for i, score in enumerate(padded[:k]))
    idcg = sum(score / math.log2(i + 2) for i, score in enumerate(ideal[:k]))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def run_lecard_official_eval(label_path: Path, rank_path: Path, max_samples: int | None = None) -> Dict[str, float | int]:
    labels = json.loads(label_path.read_text(encoding="utf-8"))
    ranks_by_query = json.loads(rank_path.read_text(encoding="utf-8"))
    query_ids = list(labels.keys())
    if max_samples is not None:
        query_ids = query_ids[:max_samples]

    p5: List[float] = []
    p10: List[float] = []
    maps: List[float] = []
    ndcg10: List[float] = []
    ndcg20: List[float] = []
    ndcg30: List[float] = []

    for query_id in query_ids:
        query_labels = {str(k): int(v) for k, v in labels[query_id].items()}
        ranked_ids = [str(case_id) for case_id in ranks_by_query[query_id] if str(case_id) in query_labels]
        relevant_ids = {case_id for case_id, score in query_labels.items() if score == 3}

        p5.append(len([case_id for case_id in ranked_ids[:5] if case_id in relevant_ids]) / 5.0)
        p10.append(len([case_id for case_id in ranked_ids[:10] if case_id in relevant_ids]) / 10.0)

        average_precision = 0.0
        hit_count = 0
        for rank, case_id in enumerate(ranked_ids, start=1):
            if case_id in relevant_ids:
                hit_count += 1
                average_precision += hit_count / rank
        maps.append(average_precision / max(1, len(relevant_ids)))

        graded_ranks = [query_labels[case_id] for case_id in ranked_ids]
        ideal_ranks = list(query_labels.values())
        ndcg10.append(graded_ndcg_at_k(graded_ranks, ideal_ranks, 10))
        ndcg20.append(graded_ndcg_at_k(graded_ranks, ideal_ranks, 20))
        ndcg30.append(graded_ndcg_at_k(graded_ranks, ideal_ranks, 30))

    n = max(1, len(query_ids))
    return {
        "P@5": round(sum(p5) / n, 4),
        "P@10": round(sum(p10) / n, 4),
        "MAP": round(sum(maps) / n, 4),
        "NDCG@10": round(sum(ndcg10) / n, 4),
        "NDCG@20": round(sum(ndcg20) / n, 4),
        "NDCG@30": round(sum(ndcg30) / n, 4),
        "evaluated_samples": len(query_ids),
    }

