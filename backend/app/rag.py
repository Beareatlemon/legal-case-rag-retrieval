from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import re
from collections import Counter
import time
from typing import Any

import numpy as np
import faiss  # type: ignore
from sentence_transformers import CrossEncoder, SentenceTransformer


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


LEGAL_SECTION_MARKERS: list[tuple[str, str]] = [
    ("公诉机关指控", "prosecution_claim"),
    ("检察院指控", "prosecution_claim"),
    ("起诉书指控", "prosecution_claim"),
    ("诉称", "claims"),
    ("辩称", "defense"),
    ("辩护意见", "defense"),
    ("经审理查明", "facts_found"),
    ("经查", "facts_found"),
    ("本院查明", "facts_found"),
    ("另查明", "supplemental_facts"),
    ("上述事实", "evidence_summary"),
    ("上述证据", "evidence_summary"),
    ("证据如下", "evidence"),
    ("本院认为", "court_opinion"),
    ("本院经审理认为", "court_opinion"),
    ("法院认为", "court_opinion"),
    ("依照", "legal_basis"),
    ("判决如下", "judgment"),
    ("裁定如下", "judgment"),
    ("综上", "summary"),
]
RETRIEVAL_QUERY_SECTION_PRIORITY = [
    "facts_found",
    "prosecution_claim",
    "supplemental_facts",
    "evidence_summary",
    "court_opinion",
    "judgment",
    "legal_basis",
    "claims",
    "general",
    "defense",
]

TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z0-9_]+")
NUMERIC_TOKEN_PATTERN = re.compile(r"^\d+(?:\.\d+)?$")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[。！？!?；;])")
BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60.0
HYBRID_OVERLAP_BOOST = 1.05
HYBRID_BM25_ONLY_PENALTY = 0.78
HYBRID_ALIGNMENT_BOOST = 0.04
HYBRID_ALIGNMENT_PENALTY = 0.85
HYBRID_VECTOR_RRF_SHARE = 0.75
HYBRID_VECTOR_SCORE_SHARE = 0.25
HYBRID_BM25_MATCH_BONUS = 0.18
HYBRID_BM25_NOVELTY_BONUS = 0.06

WEAK_BM25_TOKENS = {
    "被告",
    "被告人",
    "庭审",
    "审理",
    "异议",
    "证据",
    "证言",
    "供述",
    "记录",
    "报告",
    "证明",
    "证实",
    "机关",
    "公安",
    "检察院",
    "法院",
    "现场",
    "交易",
    "调查",
    "接受",
    "情况",
    "行为",
    "本院",
    "认为",
    "如下",
    "事实",
    "依据",
    "元",
    "年",
    "月",
    "日",
    "次",
    "某",
}


@lru_cache(maxsize=4)
def get_sentence_transformer(model_name_or_path: str) -> SentenceTransformer:
    return SentenceTransformer(model_name_or_path)


@lru_cache(maxsize=2)
def get_cross_encoder(model_name_or_path: str) -> CrossEncoder:
    return CrossEncoder(model_name_or_path)

def same_retrieval_model_family(index_model_path: str, requested_model_path: str) -> bool:
    if not index_model_path or not requested_model_path:
        return False
    return Path(index_model_path).as_posix().lower() == Path(requested_model_path).as_posix().lower()


def _format_minutes(seconds: float) -> str:
    return f"{max(seconds, 0.0) / 60:.1f}m"


def encode_context_embeddings(model_name_or_path: str, texts: list[str], *, batch_size: int) -> np.ndarray:
    model = get_sentence_transformer(model_name_or_path)
    total_texts = len(texts)
    total_batches = (total_texts + batch_size - 1) // batch_size if batch_size > 0 else 0
    print(
        "[rag-encode] start "
        f"model={model_name_or_path}, texts={total_texts}, batch_size={batch_size}, total_batches={total_batches}"
    )
    started = time.perf_counter()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")
    print(f"[rag-encode] done elapsed={_format_minutes(time.perf_counter() - started)}")
    return vectors


def encode_query_embedding(model_name_or_path: str, query: str) -> np.ndarray:
    retrieval_query = build_retrieval_query(query)
    model = get_sentence_transformer(model_name_or_path)
    return model.encode([retrieval_query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")[0]


def normalize_legal_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    normalized = normalize_legal_text(text)
    if not normalized:
        return []
    sentences = [piece.strip() for piece in SENTENCE_SPLIT_PATTERN.split(normalized) if piece.strip()]
    return sentences or [normalized]


def tokenize_for_bm25(text: str) -> list[str]:
    normalized = normalize_legal_text(text).lower()
    tokens: list[str] = []
    for piece in TOKEN_PATTERN.findall(normalized):
        if not piece:
            continue
        if all("\u4e00" <= ch <= "\u9fff" for ch in piece):
            if len(piece) == 1:
                if piece not in WEAK_BM25_TOKENS:
                    tokens.append(piece)
                continue
            if len(piece) <= 8:
                tokens.append(piece)
            tokens.extend(piece[i : i + 2] for i in range(len(piece) - 1))
            if len(piece) >= 3:
                tokens.extend(piece[i : i + 3] for i in range(len(piece) - 2))
        else:
            tokens.append(piece)
    return tokens


def bm25_token_weight(token: str) -> float:
    if not token:
        return 0.0
    if token in WEAK_BM25_TOKENS:
        return 0.15
    if NUMERIC_TOKEN_PATTERN.fullmatch(token):
        return 0.1
    if any(ch.isdigit() for ch in token):
        return 0.2
    if len(token) == 1:
        return 0.1
    if "元" in token or "年" in token or "月" in token or "日" in token:
        return 0.25
    if "证据" in token or "证言" in token or "供述" in token or "报告" in token:
        return 0.3
    return 1.0


def extract_salient_query_terms(query: str, *, limit: int = 8) -> list[str]:
    normalized = normalize_legal_text(query).lower()
    terms: list[str] = []
    seen: set[str] = set()

    for piece in TOKEN_PATTERN.findall(normalized):
        if not piece:
            continue
        if all("\u4e00" <= ch <= "\u9fff" for ch in piece):
            max_n = min(4, len(piece))
            for n in range(max_n, 1, -1):
                for i in range(len(piece) - n + 1):
                    term = piece[i : i + n]
                    if term in seen:
                        continue
                    if any(ch in {"年", "月", "日", "元", "某", "次"} for ch in term):
                        continue
                    if term in WEAK_BM25_TOKENS:
                        continue
                    if "证据" in term or "证言" in term or "供述" in term or "报告" in term:
                        continue
                    seen.add(term)
                    terms.append(term)
        else:
            if piece in seen:
                continue
            if bm25_token_weight(piece) < 1.0:
                continue
            seen.add(piece)
            terms.append(piece)
    terms.sort(key=len, reverse=True)
    return terms[:limit]


def score_query_alignment(query_terms: list[str], text: str) -> int:
    if not query_terms:
        return 0
    normalized = normalize_legal_text(text).lower()
    return sum(1 for term in query_terms if term in normalized)


def chunk_quality_penalty(text: str) -> float:
    normalized = normalize_legal_text(text)
    if not normalized:
        return 0.5

    length = len(normalized)
    digit_count = sum(ch.isdigit() for ch in normalized)
    semicolon_count = normalized.count("；") + normalized.count(";")
    comma_enum_count = normalized.count("、")
    digit_ratio = digit_count / max(length, 1)

    penalty = 1.0
    if digit_ratio >= 0.08:
        penalty *= 0.8
    if digit_ratio >= 0.15:
        penalty *= 0.7
    if semicolon_count >= 3:
        penalty *= 0.75
    if semicolon_count >= 6:
        penalty *= 0.7
    if comma_enum_count >= 10 and digit_count >= 6:
        penalty *= 0.75
    return max(0.3, penalty)


def _doc_level_score(hit: dict[str, Any]) -> float:
    return float(hit.get("score") or 0.0)


def _normalize_doc_scores(hits: list[dict[str, Any]], *, score_key: str) -> dict[str, float]:
    doc_scores: dict[str, float] = {}
    for hit in hits:
        doc_id = str(hit.get("doc_id", ""))
        if not doc_id:
            continue
        score = float(hit.get(score_key) or hit.get("score") or 0.0)
        current = doc_scores.get(doc_id)
        if current is None or score > current:
            doc_scores[doc_id] = score

    if not doc_scores:
        return {}

    values = list(doc_scores.values())
    min_score = min(values)
    max_score = max(values)
    if max_score - min_score <= 1e-6:
        return {doc_id: 1.0 for doc_id in doc_scores}
    return {doc_id: (score - min_score) / (max_score - min_score) for doc_id, score in doc_scores.items()}


def _normalize_array(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    std = float(scores.std())
    if std <= 1e-9:
        return scores - float(scores.mean())
    return (scores - float(scores.mean())) / std


def _hit_text_for_biencoder(hit: dict[str, Any]) -> str:
    title = str(hit.get("title", "") or "").strip()
    text = str(hit.get("text", "") or "").strip()
    return f"{title}\n{text}".strip() if title else text


def _merge_reranked_window(scored_window: list[dict[str, Any]], original_hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored_chunk_ids = {str(hit.get("chunk_id", "")) for hit in scored_window}
    return scored_window + [hit for hit in original_hits if str(hit.get("chunk_id", "")) not in scored_chunk_ids]


def aggregate_hits_by_doc(hits: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for hit in hits:
        doc_id = str(hit["doc_id"])
        bucket = grouped.setdefault(
            doc_id,
            {
                "doc_id": doc_id,
                "hits": [],
                "score_sum": 0.0,
                "best_score": float("-inf"),
                "best_hit": None,
                "vector_best": None,
                "bm25_best": None,
            },
        )
        bucket["hits"].append(hit)
        score = _doc_level_score(hit)
        bucket["score_sum"] += score
        if score > bucket["best_score"]:
            bucket["best_score"] = score
            bucket["best_hit"] = hit

        vector_score = hit.get("vector_score")
        if vector_score is not None:
            bucket["vector_best"] = max(float(vector_score), float(bucket["vector_best"] or float("-inf")))
        bm25_score = hit.get("bm25_score")
        if bm25_score is not None:
            bucket["bm25_best"] = max(float(bm25_score), float(bucket["bm25_best"] or float("-inf")))

    doc_hits: list[dict[str, Any]] = []
    for bucket in grouped.values():
        best_hit = dict(bucket["best_hit"])
        supporting_hits = sorted(bucket["hits"], key=lambda row: _doc_level_score(row), reverse=True)
        support_count = len(supporting_hits)
        # Favor documents with multiple supporting chunks, but cap the bonus.
        doc_score = bucket["best_score"] + min(max(support_count - 1, 0), 3) * 0.08
        best_hit["score"] = round(float(doc_score), 6)
        best_hit["retrieval_score"] = best_hit["score"]
        best_hit["support_count"] = support_count
        best_hit["result_type"] = "case"
        best_hit["supporting_chunks"] = [
            {
                "chunk_id": str(item.get("chunk_id", "")),
                "section": item.get("section"),
                "score": item.get("score"),
                "retrieval_score": item.get("retrieval_score"),
                "rerank_score": item.get("rerank_score"),
                "vector_score": item.get("vector_score"),
                "bm25_score": item.get("bm25_score"),
                "citation": item.get("citation"),
                "text": item.get("text", ""),
            }
            for item in supporting_hits[:5]
        ]
        best_hit["vector_score"] = (
            round(float(bucket["vector_best"]), 4) if bucket["vector_best"] not in {None, float("-inf")} else None
        )
        best_hit["bm25_score"] = (
            round(float(bucket["bm25_best"]), 4) if bucket["bm25_best"] not in {None, float("-inf")} else None
        )
        doc_hits.append(best_hit)

    doc_hits.sort(key=lambda row: row["score"], reverse=True)
    final_hits = doc_hits[:top_k]
    for rank, hit in enumerate(final_hits, start=1):
        hit["rank"] = rank
    return final_hits


def split_long_text(text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    step = max(1, chunk_size - chunk_overlap)
    while start < len(normalized):
        end = min(len(normalized), start + chunk_size)
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        start += step
    return chunks


def guess_section_label(text: str) -> str:
    for marker, label in LEGAL_SECTION_MARKERS:
        if marker in text:
            return label
    return "general"


def split_legal_sections(text: str) -> list[dict[str, str]]:
    normalized = normalize_legal_text(text)
    if not normalized:
        return []

    matches: list[tuple[int, str]] = []
    for marker, label in LEGAL_SECTION_MARKERS:
        start = 0
        while True:
            idx = normalized.find(marker, start)
            if idx == -1:
                break
            matches.append((idx, label))
            start = idx + len(marker)

    unique_matches: list[tuple[int, str]] = []
    seen_positions: set[int] = set()
    for idx, label in sorted(matches, key=lambda item: item[0]):
        if idx in seen_positions:
            continue
        seen_positions.add(idx)
        unique_matches.append((idx, label))

    if not unique_matches or unique_matches[0][0] > 0:
        unique_matches.insert(0, (0, guess_section_label(normalized[: min(len(normalized), 80)])))

    sections: list[dict[str, str]] = []
    for i, (start_idx, label) in enumerate(unique_matches):
        end_idx = unique_matches[i + 1][0] if i + 1 < len(unique_matches) else len(normalized)
        piece = normalized[start_idx:end_idx].strip()
        if piece:
            sections.append({"section": label, "text": piece})
    return sections


def build_retrieval_query(
    text: str,
    *,
    max_chars: int = 768,
    max_sentences: int = 10,
) -> str:
    normalized = normalize_legal_text(text)
    if not normalized:
        return ""
    if len(normalized) <= max_chars:
        return normalized

    sections = split_legal_sections(normalized)
    if not sections:
        sections = [{"section": "general", "text": normalized}]

    sections_by_label: dict[str, list[str]] = {}
    for section in sections:
        sections_by_label.setdefault(section["section"], []).append(section["text"])

    selected_sentences: list[str] = []
    seen_sentences: set[str] = set()

    def add_sentence(sentence: str) -> None:
        cleaned = sentence.strip()
        if not cleaned or cleaned in seen_sentences:
            return
        selected_sentences.append(cleaned)
        seen_sentences.add(cleaned)

    for label in RETRIEVAL_QUERY_SECTION_PRIORITY:
        for section_text in sections_by_label.get(label, []):
            for sentence in split_sentences(section_text):
                add_sentence(sentence)
                if len(selected_sentences) >= max_sentences:
                    break
            if len(selected_sentences) >= max_sentences:
                break
        if len(selected_sentences) >= max_sentences:
            break

    if not selected_sentences:
        for sentence in split_sentences(normalized):
            add_sentence(sentence)
            if len(selected_sentences) >= max_sentences:
                break

    result_parts: list[str] = []
    current_length = 0
    for sentence in selected_sentences:
        separator_len = 1 if result_parts else 0
        if current_length + len(sentence) + separator_len > max_chars:
            remaining = max_chars - current_length - separator_len
            if remaining > 20:
                result_parts.append(sentence[:remaining].rstrip())
            break
        result_parts.append(sentence)
        current_length += len(sentence) + separator_len

    result = "\n".join(result_parts).strip()
    if not result:
        return normalized[:max_chars].strip()
    return result


def chunk_legal_document(text: str, *, chunk_size: int, chunk_overlap: int) -> list[dict[str, str]]:
    structured_sections = split_legal_sections(text)
    if not structured_sections:
        return []

    chunks: list[dict[str, str]] = []
    for section in structured_sections:
        section_text = section["text"]
        if len(section_text) <= chunk_size:
            chunks.append(section)
            continue
        for piece in split_long_text(section_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
            chunks.append({"section": section["section"], "text": piece})
    return chunks


def load_corpus_chunks(corpus_path: Path, *, chunk_size: int, chunk_overlap: int) -> tuple[int, list[dict[str, str]]]:
    docs = read_jsonl(corpus_path)
    chunks: list[dict[str, str]] = []
    for row in docs:
        doc_id = str(row.get("_id", ""))
        title = str(row.get("title", "") or "").strip()
        text = str(row.get("text", "") or "").strip()
        full_text = f"{title}\n{text}" if title else text
        for idx, piece in enumerate(chunk_legal_document(full_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)):
            chunks.append(
                {
                    "chunk_id": f"{doc_id}#{idx + 1}",
                    "doc_id": doc_id,
                    "title": title,
                    "section": piece["section"],
                    "text": piece["text"],
                    "citation": f"[{doc_id}#{idx + 1}]",
                }
            )
    return len(docs), chunks


def _faiss_index_path(index_dir: Path) -> Path:
    return index_dir / "faiss.index"


@lru_cache(maxsize=4)
def load_metadata(index_dir_str: str) -> list[dict[str, Any]]:
    return read_jsonl(Path(index_dir_str) / "metadata.jsonl")


@lru_cache(maxsize=2)
def build_bm25_cache(index_dir_str: str) -> dict[str, Any]:
    metadata = load_metadata(index_dir_str)
    postings: dict[str, list[tuple[int, int]]] = {}
    doc_lengths = np.zeros(len(metadata), dtype=np.float32)

    for idx, row in enumerate(metadata):
        tokens = tokenize_for_bm25(row.get("text", ""))
        counts = Counter(tokens)
        doc_lengths[idx] = sum(counts.values())
        for token, freq in counts.items():
            postings.setdefault(token, []).append((idx, freq))

    return {
        "doc_count": len(metadata),
        "avg_doc_length": float(doc_lengths.mean()) if len(doc_lengths) else 0.0,
        "doc_lengths": doc_lengths,
        "postings": postings,
    }


def _resolve_ivf_nlist(chunk_count: int, requested_nlist: int) -> int:
    # IVF should not have more centroids than vectors, and small samples need fewer lists.
    max_reasonable_nlist = max(1, chunk_count // 39)
    return max(1, min(requested_nlist, chunk_count, max_reasonable_nlist))


def _build_faiss_index(
    embeddings: np.ndarray,
    *,
    index_type: str,
    nlist: int,
    hnsw_m: int,
    hnsw_ef_construction: int,
) -> tuple[faiss.Index, dict[str, int | None]]:
    dim = embeddings.shape[1]

    if index_type == "flat":
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        return index, {
            "nlist": None,
            "nprobe": None,
            "hnsw_m": None,
            "hnsw_ef_construction": None,
            "hnsw_ef_search": None,
        }

    if index_type == "ivf":
        actual_nlist = _resolve_ivf_nlist(len(embeddings), nlist)
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, actual_nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.add(embeddings)
        return index, {
            "nlist": actual_nlist,
            "nprobe": None,
            "hnsw_m": None,
            "hnsw_ef_construction": None,
            "hnsw_ef_search": None,
        }

    if index_type == "hnsw":
        index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = hnsw_ef_construction
        index.add(embeddings)
        return index, {
            "nlist": None,
            "nprobe": None,
            "hnsw_m": hnsw_m,
            "hnsw_ef_construction": hnsw_ef_construction,
            "hnsw_ef_search": None,
        }

    raise ValueError(f"Unsupported FAISS index_type: {index_type}")


def _apply_search_params(index: faiss.Index, manifest: dict[str, Any]) -> None:
    index_type = str(manifest.get("index_type", "flat"))
    if index_type == "ivf":
        nprobe = int(manifest.get("nprobe") or 1)
        faiss.extract_index_ivf(index).nprobe = nprobe
    elif index_type == "hnsw":
        ef_search = int(manifest.get("hnsw_ef_search") or 64)
        index.hnsw.efSearch = ef_search


def build_rag_index(
    *,
    model_name_or_path: str,
    corpus_path: Path,
    index_dir: Path,
    index_type: str,
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int,
    nlist: int,
    nprobe: int,
    hnsw_m: int,
    hnsw_ef_construction: int,
    hnsw_ef_search: int,
) -> dict[str, Any]:
    index_dir.mkdir(parents=True, exist_ok=True)
    print(f"[rag-index] loading corpus chunks from {corpus_path}")
    document_count, chunks = load_corpus_chunks(corpus_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise ValueError(f"No chunks were produced from corpus: {corpus_path}")
    print(f"[rag-index] loaded documents={document_count}, chunks={len(chunks)}")

    embed_started = time.perf_counter()
    embeddings = encode_context_embeddings(
        model_name_or_path,
        [chunk["text"] for chunk in chunks],
        batch_size=batch_size,
    )
    print(
        "[rag-index] embeddings_ready "
        f"shape={tuple(embeddings.shape)}, elapsed={_format_minutes(time.perf_counter() - embed_started)}"
    )

    persist_started = time.perf_counter()
    np.save(index_dir / "embeddings.npy", embeddings)
    with (index_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"[rag-index] persisted embeddings+metadata elapsed={_format_minutes(time.perf_counter() - persist_started)}")

    faiss_started = time.perf_counter()
    index, params = _build_faiss_index(
        embeddings,
        index_type=index_type,
        nlist=nlist,
        hnsw_m=hnsw_m,
        hnsw_ef_construction=hnsw_ef_construction,
    )
    if index_type == "ivf":
        faiss.extract_index_ivf(index).nprobe = nprobe
        params["nprobe"] = nprobe
    elif index_type == "hnsw":
        index.hnsw.efSearch = hnsw_ef_search
        params["hnsw_ef_search"] = hnsw_ef_search
    faiss.write_index(index, str(_faiss_index_path(index_dir)))
    print(f"[rag-index] faiss_ready index_type={index_type}, elapsed={_format_minutes(time.perf_counter() - faiss_started)}")

    manifest = {
        "model_path": model_name_or_path,
        "corpus_path": str(corpus_path),
        "index_dir": str(index_dir),
        "backend": "faiss",
        "index_type": index_type,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "document_count": document_count,
        "chunk_count": len(chunks),
        "embedding_dim": int(embeddings.shape[1]),
        **params,
    }
    (index_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[rag-index] manifest_written {index_dir / 'manifest.json'}")
    return manifest


def load_manifest(index_dir: Path) -> dict[str, Any]:
    manifest_path = index_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"RAG index manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def vector_search_index(*, query: str, model_name_or_path: str, index_dir: Path, top_k: int) -> dict[str, Any]:
    manifest = load_manifest(index_dir)
    metadata = load_metadata(str(index_dir.resolve()))
    query_embedding = encode_query_embedding(model_name_or_path, query)
    index_path = _faiss_index_path(index_dir)
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")

    index = faiss.read_index(str(index_path))
    _apply_search_params(index, manifest)
    scores, indices = index.search(query_embedding.reshape(1, -1), top_k)
    scored_hits = zip(indices[0].tolist(), scores[0].tolist(), strict=False)

    hits: list[dict[str, Any]] = []
    for rank, (idx, score) in enumerate(scored_hits, start=1):
        if idx < 0 or idx >= len(metadata):
            continue
        hit = dict(metadata[idx])
        hit["rank"] = rank
        hit["score"] = round(float(score), 4)
        hit["vector_score"] = hit["score"]
        hit["retrieval_score"] = hit["score"]
        hits.append(hit)

    return {
        "manifest": manifest,
        "backend": manifest.get("backend", "faiss"),
        "hits": hits,
    }


def bm25_search_index(*, query: str, index_dir: Path, top_k: int) -> dict[str, Any]:
    manifest = load_manifest(index_dir)
    index_dir_str = str(index_dir.resolve())
    metadata = load_metadata(index_dir_str)
    cache = build_bm25_cache(index_dir_str)
    retrieval_query = build_retrieval_query(query)
    query_tokens = tokenize_for_bm25(retrieval_query)
    if not query_tokens:
        return {"manifest": manifest, "backend": "bm25", "hits": []}

    scores = np.zeros(cache["doc_count"], dtype=np.float32)
    avg_doc_length = max(cache["avg_doc_length"], 1.0)
    doc_lengths = cache["doc_lengths"]
    postings = cache["postings"]
    seen_tokens: set[str] = set()

    for token in query_tokens:
        if token in seen_tokens:
            continue
        seen_tokens.add(token)
        token_weight = bm25_token_weight(token)
        if token_weight <= 0.0:
            continue
        docs = postings.get(token)
        if not docs:
            continue
        doc_freq = len(docs)
        idf = np.log(1.0 + (cache["doc_count"] - doc_freq + 0.5) / (doc_freq + 0.5))
        doc_indices = np.array([doc_idx for doc_idx, _ in docs], dtype=np.int32)
        term_freqs = np.array([freq for _, freq in docs], dtype=np.float32)
        denom = term_freqs + BM25_K1 * (1.0 - BM25_B + BM25_B * doc_lengths[doc_indices] / avg_doc_length)
        scores[doc_indices] += token_weight * idf * (term_freqs * (BM25_K1 + 1.0)) / np.maximum(denom, 1e-6)

    top_indices = np.argpartition(scores, -top_k)[-top_k:]
    sorted_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
    hits: list[dict[str, Any]] = []
    for rank, idx in enumerate(sorted_indices.tolist(), start=1):
        score = float(scores[idx])
        if score <= 0.0:
            continue
        hit = dict(metadata[idx])
        hit["rank"] = rank
        hit["score"] = round(score, 4)
        hit["bm25_score"] = hit["score"]
        hit["retrieval_score"] = hit["score"]
        hits.append(hit)

    return {
        "manifest": manifest,
        "backend": "bm25",
        "hits": hits,
    }


def hybrid_search_index(
    *,
    query: str,
    model_name_or_path: str,
    index_dir: Path,
    top_k: int,
    vector_weight: float,
    bm25_weight: float,
) -> dict[str, Any]:
    retrieval_query = build_retrieval_query(query)
    query_terms = extract_salient_query_terms(retrieval_query)
    vector_result = vector_search_index(
        query=query,
        model_name_or_path=model_name_or_path,
        index_dir=index_dir,
        top_k=top_k,
    )
    bm25_result = bm25_search_index(
        query=retrieval_query,
        index_dir=index_dir,
        top_k=top_k,
    )
    vector_doc_hits = aggregate_hits_by_doc(vector_result["hits"], top_k=max(len(vector_result["hits"]), top_k))
    bm25_doc_hits = aggregate_hits_by_doc(bm25_result["hits"], top_k=max(len(bm25_result["hits"]), top_k))
    vector_norm = _normalize_doc_scores(vector_doc_hits, score_key="vector_score")
    bm25_norm = _normalize_doc_scores(bm25_doc_hits, score_key="bm25_score")

    combined: dict[str, dict[str, Any]] = {}
    vector_doc_ids = {str(hit["doc_id"]) for hit in vector_doc_hits}

    for hit in vector_doc_hits:
        doc_id = str(hit["doc_id"])
        row = dict(hit)
        row["score"] = vector_weight * (
            HYBRID_VECTOR_RRF_SHARE * (1.0 / (RRF_K + hit["rank"]))
            + HYBRID_VECTOR_SCORE_SHARE * vector_norm.get(doc_id, 0.0)
        )
        row["vector_score"] = hit.get("vector_score", hit.get("score"))
        row.setdefault("bm25_score", None)
        combined[doc_id] = row

    novelty_budget = max(1, top_k // 3)
    novelty_used = 0
    for hit in bm25_doc_hits:
        doc_id = str(hit["doc_id"])
        alignment_matches = score_query_alignment(query_terms, hit.get("text", ""))
        bm25_signal = bm25_weight * bm25_norm.get(doc_id, 0.0)

        if doc_id in combined:
            row = combined[doc_id]
            row["bm25_score"] = hit.get("bm25_score", hit.get("score"))
            row["score"] += HYBRID_BM25_MATCH_BONUS * bm25_signal
            if alignment_matches > 0:
                row["score"] += 0.01 * min(alignment_matches, 3)
            continue

        if novelty_used >= novelty_budget:
            continue
        if hit["rank"] > max(3, top_k):
            continue
        if alignment_matches <= 0:
            continue
        if chunk_quality_penalty(hit.get("text", "")) < 0.75:
            continue

        row = dict(hit)
        row["vector_score"] = None
        row["bm25_score"] = hit.get("bm25_score", hit.get("score"))
        row["score"] = HYBRID_BM25_NOVELTY_BONUS * bm25_signal
        combined[doc_id] = row
        novelty_used += 1

    merged_hits = []
    for row in combined.values():
        has_vector = row.get("vector_score") is not None
        has_bm25 = row.get("bm25_score") is not None
        if has_vector and has_bm25:
            row["score"] *= HYBRID_OVERLAP_BOOST
        elif has_bm25 and not has_vector:
            row["score"] *= HYBRID_BM25_ONLY_PENALTY

        alignment_matches = score_query_alignment(query_terms, row.get("text", ""))
        row["alignment_matches"] = alignment_matches
        if alignment_matches == 0 and query_terms:
            row["score"] *= HYBRID_ALIGNMENT_PENALTY
        elif alignment_matches > 0:
            row["score"] *= 1.0 + HYBRID_ALIGNMENT_BOOST * min(alignment_matches, 3)

        row["score"] *= chunk_quality_penalty(row.get("text", ""))
        row["score"] = round(float(row["score"]), 6)
        row["retrieval_score"] = row["score"]
        merged_hits.append(row)

    merged_hits.sort(key=lambda row: row["score"], reverse=True)
    final_hits = merged_hits[:top_k]
    for rank, hit in enumerate(final_hits, start=1):
        hit["rank"] = rank

    return {
        "manifest": vector_result["manifest"],
        "backend": "hybrid",
        "hits": final_hits,
    }


def vector_fusion_search_index(
    *,
    query: str,
    model_name_or_path: str,
    index_dir: Path,
    top_k: int,
    base_model_name_or_path: str,
    base_index_dir: Path,
    base_weight: float,
) -> dict[str, Any]:
    primary_result = vector_search_index(
        query=query,
        model_name_or_path=model_name_or_path,
        index_dir=index_dir,
        top_k=top_k,
    )
    base_result = vector_search_index(
        query=query,
        model_name_or_path=base_model_name_or_path,
        index_dir=base_index_dir,
        top_k=top_k,
    )
    primary_doc_hits = aggregate_hits_by_doc(primary_result["hits"], top_k=max(len(primary_result["hits"]), top_k))
    base_doc_hits = aggregate_hits_by_doc(base_result["hits"], top_k=max(len(base_result["hits"]), top_k))
    primary_norm = _normalize_doc_scores(primary_doc_hits, score_key="score")
    base_norm = _normalize_doc_scores(base_doc_hits, score_key="score")

    clamped_base_weight = min(max(base_weight, 0.0), 1.0)
    finetuned_weight = 1.0 - clamped_base_weight
    combined: dict[str, dict[str, Any]] = {}

    for hit in primary_doc_hits:
        doc_id = str(hit["doc_id"])
        row = dict(hit)
        row["score"] = finetuned_weight * primary_norm.get(doc_id, 0.0)
        row["vector_score"] = row["score"]
        row["primary_vector_score"] = hit.get("vector_score", hit.get("score"))
        row["base_vector_score"] = None
        combined[doc_id] = row

    for hit in base_doc_hits:
        doc_id = str(hit["doc_id"])
        row = combined.get(doc_id)
        if row is None:
            row = dict(hit)
            row["score"] = 0.0
            row["primary_vector_score"] = None
            combined[doc_id] = row
        row["score"] += clamped_base_weight * base_norm.get(doc_id, 0.0)
        row["base_vector_score"] = hit.get("vector_score", hit.get("score"))
        row["vector_score"] = row["score"]

    merged_hits = list(combined.values())
    merged_hits.sort(key=lambda row: row["score"], reverse=True)
    final_hits = merged_hits[:top_k]
    for rank, hit in enumerate(final_hits, start=1):
        hit["rank"] = rank
        hit["score"] = round(float(hit["score"]), 6)
        hit["vector_score"] = hit["score"]
        hit["retrieval_score"] = hit["score"]

    return {
        "manifest": {
            "primary_model_path": str(primary_result["manifest"].get("model_path", model_name_or_path)),
            "primary_index_dir": str(index_dir),
            "base_model_path": str(base_result["manifest"].get("model_path", base_model_name_or_path)),
            "base_index_dir": str(base_index_dir),
            "base_weight": clamped_base_weight,
        },
        "backend": "vector_fusion",
        "hits": final_hits,
    }


def search_index(
    *,
    query: str,
    model_name_or_path: str,
    index_dir: Path,
    top_k: int,
    retrieval_mode: str,
    vector_weight: float,
    bm25_weight: float,
    fusion_base_model_name_or_path: str | None = None,
    fusion_base_index_dir: Path | None = None,
    fusion_base_weight: float = 0.7,
) -> dict[str, Any]:
    if retrieval_mode == "vector":
        return vector_search_index(
            query=query,
            model_name_or_path=model_name_or_path,
            index_dir=index_dir,
            top_k=top_k,
        )
    if retrieval_mode == "vector_fusion":
        if not fusion_base_model_name_or_path or fusion_base_index_dir is None:
            raise ValueError("vector_fusion requires fusion_base_model_name_or_path and fusion_base_index_dir")
        return vector_fusion_search_index(
            query=query,
            model_name_or_path=model_name_or_path,
            index_dir=index_dir,
            top_k=top_k,
            base_model_name_or_path=fusion_base_model_name_or_path,
            base_index_dir=fusion_base_index_dir,
            base_weight=fusion_base_weight,
        )
    if retrieval_mode == "bm25":
        return bm25_search_index(
            query=query,
            index_dir=index_dir,
            top_k=top_k,
        )
    return hybrid_search_index(
        query=query,
        model_name_or_path=model_name_or_path,
        index_dir=index_dir,
        top_k=top_k,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
    )


def rerank_hits(*, query: str, hits: list[dict[str, Any]], reranker_model: str, top_k: int) -> list[dict[str, Any]]:
    if not hits:
        return []

    reranker = get_cross_encoder(reranker_model)
    pairs = [(query, hit["text"]) for hit in hits]
    scores = reranker.predict(pairs, show_progress_bar=False)

    rescored: list[dict[str, Any]] = []
    for hit, rerank_score in zip(hits, scores, strict=False):
        updated = dict(hit)
        updated["rerank_score"] = round(float(rerank_score), 4)
        updated["score"] = updated["rerank_score"]
        rescored.append(updated)

    rescored.sort(key=lambda row: row["score"], reverse=True)
    final_hits = rescored[:top_k]
    for rank, hit in enumerate(final_hits, start=1):
        hit["rank"] = rank
    return final_hits


def rerank_hits_with_biencoder_fusion(
    *,
    query: str,
    hits: list[dict[str, Any]],
    reranker_model: str,
    top_k: int,
    fusion_weight: float = 0.15,
    rerank_window: int = 10,
    batch_size: int = 64,
) -> list[dict[str, Any]]:
    if not hits:
        return []

    window = len(hits) if rerank_window <= 0 else min(rerank_window, len(hits))
    window_hits = hits[:window]
    model = get_sentence_transformer(reranker_model)
    retrieval_query = build_retrieval_query(query)
    query_embedding = model.encode(
        [retrieval_query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]
    candidate_embeddings = model.encode(
        [_hit_text_for_biencoder(hit) for hit in window_hits],
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    tuned_scores = np.asarray(candidate_embeddings).dot(query_embedding)
    base_scores = np.asarray([float(hit.get("score", 0.0)) for hit in window_hits], dtype=float)
    weight = min(max(float(fusion_weight), 0.0), 1.0)
    fused_scores = (1.0 - weight) * _normalize_array(base_scores) + weight * _normalize_array(tuned_scores)

    scored_window: list[dict[str, Any]] = []
    for hit, tuned_score, fused_score in zip(window_hits, tuned_scores, fused_scores, strict=False):
        updated = dict(hit)
        updated["base_rerank_score"] = hit.get("score")
        updated["biencoder_score"] = round(float(tuned_score), 6)
        updated["rerank_score"] = round(float(fused_score), 6)
        updated["score"] = updated["rerank_score"]
        scored_window.append(updated)

    scored_window.sort(key=lambda row: float(row["score"]), reverse=True)
    merged_hits = _merge_reranked_window(scored_window, hits)
    final_hits = merged_hits[:top_k]
    for rank, hit in enumerate(final_hits, start=1):
        hit["rank"] = rank
    return final_hits

