from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .llm import SIMILARITY_EXPLANATION_PROMPT, answer_with_context, explain_similarity_with_fallback
from .rag import (
    aggregate_hits_by_doc,
    build_rag_index,
    build_retrieval_query,
    load_manifest,
    rerank_hits,
    rerank_hits_with_biencoder_fusion,
    same_retrieval_model_family,
    search_index,
)
from .rag_benchmark import (
    evaluate_generation_outputs,
    evaluate_reranker_rankings,
    evaluate_retrieval_rankings,
    load_queries_and_qrels,
)
from .schemas import (
    BiencoderEvalRequest,
    BiencoderEvalResponse,
    BiencoderTrainRequest,
    BiencoderTrainResponse,
    ProjectAssetStatusResponse,
    RagBenchmarkLayerResult,
    RagBenchmarkRequest,
    RagBenchmarkResponse,
    RagIndexBuildRequest,
    RagIndexBuildResponse,
    RagQueryRequest,
    RagQueryResponse,
    RagSourceChunk,
    SimilarityExplainRequest,
    SimilarityExplainResponse,
)
from .training import PRESETS, evaluate_biencoder, find_latest_checkpoint, train_biencoder


app = FastAPI(title="LeCaRD Legal RAG Workbench API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _resolve_rag_model_path(override: str | None) -> str:
    if override:
        return override
    trained_config = settings.trained_model_dir / "config.json"
    if trained_config.exists():
        return str(settings.trained_model_dir)
    return settings.rag_default_model


def _looks_like_local_path(value: str) -> bool:
    return ("\\" in value) or ("/" in value) or value.startswith(".")


def _can_use_dense_retrieval(model_name_or_path: str) -> bool:
    if not model_name_or_path:
        return False
    if _looks_like_local_path(model_name_or_path):
        return Path(model_name_or_path).exists()
    return True


def _resolve_fusion_settings() -> tuple[str, Path, float]:
    return (
        settings.rag_fusion_base_model,
        settings.rag_fusion_base_index_dir,
        settings.rag_fusion_base_weight,
    )


def _resolve_reranker_settings(
    reranker_type: str,
    reranker_model: str | None,
    fusion_weight: float | None,
    rerank_window: int | None,
) -> tuple[str, str, float, int]:
    if reranker_type == "bi_encoder_fusion":
        return (
            "bi_encoder_fusion",
            reranker_model or settings.rag_biencoder_reranker_model,
            settings.rag_biencoder_reranker_fusion_weight if fusion_weight is None else fusion_weight,
            settings.rag_biencoder_reranker_window if rerank_window is None else rerank_window,
        )
    return (
        "cross_encoder",
        reranker_model or settings.rag_reranker_model,
        settings.rag_biencoder_reranker_fusion_weight if fusion_weight is None else fusion_weight,
        settings.rag_biencoder_reranker_window if rerank_window is None else rerank_window,
    )


def _format_retrieval_model_label(retrieval_mode: str, model_name_or_path: str, fusion_base_model: str) -> str:
    if retrieval_mode == "vector_fusion":
        return f"{model_name_or_path} + {fusion_base_model}"
    if retrieval_mode == "bm25":
        return "bm25-only"
    return model_name_or_path


@lru_cache(maxsize=4)
def _load_case_lookup(corpus_path_str: str) -> dict[str, dict[str, str]]:
    corpus_path = Path(corpus_path_str)
    lookup: dict[str, dict[str, str]] = {}
    if not corpus_path.exists():
        return lookup
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = str(row.get("_id") or row.get("doc_id") or row.get("id") or "")
            if not doc_id:
                continue
            title = str(row.get("title") or "").strip()
            text = str(row.get("text") or row.get("contents") or "").strip()
            full_text = f"{title}\n{text}" if title else text
            lookup[doc_id] = {"title": title, "text": full_text}
    return lookup


def _resolve_source_corpus_path(index_dir: Path) -> Path:
    try:
        manifest = load_manifest(index_dir)
    except FileNotFoundError:
        return settings.lecardv2_dir / "corpus.jsonl"
    corpus_path = str(manifest.get("corpus_path") or "").strip()
    return Path(corpus_path) if corpus_path else settings.lecardv2_dir / "corpus.jsonl"


def _build_source_case(hit: dict[str, Any], case_lookup: dict[str, dict[str, str]], *, include_contexts: bool) -> RagSourceChunk:
    doc_id = str(hit["doc_id"])
    case = case_lookup.get(doc_id, {})
    full_case_text = case.get("text") or str(hit.get("text", ""))
    title = case.get("title") or str(hit.get("title", ""))
    supporting_chunks = []
    for chunk in hit.get("supporting_chunks", []):
        item = dict(chunk)
        if not include_contexts:
            item["text"] = ""
        supporting_chunks.append(item)

    return RagSourceChunk(
        chunk_id=hit["chunk_id"],
        doc_id=doc_id,
        rank=hit["rank"],
        score=hit["score"],
        retrieval_score=hit.get("retrieval_score"),
        rerank_score=hit.get("rerank_score"),
        vector_score=hit.get("vector_score"),
        bm25_score=hit.get("bm25_score"),
        title=title,
        section=hit.get("section"),
        text=full_case_text if include_contexts else "",
        citation=hit["citation"],
        result_type=hit.get("result_type", "case"),
        support_count=hit.get("support_count"),
        supporting_chunks=supporting_chunks,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "project": "lecard-rag-workbench"}


@app.get("/api/status/assets", response_model=ProjectAssetStatusResponse)
def get_asset_status() -> ProjectAssetStatusResponse:
    latest_checkpoint = find_latest_checkpoint(settings.checkpoints_dir)
    rag_index_ready = (settings.rag_index_dir / "manifest.json").exists()
    settings.runtime_logs_dir.mkdir(parents=True, exist_ok=True)
    return ProjectAssetStatusResponse(
        lecardv2_ready=(
            (settings.lecardv2_dir / "queries.jsonl").exists()
            and (settings.lecardv2_dir / "corpus.jsonl").exists()
            and (settings.lecardv2_dir / "qrels" / "test.jsonl").exists()
        ),
        train_pairs_ready=settings.lecardv2_train_pairs_path.exists(),
        trained_model_ready=(settings.trained_model_dir / "config.json").exists(),
        rag_index_ready=rag_index_ready,
        latest_checkpoint=str(latest_checkpoint) if latest_checkpoint else None,
        default_train_pairs=str(settings.lecardv2_train_pairs_path),
        default_dev_pairs=str(settings.lecardv2_dev_pairs_path),
        default_lecardv2_train_pairs=str(settings.lecardv2_train_pairs_path),
        default_lecardv2_dev_pairs=str(settings.lecardv2_dev_pairs_path),
        default_lecardv2_chunk_train_triplets=str(settings.lecardv2_chunk_train_triplets_path),
        default_lecardv2_chunk_dev_triplets=str(settings.lecardv2_chunk_dev_triplets_path),
        default_lecardv2_chunk_train_labeled_pairs=str(settings.lecardv2_chunk_train_labeled_pairs_path),
        default_lecardv2_chunk_dev_labeled_pairs=str(settings.lecardv2_chunk_dev_labeled_pairs_path),
        default_model_dir=str(settings.trained_model_dir),
        default_bgem3_model_dir=str(settings.bgem3_local_model_dir),
        default_bgem3_experiment_model_dir=str(settings.bgem3_experiment_model_dir),
        default_bgem3_triplet_experiment_model_dir=str(settings.bgem3_triplet_experiment_model_dir),
        default_lecardv2_dir=str(settings.lecardv2_dir),
        default_rag_index_dir=str(settings.rag_index_dir),
        runtime_logs_dir=str(settings.runtime_logs_dir),
        checkpoints_dir=str(settings.checkpoints_dir),
        default_bgem3_experiment_checkpoint_dir=str(settings.bgem3_experiment_checkpoint_dir),
        default_bgem3_triplet_experiment_checkpoint_dir=str(settings.bgem3_triplet_experiment_checkpoint_dir),
    )


@app.post("/api/train/biencoder", response_model=BiencoderTrainResponse)
def train_project_biencoder(payload: BiencoderTrainRequest) -> BiencoderTrainResponse:
    request_data = payload.model_dump()
    preset_name = request_data.pop("preset")
    if preset_name:
        for key, value in PRESETS[preset_name].items():
            request_data[key] = value

    train_pairs_value = request_data.pop("train_pairs")
    dev_pairs_value = request_data.pop("dev_pairs")
    output_dir_value = request_data.pop("output_dir")
    checkpoint_dir_value = request_data.pop("checkpoint_dir")
    train_pairs = Path(train_pairs_value) if train_pairs_value else settings.lecardv2_train_pairs_path
    dev_pairs = Path(dev_pairs_value) if dev_pairs_value else None
    output_dir = Path(output_dir_value) if output_dir_value else settings.trained_model_dir
    checkpoint_dir = Path(checkpoint_dir_value) if checkpoint_dir_value else settings.checkpoints_dir

    try:
        result = train_biencoder(
            train_pairs=train_pairs,
            dev_pairs=dev_pairs,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            **request_data,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Training failed: {exc}") from exc
    return BiencoderTrainResponse(**result)


@app.post("/api/eval/lecardv2-model", response_model=BiencoderEvalResponse)
def eval_trained_biencoder(payload: BiencoderEvalRequest) -> BiencoderEvalResponse:
    try:
        result = evaluate_biencoder(
            model_path=Path(payload.model_path) if payload.model_path else settings.trained_model_dir,
            data_dir=Path(payload.data_dir) if payload.data_dir else settings.lecardv2_dir,
            batch_size=payload.batch_size,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Bi-encoder evaluation failed: {exc}") from exc
    return BiencoderEvalResponse(**result)


@app.post("/api/explain/similarity", response_model=SimilarityExplainResponse)
def explain_case_similarity(payload: SimilarityExplainRequest) -> SimilarityExplainResponse:
    prompt = SIMILARITY_EXPLANATION_PROMPT.format(query=payload.query_text, candidate=payload.candidate_text)
    if payload.show_prompt and not settings.llm_api_key:
        return SimilarityExplainResponse(model=payload.model or settings.llm_model, explanation={}, prompt=prompt)
    if not settings.llm_api_key:
        raise HTTPException(status_code=503, detail="LLM_API_KEY is not configured.")
    try:
        model_used, explanation = explain_similarity_with_fallback(
            query=payload.query_text,
            candidate=payload.candidate_text,
            api_url=payload.api_url or settings.llm_api_url,
            api_key=settings.llm_api_key,
            primary_model=payload.model or settings.llm_model,
            fallback_model=payload.fallback_model or settings.llm_fallback_model,
            primary_timeout=settings.llm_primary_timeout_seconds,
            fallback_timeout=settings.llm_fallback_timeout_seconds,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM explanation failed: {exc}") from exc
    return SimilarityExplainResponse(
        model=model_used,
        explanation=explanation,
        prompt=prompt if payload.show_prompt else None,
    )


@app.post("/api/rag/index", response_model=RagIndexBuildResponse)
def build_local_rag_index(payload: RagIndexBuildRequest) -> RagIndexBuildResponse:
    model_name_or_path = _resolve_rag_model_path(payload.model_path)
    corpus_path = Path(payload.corpus_path) if payload.corpus_path else settings.lecardv2_dir / "corpus.jsonl"
    index_dir = Path(payload.index_dir) if payload.index_dir else settings.rag_index_dir

    try:
        result = build_rag_index(
            model_name_or_path=model_name_or_path,
            corpus_path=corpus_path,
            index_dir=index_dir,
            index_type=payload.index_type,
            chunk_size=payload.chunk_size,
            chunk_overlap=payload.chunk_overlap,
            batch_size=payload.batch_size,
            nlist=payload.nlist,
            nprobe=payload.nprobe,
            hnsw_m=payload.hnsw_m,
            hnsw_ef_construction=payload.hnsw_ef_construction,
            hnsw_ef_search=payload.hnsw_ef_search,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"RAG index build failed: {exc}") from exc
    return RagIndexBuildResponse(**result)


@app.post("/api/rag/query", response_model=RagQueryResponse)
def run_rag_query(payload: RagQueryRequest) -> RagQueryResponse:
    if not settings.llm_api_key:
        raise HTTPException(status_code=503, detail="LLM_API_KEY is not configured.")

    model_name_or_path = _resolve_rag_model_path(payload.model_path)
    index_dir = Path(payload.index_dir) if payload.index_dir else settings.rag_index_dir
    retrieve_k = payload.retrieve_k or max(payload.top_k, settings.rag_default_retrieve_k)
    retrieval_mode = payload.retrieval_mode or settings.rag_default_retrieval_mode
    vector_weight = payload.vector_weight if payload.vector_weight is not None else settings.rag_default_vector_weight
    bm25_weight = payload.bm25_weight if payload.bm25_weight is not None else settings.rag_default_bm25_weight
    if retrieval_mode in {"hybrid", "vector", "vector_fusion"}:
        manifest = load_manifest(index_dir)
        index_model_path = str(manifest.get("model_path", ""))
        if not same_retrieval_model_family(index_model_path, model_name_or_path):
            raise HTTPException(
                status_code=409,
                detail=(
                    "RAG index model does not match the requested retrieval model. "
                    f"Index was built with: {index_model_path}. Requested: {model_name_or_path}. "
                    "Please rebuild the index with /api/rag/index before querying."
                ),
            )
    if retrieval_mode in {"hybrid", "vector", "vector_fusion"} and not _can_use_dense_retrieval(model_name_or_path):
        if retrieval_mode == "vector":
            raise HTTPException(
                status_code=503,
                detail=f"Dense retrieval model is unavailable: {model_name_or_path}. Switch to bm25 or provide a valid model_path.",
            )
        retrieval_mode = "bm25"
    fusion_base_model, fusion_base_index_dir, fusion_base_weight = _resolve_fusion_settings()
    if retrieval_mode == "vector_fusion":
        base_manifest = load_manifest(fusion_base_index_dir)
        base_index_model_path = str(base_manifest.get("model_path", ""))
        if not same_retrieval_model_family(base_index_model_path, fusion_base_model):
            raise HTTPException(
                status_code=409,
                detail=(
                    "Fusion base index model does not match the configured fusion base model. "
                    f"Index was built with: {base_index_model_path}. Base model: {fusion_base_model}. "
                    "Please rebuild the base index before querying with vector_fusion."
                ),
            )
        if not _can_use_dense_retrieval(fusion_base_model):
            raise HTTPException(
                status_code=503,
                detail=f"Fusion base dense retrieval model is unavailable: {fusion_base_model}.",
            )
    reranker_type, reranker_model, reranker_fusion_weight, reranker_window = _resolve_reranker_settings(
        payload.reranker_type,
        payload.reranker_model,
        payload.reranker_fusion_weight,
        payload.reranker_window,
    )
    reranked = bool(payload.use_reranker and reranker_model)
    retrieval_query = build_retrieval_query(payload.query)

    try:
        try:
            retrieval = search_index(
                query=payload.query,
                model_name_or_path=model_name_or_path,
                index_dir=index_dir,
                top_k=retrieve_k,
                retrieval_mode=retrieval_mode,
                vector_weight=vector_weight,
                bm25_weight=bm25_weight,
                fusion_base_model_name_or_path=fusion_base_model,
                fusion_base_index_dir=fusion_base_index_dir,
                fusion_base_weight=fusion_base_weight,
            )
        except Exception:
            if retrieval_mode != "hybrid":
                raise
            retrieval_mode = "bm25"
            retrieval = search_index(
                query=payload.query,
                model_name_or_path="",
                index_dir=index_dir,
                top_k=retrieve_k,
                retrieval_mode=retrieval_mode,
                vector_weight=0.0,
                bm25_weight=bm25_weight,
                fusion_base_model_name_or_path=None,
                fusion_base_index_dir=None,
                fusion_base_weight=0.0,
            )
        hits = retrieval["hits"]
        if reranked:
            if reranker_type == "bi_encoder_fusion":
                hits = rerank_hits_with_biencoder_fusion(
                    query=payload.query,
                    hits=hits,
                    reranker_model=reranker_model,
                    top_k=payload.top_k,
                    fusion_weight=reranker_fusion_weight,
                    rerank_window=reranker_window,
                )
            else:
                hits = rerank_hits(
                    query=retrieval_query,
                    hits=hits,
                    reranker_model=reranker_model,
                    top_k=payload.top_k,
                )
            hits = aggregate_hits_by_doc(hits, top_k=payload.top_k)
        else:
            hits = aggregate_hits_by_doc(hits, top_k=payload.top_k)

        model_used, answer = answer_with_context(
            query=payload.query,
            hits=hits,
            api_url=settings.llm_api_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            fallback_model=settings.llm_fallback_model,
            primary_timeout=settings.llm_primary_timeout_seconds,
            fallback_timeout=settings.llm_fallback_timeout_seconds,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"RAG query failed: {exc}") from exc

    source_corpus_path = _resolve_source_corpus_path(index_dir)
    case_lookup = _load_case_lookup(str(source_corpus_path))
    source_chunks = [_build_source_case(hit, case_lookup, include_contexts=payload.include_contexts) for hit in hits]
    return RagQueryResponse(
        answer=answer,
        model=model_used,
        retrieval_model=_format_retrieval_model_label(retrieval_mode, model_name_or_path, fusion_base_model),
        retrieval_mode=retrieval_mode,
        reranker_model=f"{reranker_type}:{reranker_model}" if reranked else None,
        index_dir=str(index_dir),
        backend=retrieval["backend"],
        contexts_used=len(source_chunks),
        retrieved_candidates=len(retrieval["hits"]),
        reranked=reranked,
        source_chunks=source_chunks,
    )


@app.post("/api/eval/rag", response_model=RagBenchmarkResponse)
def eval_rag_pipeline(payload: RagBenchmarkRequest) -> RagBenchmarkResponse:
    data_dir = Path(payload.data_dir) if payload.data_dir else settings.lecardv2_dir
    index_dir = Path(payload.index_dir) if payload.index_dir else settings.rag_index_dir
    model_name_or_path = _resolve_rag_model_path(payload.model_path)
    reranker_type, reranker_model, reranker_fusion_weight, reranker_window = _resolve_reranker_settings(
        payload.reranker_type,
        payload.reranker_model,
        payload.reranker_fusion_weight,
        payload.reranker_window,
    )
    notes: list[str] = []

    queries = load_queries_and_qrels(data_dir, max_queries=payload.max_queries)
    if not queries:
        raise HTTPException(status_code=404, detail=f"No benchmark queries with qrels found under: {data_dir}")

    if payload.retrieval_mode in {"hybrid", "vector", "vector_fusion"}:
        manifest = load_manifest(index_dir)
        index_model_path = str(manifest.get("model_path", ""))
        if not same_retrieval_model_family(index_model_path, model_name_or_path):
            raise HTTPException(
                status_code=409,
                detail=(
                    "RAG index model does not match the requested retrieval model. "
                    f"Index was built with: {index_model_path}. Requested: {model_name_or_path}. "
                    "Please rebuild the index with /api/rag/index before benchmarking."
                ),
            )
    fusion_base_model, fusion_base_index_dir, fusion_base_weight = _resolve_fusion_settings()
    if payload.retrieval_mode == "vector_fusion":
        base_manifest = load_manifest(fusion_base_index_dir)
        base_index_model_path = str(base_manifest.get("model_path", ""))
        if not same_retrieval_model_family(base_index_model_path, fusion_base_model):
            raise HTTPException(
                status_code=409,
                detail=(
                    "Fusion base index model does not match the configured fusion base model. "
                    f"Index was built with: {base_index_model_path}. Base model: {fusion_base_model}. "
                    "Please rebuild the base index before benchmarking with vector_fusion."
                ),
            )

    try:
        retrieval_metrics, retrieval_rows, retrieval_latency_ms = evaluate_retrieval_rankings(
            queries,
            model_name_or_path=model_name_or_path,
            index_dir=index_dir,
            retrieval_mode=payload.retrieval_mode,
            vector_weight=payload.vector_weight,
            bm25_weight=payload.bm25_weight,
            retrieve_k=payload.retrieve_k,
            top_k=payload.top_k,
            fusion_base_model_name_or_path=fusion_base_model,
            fusion_base_index_dir=fusion_base_index_dir,
            fusion_base_weight=fusion_base_weight,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"RAG retrieval benchmark failed: {exc}") from exc

    rerank_layer: RagBenchmarkLayerResult | None = None
    generation_source_rows = retrieval_rows
    if payload.use_reranker and reranker_model:
        try:
            rerank_metrics, generation_source_rows, rerank_latency_ms = evaluate_reranker_rankings(
                retrieval_rows,
                reranker_model=reranker_model,
                top_k=payload.top_k,
                reranker_type=reranker_type,
                fusion_weight=reranker_fusion_weight,
                rerank_window=reranker_window,
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"RAG rerank benchmark failed: {exc}") from exc
        rerank_layer = RagBenchmarkLayerResult(
            name="rerank",
            query_count=len(generation_source_rows),
            avg_latency_ms=round(rerank_latency_ms, 2),
            metrics=rerank_metrics,
            note=f"{reranker_type} reranker quality after dense/sparse retrieval.",
        )
    elif payload.use_reranker:
        notes.append("Reranker evaluation was requested but no reranker model was configured.")

    generation_layer: RagBenchmarkLayerResult | None = None
    if payload.evaluate_generation:
        if not settings.llm_api_key:
            notes.append("Generation evaluation was skipped because LLM_API_KEY is not configured.")
        else:
            sample_size = min(payload.generation_sample_size, len(generation_source_rows))
            try:
                generation_metrics, generation_latency_ms = evaluate_generation_outputs(
                    generation_source_rows,
                    api_url=settings.llm_api_url,
                    api_key=settings.llm_api_key,
                    model=settings.llm_model,
                    fallback_model=settings.llm_fallback_model,
                    primary_timeout=settings.llm_primary_timeout_seconds,
                    fallback_timeout=settings.llm_fallback_timeout_seconds,
                    sample_size=sample_size,
                )
            except Exception as exc:
                raise HTTPException(status_code=502, detail=f"RAG generation benchmark failed: {exc}") from exc
            generation_layer = RagBenchmarkLayerResult(
                name="generation",
                query_count=sample_size,
                avg_latency_ms=round(generation_latency_ms, 2),
                metrics=generation_metrics,
                note="Automatic answer-structure and citation-support checks over sampled benchmark queries.",
            )

    return RagBenchmarkResponse(
        data_dir=str(data_dir),
        index_dir=str(index_dir),
        retrieval_model=_format_retrieval_model_label(payload.retrieval_mode, model_name_or_path, fusion_base_model),
        retrieval_mode=payload.retrieval_mode,
        reranker_model=f"{reranker_type}:{reranker_model}" if (payload.use_reranker and reranker_model) else None,
        query_count=len(queries),
        retrieval_layer=RagBenchmarkLayerResult(
            name="retrieval",
            query_count=len(retrieval_rows),
            avg_latency_ms=round(retrieval_latency_ms, 2),
            metrics=retrieval_metrics,
            note="Initial retrieval quality before reranking.",
        ),
        rerank_layer=rerank_layer,
        generation_layer=generation_layer,
        notes=notes,
    )

