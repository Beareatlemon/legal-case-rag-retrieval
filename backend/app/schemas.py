from __future__ import annotations

from typing import Any, Dict, Literal

from pydantic import BaseModel, Field


class SimilarityExplainRequest(BaseModel):
    query_text: str = Field(min_length=1)
    candidate_text: str = Field(min_length=1)
    show_prompt: bool = False
    api_url: str | None = None
    model: str | None = None
    fallback_model: str | None = None


class SimilarityExplainResponse(BaseModel):
    model: str
    explanation: Dict[str, Any]
    prompt: str | None = None


class BiencoderTrainRequest(BaseModel):
    preset: Literal[
        "rtx4060",
        "rtx4090",
        "bgem3_rtx4060",
        "bgem3_triplet_rtx4060",
        "bgem3_conservative_rtx4060",
        "bgem3_margin_rtx4060",
        "bgem3_labeled_rtx4060",
    ] | None = None
    train_pairs: str | None = None
    dev_pairs: str | None = None
    base_model: str = "BAAI/bge-large-zh-v1.5"
    output_dir: str | None = None
    checkpoint_dir: str | None = None
    checkpoint_save_steps: int = Field(default=100, ge=1)
    checkpoint_save_total_limit: int = Field(default=3, ge=1)
    resume: bool = False
    monitor_seconds: int = Field(default=60, ge=0)
    epochs: int = Field(default=1, ge=1)
    batch_size: int = Field(default=2, ge=1)
    limit: int | None = Field(default=2000, ge=1)
    max_seq_length: int = Field(default=512, ge=1)
    use_amp: bool = False
    evaluation_steps: int | None = Field(default=None, ge=1)


class BiencoderTrainResponse(BaseModel):
    model_source: str
    resumed_from: str | None = None
    train_pairs: str
    dev_pairs: str | None = None
    examples_loaded: int
    dev_examples_loaded: int = 0
    has_negatives: bool = False
    has_labels: bool = False
    has_margin_labels: bool = False
    loss_name: str
    evaluator_name: str | None = None
    evaluation_steps: int = 0
    epochs: int
    batch_size: int
    max_seq_length: int
    steps_per_epoch: int
    warmup_steps: int
    output_dir: str
    checkpoint_dir: str
    checkpoint_save_steps: int
    checkpoint_save_total_limit: int
    use_amp: bool
    device_info: Dict[str, Any]
    gpu_logs: list[Dict[str, str]]


class BiencoderEvalRequest(BaseModel):
    model_path: str | None = None
    data_dir: str | None = None
    batch_size: int = Field(default=8, ge=1)


class BiencoderEvalResponse(BaseModel):
    model_path: str
    data_dir: str
    query_count: int
    corpus_count: int
    qrels_count: int
    metrics: Dict[str, float]


class ProjectAssetStatusResponse(BaseModel):
    lecardv2_ready: bool
    train_pairs_ready: bool
    trained_model_ready: bool
    rag_index_ready: bool
    latest_checkpoint: str | None = None
    default_train_pairs: str
    default_dev_pairs: str
    default_lecardv2_train_pairs: str
    default_lecardv2_dev_pairs: str
    default_lecardv2_chunk_train_triplets: str
    default_lecardv2_chunk_dev_triplets: str
    default_lecardv2_chunk_train_labeled_pairs: str
    default_lecardv2_chunk_dev_labeled_pairs: str
    default_model_dir: str
    default_bgem3_model_dir: str
    default_bgem3_experiment_model_dir: str
    default_bgem3_triplet_experiment_model_dir: str
    default_lecardv2_dir: str
    default_rag_index_dir: str
    runtime_logs_dir: str
    checkpoints_dir: str
    default_bgem3_experiment_checkpoint_dir: str
    default_bgem3_triplet_experiment_checkpoint_dir: str


class RagIndexBuildRequest(BaseModel):
    model_path: str | None = None
    corpus_path: str | None = None
    index_dir: str | None = None
    index_type: Literal["flat", "ivf", "hnsw"] = "hnsw"
    chunk_size: int = Field(default=800, ge=100, le=4000)
    chunk_overlap: int = Field(default=120, ge=0, le=1000)
    batch_size: int = Field(default=8, ge=1, le=256)
    nlist: int = Field(default=256, ge=1, le=65536)
    nprobe: int = Field(default=16, ge=1, le=65536)
    hnsw_m: int = Field(default=32, ge=4, le=128)
    hnsw_ef_construction: int = Field(default=80, ge=8, le=512)
    hnsw_ef_search: int = Field(default=64, ge=8, le=512)


class RagIndexBuildResponse(BaseModel):
    model_path: str
    corpus_path: str
    index_dir: str
    backend: str
    index_type: str
    chunk_size: int
    chunk_overlap: int
    document_count: int
    chunk_count: int
    embedding_dim: int
    nlist: int | None = None
    nprobe: int | None = None
    hnsw_m: int | None = None
    hnsw_ef_construction: int | None = None
    hnsw_ef_search: int | None = None


class RagQueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    retrieve_k: int | None = Field(default=None, ge=1, le=100)
    model_path: str | None = None
    index_dir: str | None = None
    include_contexts: bool = True
    retrieval_mode: Literal["hybrid", "vector", "bm25", "vector_fusion"] = "hybrid"
    vector_weight: float = Field(default=1.0, ge=0.0, le=10.0)
    bm25_weight: float = Field(default=1.0, ge=0.0, le=10.0)
    use_reranker: bool = False
    reranker_type: Literal["cross_encoder", "bi_encoder_fusion"] = "cross_encoder"
    reranker_model: str | None = None
    reranker_fusion_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    reranker_window: int | None = Field(default=None, ge=1, le=100)


class RagSourceChunk(BaseModel):
    chunk_id: str
    doc_id: str
    rank: int
    score: float
    retrieval_score: float | None = None
    rerank_score: float | None = None
    vector_score: float | None = None
    bm25_score: float | None = None
    title: str
    section: str | None = None
    text: str
    citation: str
    result_type: str = "case"
    support_count: int | None = None
    supporting_chunks: list[Dict[str, Any]] = Field(default_factory=list)


class RagQueryResponse(BaseModel):
    answer: str
    model: str
    retrieval_model: str
    retrieval_mode: str
    reranker_model: str | None = None
    index_dir: str
    backend: str
    contexts_used: int
    retrieved_candidates: int
    reranked: bool
    source_chunks: list[RagSourceChunk]


class RagBenchmarkRequest(BaseModel):
    data_dir: str | None = None
    model_path: str | None = None
    index_dir: str | None = None
    retrieval_mode: Literal["hybrid", "vector", "bm25", "vector_fusion"] = "hybrid"
    vector_weight: float = Field(default=1.0, ge=0.0, le=10.0)
    bm25_weight: float = Field(default=1.0, ge=0.0, le=10.0)
    retrieve_k: int = Field(default=20, ge=1, le=100)
    top_k: int = Field(default=5, ge=1, le=20)
    max_queries: int = Field(default=20, ge=1)
    use_reranker: bool = True
    reranker_type: Literal["cross_encoder", "bi_encoder_fusion"] = "cross_encoder"
    reranker_model: str | None = None
    reranker_fusion_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    reranker_window: int | None = Field(default=None, ge=1, le=100)
    evaluate_generation: bool = True
    generation_sample_size: int = Field(default=10, ge=1, le=100)


class RagBenchmarkLayerResult(BaseModel):
    name: str
    query_count: int
    avg_latency_ms: float
    metrics: Dict[str, float]
    note: str = ""


class RagBenchmarkResponse(BaseModel):
    data_dir: str
    index_dir: str
    retrieval_model: str
    retrieval_mode: str
    reranker_model: str | None = None
    query_count: int
    retrieval_layer: RagBenchmarkLayerResult
    rerank_layer: RagBenchmarkLayerResult | None = None
    generation_layer: RagBenchmarkLayerResult | None = None
    notes: list[str]

