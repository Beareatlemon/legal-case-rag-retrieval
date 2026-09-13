from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    data_dir: Path = ROOT_DIR / "data"
    models_dir: Path = ROOT_DIR / "models"
    runtime_logs_dir: Path = ROOT_DIR / "runtime_logs"
    indexes_dir: Path = ROOT_DIR / "data" / "indexes"
    lecardv2_dir: Path = ROOT_DIR / "data" / "lecardv2"
    train_data_dir: Path = ROOT_DIR / "data" / "train"
    checkpoints_dir: Path = ROOT_DIR / "models" / "checkpoints" / "biencoder"
    trained_model_dir: Path = ROOT_DIR / "models" / "biencoder"
    bgem3_local_model_dir: Path = Path(
        os.getenv(
            "BGE_M3_MODEL",
            str(models_dir / "bge-m3"),
        )
    )
    lecardv2_train_pairs_path: Path = train_data_dir / "lecardv2_train_pairs.jsonl"
    lecardv2_dev_pairs_path: Path = train_data_dir / "lecardv2_dev_pairs.jsonl"
    lecardv2_chunk_train_triplets_path: Path = train_data_dir / "lecardv2_chunk_train_triplets.jsonl"
    lecardv2_chunk_dev_triplets_path: Path = train_data_dir / "lecardv2_chunk_dev_triplets.jsonl"
    lecardv2_chunk_train_labeled_pairs_path: Path = train_data_dir / "lecardv2_chunk_train_labeled_pairs.jsonl"
    lecardv2_chunk_dev_labeled_pairs_path: Path = train_data_dir / "lecardv2_chunk_dev_labeled_pairs.jsonl"
    lecardv2_chunk_train_margin_triplets_path: Path = train_data_dir / "lecardv2_chunk_train_margin_triplets.jsonl"
    lecardv2_chunk_dev_margin_triplets_path: Path = train_data_dir / "lecardv2_chunk_dev_margin_triplets.jsonl"
    bgem3_experiment_model_dir: Path = models_dir / "bgem3_lecardv2_experiment"
    bgem3_experiment_checkpoint_dir: Path = models_dir / "checkpoints" / "bgem3_lecardv2_experiment"
    bgem3_experiment_index_dir: Path = indexes_dir / "lecardv2_bgem3_experiment"
    bgem3_triplet_experiment_model_dir: Path = models_dir / "bgem3_lecardv2_triplet_experiment"
    bgem3_triplet_experiment_checkpoint_dir: Path = models_dir / "checkpoints" / "bgem3_lecardv2_triplet_experiment"
    rag_index_dir: Path = Path(os.getenv("RAG_INDEX_DIR", str(indexes_dir / "lecardv2")))
    rag_default_model: str = os.getenv("RAG_MODEL", str(bgem3_local_model_dir))
    rag_default_index_type: str = os.getenv("RAG_INDEX_TYPE", "hnsw")
    rag_default_hnsw_m: int = int(os.getenv("RAG_HNSW_M", "32"))
    rag_default_hnsw_ef_construction: int = int(os.getenv("RAG_HNSW_EF_CONSTRUCTION", "80"))
    rag_default_hnsw_ef_search: int = int(os.getenv("RAG_HNSW_EF_SEARCH", "64"))
    rag_default_retrieval_mode: str = os.getenv("RAG_RETRIEVAL_MODE", "hybrid")
    rag_default_vector_weight: float = float(os.getenv("RAG_VECTOR_WEIGHT", "1.0"))
    rag_default_bm25_weight: float = float(os.getenv("RAG_BM25_WEIGHT", "1.0"))
    rag_fusion_base_model: str = os.getenv("RAG_FUSION_BASE_MODEL", str(bgem3_local_model_dir))
    rag_fusion_base_index_dir: Path = Path(os.getenv("RAG_FUSION_BASE_INDEX_DIR", str(indexes_dir / "lecardv2_bgem3")))
    rag_fusion_base_weight: float = float(os.getenv("RAG_FUSION_BASE_WEIGHT", "0.7"))
    rag_reranker_model: str = os.getenv("RAG_RERANKER_MODEL", "")
    rag_biencoder_reranker_model: str = os.getenv(
        "RAG_BIENCODER_RERANKER_MODEL",
        str(models_dir / "experiments" / "bgem3_labeled_hn" / "bgem3_qrels_boost_margin_q120_cuda"),
    )
    rag_biencoder_reranker_fusion_weight: float = float(os.getenv("RAG_BIENCODER_RERANKER_FUSION_WEIGHT", "0.15"))
    rag_biencoder_reranker_window: int = int(os.getenv("RAG_BIENCODER_RERANKER_WINDOW", "10"))
    rag_default_retrieve_k: int = int(os.getenv("RAG_RETRIEVE_K", "20"))
    llm_api_url: str = os.getenv("LLM_API_URL", "https://integrate.api.nvidia.com/v1/chat/completions")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "z-ai/glm4.7")
    llm_fallback_model: str = os.getenv("LLM_FALLBACK_MODEL", "meta/llama-3.1-8b-instruct")
    llm_primary_timeout_seconds: int = int(os.getenv("LLM_PRIMARY_TIMEOUT_SECONDS", "30"))
    llm_fallback_timeout_seconds: int = int(os.getenv("LLM_FALLBACK_TIMEOUT_SECONDS", "120"))


settings = Settings()

