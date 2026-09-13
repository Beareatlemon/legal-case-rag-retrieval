from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.rag import build_rag_index, load_corpus_chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a LeCaRDv2 RAG index with progress logging for LEAD or SentenceTransformer models.")
    parser.add_argument(
        "--model",
        default=str(ROOT / "models" / "lead_official" / "dpr_biencoder.70"),
        help="Path to the LEAD DPR checkpoint.",
    )
    parser.add_argument(
        "--corpus",
        default=str(ROOT / "data" / "lecardv2" / "corpus.jsonl"),
        help="Path to the LeCaRDv2 corpus jsonl.",
    )
    parser.add_argument(
        "--index-dir",
        default=str(ROOT / "data" / "indexes" / "lecardv2"),
        help="Output directory for the FAISS index.",
    )
    parser.add_argument("--index-type", default="hnsw", choices=["flat", "ivf", "hnsw"])
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--nlist", type=int, default=256)
    parser.add_argument("--nprobe", type=int, default=16)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--hnsw-ef-construction", type=int, default=80)
    parser.add_argument("--hnsw-ef-search", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus_path = Path(args.corpus)
    index_dir = Path(args.index_dir)

    print(f"[build-lead-index] model={args.model}")
    print(f"[build-lead-index] corpus={corpus_path}")
    print(f"[build-lead-index] index_dir={index_dir}")
    print(
        "[build-lead-index] params="
        f"index_type={args.index_type}, chunk_size={args.chunk_size}, "
        f"chunk_overlap={args.chunk_overlap}, batch_size={args.batch_size}"
    )

    doc_count, chunks = load_corpus_chunks(
        corpus_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    total_batches = (len(chunks) + args.batch_size - 1) // args.batch_size
    print(
        "[build-lead-index] preflight="
        f"documents={doc_count}, chunks={len(chunks)}, total_batches={total_batches}"
    )
    print(
        "[build-lead-index] eta_hint="
        "The encoder will emit periodic progress logs with elapsed time, throughput, and ETA."
    )

    started = time.perf_counter()
    result = build_rag_index(
        model_name_or_path=args.model,
        corpus_path=corpus_path,
        index_dir=index_dir,
        index_type=args.index_type,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
        nlist=args.nlist,
        nprobe=args.nprobe,
        hnsw_m=args.hnsw_m,
        hnsw_ef_construction=args.hnsw_ef_construction,
        hnsw_ef_search=args.hnsw_ef_search,
    )
    elapsed = time.perf_counter() - started
    print(f"[build-lead-index] completed elapsed_minutes={elapsed/60:.2f}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

