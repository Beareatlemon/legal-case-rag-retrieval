from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.config import settings
from backend.app.training import PRESETS, train_biencoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MarginMSE-based LeCaRDv2 fine-tuning for bge-m3.")
    parser.add_argument("--train-pairs", type=Path, default=settings.lecardv2_chunk_train_margin_triplets_path)
    parser.add_argument("--base-model", default=str(settings.bgem3_local_model_dir))
    parser.add_argument("--output-dir", type=Path, default=settings.models_dir / "bgem3_lecardv2_margin_experiment")
    parser.add_argument("--checkpoint-dir", type=Path, default=settings.models_dir / "checkpoints" / "bgem3_lecardv2_margin_experiment")
    parser.add_argument("--preset", default="bgem3_margin_rtx4060", choices=sorted(PRESETS))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--checkpoint-save-steps", type=int, default=None)
    parser.add_argument("--checkpoint-save-total-limit", type=int, default=None)
    parser.add_argument("--monitor-seconds", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    preset = PRESETS[args.preset]
    result = train_biencoder(
        train_pairs=args.train_pairs,
        base_model=args.base_model,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_save_steps=args.checkpoint_save_steps or int(preset["checkpoint_save_steps"]),
        checkpoint_save_total_limit=args.checkpoint_save_total_limit or int(preset["checkpoint_save_total_limit"]),
        resume=args.resume,
        monitor_seconds=args.monitor_seconds if args.monitor_seconds is not None else int(preset["monitor_seconds"]),
        epochs=args.epochs,
        batch_size=args.batch_size or int(preset["batch_size"]),
        limit=None if args.limit == 0 else args.limit,
        max_seq_length=args.max_seq_length or int(preset["max_seq_length"]),
        use_amp=args.use_amp or bool(preset.get("use_amp", False)),
        learning_rate=args.learning_rate if args.learning_rate is not None else preset.get("learning_rate"),
    )
    print(result)


if __name__ == "__main__":
    main()

