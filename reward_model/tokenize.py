from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline tokenization for OrchRM preference datasets.")
    p.add_argument("--train-path",      required=True,  help="Path to train JSONL file.")
    p.add_argument("--val-path",        required=True,  help="Path to validation JSONL file.")
    p.add_argument("--tokenizer-path",  required=True,  help="HF tokenizer path or hub name (e.g. Skywork/Skywork-Reward-Llama-3.1-8B).")
    p.add_argument("--out-dir",         required=True,  help="Output directory for the tokenized DatasetDict.")
    p.add_argument("--max-length",      type=int, default=7168, help="Max token length (default: 7168).")
    args = p.parse_args()
    # Normalise hyphens to underscores for attribute access
    args.train_path     = args.train_path
    args.val_path       = args.val_path
    args.tokenizer_path = args.tokenizer_path
    args.out_dir        = args.out_dir
    args.max_length     = args.max_length
    return args


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        trust_remote_code=True,
        use_fast=False,
    )

    dataset = load_dataset(
        "json",
        data_files={
            "train": args.train_path,
            "validation": args.val_path,
        },
    )

    def tokenize_pair(ex: dict) -> dict:
        chosen_text = ex.get("chosen_text")
        rejected_text = ex.get("rejected_text")

        if not isinstance(chosen_text, str) or not chosen_text.strip():
            raise ValueError("Missing or empty chosen_text in dataset row.")
        if not isinstance(rejected_text, str) or not rejected_text.strip():
            raise ValueError("Missing or empty rejected_text in dataset row.")

        chosen_full_ids = tokenizer(chosen_text, truncation=False)["input_ids"]
        rejected_full_ids = tokenizer(rejected_text, truncation=False)["input_ids"]

        chosen_was_truncated = len(chosen_full_ids) > args.max_length
        rejected_was_truncated = len(rejected_full_ids) > args.max_length

        return {
            "chosen_input_ids": chosen_full_ids[: args.max_length],
            "rejected_input_ids": rejected_full_ids[: args.max_length],
            "chosen_was_truncated": chosen_was_truncated,
            "rejected_was_truncated": rejected_was_truncated,
            "any_was_truncated": chosen_was_truncated or rejected_was_truncated,
            "chosen_full_len": len(chosen_full_ids),
            "rejected_full_len": len(rejected_full_ids),
        }

    tokenized = dataset.map(tokenize_pair, remove_columns=dataset["train"].column_names)

    for split in ["train", "validation"]:
        total = len(tokenized[split])
        chosen_truncated = sum(tokenized[split]["chosen_was_truncated"])
        rejected_truncated = sum(tokenized[split]["rejected_was_truncated"])
        any_truncated = sum(tokenized[split]["any_was_truncated"])
        print(
            f"{split}: total={total} chosen_truncated={chosen_truncated} "
            f"rejected_truncated={rejected_truncated} any_truncated={any_truncated}"
        )

    out_dir = Path(args.out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    tokenized.save_to_disk(str(out_dir))
    print(f"Saved tokenized dataset to: {out_dir}")


if __name__ == "__main__":
    main()
