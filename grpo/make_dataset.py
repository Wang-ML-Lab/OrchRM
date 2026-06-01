#!/usr/bin/env python3
"""Build unified GRPO parquet datasets for verl.

Output schema per row:
- data_source: str
- ability: str
- prompt: list[{"role": str, "content": str}]            # policy prompt
- reward_model: {"ground_truth": str, "style": "model"}
- extra_info: {"rm_prompt": str, ...}
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

# Repo root — used to locate the bundled prompts/ package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# make_prompt.py lives at utils/prompts/make_prompt.py inside the repo.
_DEFAULT_MAKE_PROMPT_PATH = str(_REPO_ROOT / "utils" / "prompts" / "make_prompt.py")

# Example dataset configs — pass your own via --dataset-config (JSON file).
_EXAMPLE_TRAIN_DATASETS = {
    "math": {
        "dataset_name": "deepscaler",
        "source": "FlyLee/deepscaler-randomsampled-10k",
        "source_format": "hf",
        "split": "train",
        "size": 5000,
    },
    "browsecomp": {
        "dataset_name": "browsecomp",
        "source": "your-hf-username/browsecomp-train",  # replace with your HF dataset
        "source_format": "hf",
        "split": "train",
        "size": 0,
    },
    "hotpotqa": {
        "dataset_name": "hotpotqa-distractor",
        "source": "hotpotqa/hotpot_qa",
        "source_format": "hf",
        "hf_config": "distractor",
        "split": "train",
        "size": 5000,
    },
}

_EXAMPLE_TEST_DATASETS = {
    "math": {
        "dataset_name": "deepscaler",
        "source": "FlyLee/deepscaler-randomsampled-10k",
        "source_format": "hf",
        "split": "test",
        "size": 200,
    },
    "browsecomp": {
        "dataset_name": "browsecomp-plus",
        "source": "your-hf-username/browsecomp-test",  # replace with your HF dataset
        "source_format": "hf",
        "split": "test",
        "size": 200,
    },
    "hotpotqa": {
        "dataset_name": "hotpotqa-distractor",
        "source": "hotpotqa/hotpot_qa",
        "source_format": "hf",
        "hf_config": "distractor",
        "split": "validation",
        "size": 200,
    },
}


@dataclass
class UnifiedRow:
    data_source: str
    ability: str
    prompt: list[dict[str, str]]
    reward_model: dict[str, str]
    extra_info: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_source": self.data_source,
            "ability": self.ability,
            "prompt": self.prompt,
            "reward_model": self.reward_model,
            "extra_info": self.extra_info,
        }


def _pick_first(row: dict[str, Any], keys: list[str]) -> Any:
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return None


def _to_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)

def _load_make_prompt_module(make_prompt_path: str):
    path = Path(make_prompt_path)
    spec = importlib.util.spec_from_file_location("grpo_make_prompt", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class PromptBuilder:
    """Build prompt messages via prompts/make_prompt.py."""

    def __init__(
        self,
        make_prompt_path: str,
        qwen_tokenizer_path: str,
        problem_type: str,
        node_model: str,
    ):
        self._module = _load_make_prompt_module(make_prompt_path)
        self._problem_type = problem_type
        self._node_model = node_model
        self._qwen_tokenizer = self._load_qwen_tokenizer(qwen_tokenizer_path)

    @staticmethod
    def _load_qwen_tokenizer(path: str):
        try:
            from transformers import AutoTokenizer
        except Exception as e:
            raise RuntimeError(
                "transformers is required to render RM prompts. Install with: pip install transformers"
            ) from e
        return AutoTokenizer.from_pretrained(path, use_fast=False)

    def build_messages(self, question: str, dataset_name: str) -> list[dict[str, str]]:
        system_prompt, user_prompt, developer_prompt = self._module.make_prompt(
            question, self._problem_type, self._node_model, dataset_name
        )
        msgs: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if developer_prompt is not None:
            msgs.append({"role": "user", "content": developer_prompt})
        return msgs

    def build_reward_prompt(self, question: str, dataset_name: str) -> str:
        return self._module.make_mas_prompt(
            question, self._problem_type, self._node_model, dataset_name,
            tokenizer=self._qwen_tokenizer,
        )


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _read_hf_dataset(source: str, split: str, hf_config: str | None = None) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as e:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "HuggingFace 'datasets' is required for source_format='hf'. "
            "Install with: pip install datasets"
        ) from e
    if hf_config:
        ds = load_dataset(source, hf_config, split=split)
    else:
        ds = load_dataset(source, split=split)
    return [ds[i] for i in range(len(ds))]


def _load_rows(spec: dict[str, Any]) -> list[dict[str, Any]]:
    source = spec["source"]
    source_format = spec["source_format"]
    split = spec.get("split", "train")
    hf_config = spec.get("hf_config")

    if source_format == "jsonl":
        return _read_jsonl(source)
    if source_format == "hf":
        return _read_hf_dataset(source, split, hf_config=hf_config)
    raise ValueError(f"Unsupported source_format={source_format}")


def _extract_question_answer(domain: str, row: dict[str, Any]) -> tuple[str, str]:
    if domain == "browsecomp":
        q = _pick_first(row, ["query", "question", "prompt"])
        a = _pick_first(row, ["answer", "gold", "final_answer"])
    elif domain == "hotpotqa":
        q = _pick_first(row, ["question", "query", "prompt"])
        a = _pick_first(row, ["answer", "gold", "final_answer"])
    elif domain == "math":
        q = _pick_first(row, ["question", "problem", "query", "prompt"])
        a = _pick_first(row, ["answer", "ground_truth", "target", "final_answer", "solution"])
    else:
        q = _pick_first(row, ["question", "query", "prompt"])
        a = _pick_first(row, ["answer", "ground_truth", "target", "final_answer"])
    return _to_str(q), _to_str(a)


def _normalize_domain_rows(
    domain: str,
    spec: dict[str, Any],
    seed: int,
    prompt_builder: PromptBuilder,
) -> list[dict[str, Any]]:
    raw_rows = _load_rows(spec)
    if not raw_rows:
        raise ValueError(f"No rows loaded for domain={domain} from source={spec['source']}")

    rng = random.Random(seed)
    rng.shuffle(raw_rows)
    size = spec.get("size", 0)
    sampled = raw_rows[:size] if size > 0 else raw_rows

    dataset_name = spec.get("dataset_name", domain)
    out_rows: list[dict[str, Any]] = []
    for i, row in enumerate(sampled):
        question, answer = _extract_question_answer(domain, row)
        if not question:
            continue
        sample_id = _to_str(_pick_first(row, ["id", "query_id", "qid", "_id", "uuid"])) or str(i)
        policy_prompt = prompt_builder.build_messages(question, domain)
        rm_prompt = prompt_builder.build_reward_prompt(question, domain)
        out_rows.append(UnifiedRow(
            data_source=dataset_name,
            ability=domain,
            prompt=policy_prompt,
            reward_model={"ground_truth": answer, "style": "model"},
            extra_info={"qid": sample_id, "question": question, "rm_prompt": rm_prompt},
        ).to_dict())
    return out_rows


def _split_train_val(rows: list[dict[str, Any]], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    if not rows:
        return [], []
    rng = random.Random(seed)
    rows_copy = list(rows)
    rng.shuffle(rows_copy)
    val_size = max(1, int(len(rows_copy) * val_ratio))
    val_rows = rows_copy[:val_size]
    train_rows = rows_copy[val_size:]
    if not train_rows:
        train_rows = val_rows
        val_rows = []
    return train_rows, val_rows


def _validate_rows(rows: Iterable[dict[str, Any]]) -> None:
    required = ["data_source", "ability", "prompt", "reward_model", "extra_info"]
    for idx, row in enumerate(rows):
        for key in required:
            if key not in row:
                raise ValueError(f"Missing required key '{key}' at row {idx}")
        if not isinstance(row["prompt"], list) or not row["prompt"]:
            raise ValueError(f"Invalid prompt at row {idx}")
        if "ground_truth" not in row["reward_model"]:
            raise ValueError(f"Missing reward_model.ground_truth at row {idx}")
        if row["reward_model"].get("style") != "model":
            raise ValueError(f"reward_model.style must be 'model' at row {idx}")
        if "rm_prompt" not in row["extra_info"]:
            raise ValueError(f"Missing extra_info.rm_prompt at row {idx}")
        if not isinstance(row["extra_info"]["rm_prompt"], str) or not row["extra_info"]["rm_prompt"].strip():
            raise ValueError(f"extra_info.rm_prompt must be a non-empty string at row {idx}")


def _write_parquet(rows: list[dict[str, Any]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, output_path)


def _write_preview_json(rows: list[dict[str, Any]], output_path: str, n: int = 5) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rows[:n], f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build GRPO parquet datasets for verl training.")

    p.add_argument("--output-dir",    required=True,
                   help="Root output directory. Train/val parquets written to <output-dir>/train/.")
    p.add_argument("--qwen-tokenizer-path", required=True,
                   help="HF path or hub name for the Qwen tokenizer used in RM prompt formatting.")
    p.add_argument("--dataset-config", default="",
                   help="Path to a JSON file with dataset spec (same structure as _EXAMPLE_TRAIN_DATASETS). "
                        "If omitted, uses the built-in example config (HF sources).")
    p.add_argument("--test",          action="store_true",
                   help="Use test/eval dataset spec instead of train spec.")
    p.add_argument("--make-prompt-path", default=_DEFAULT_MAKE_PROMPT_PATH,
                   help=f"Path to make_prompt.py (default: repo's prompts/make_prompt.py).")
    p.add_argument("--problem-type",  default="harmony_medium",
                   help="Prompt problem type (default: harmony_medium).")
    p.add_argument("--node-model",    default="gpt-4o",
                   help="Sub-agent model name for prompt formatting (default: gpt-4o).")
    p.add_argument("--val-ratio",     type=float, default=0.1)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--preview",       type=int,   default=5,
                   help="Number of preview rows to write as JSON (default: 5).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.dataset_config:
        with open(args.dataset_config, encoding="utf-8") as f:
            dataset_spec = json.load(f)
    elif args.test:
        dataset_spec = _EXAMPLE_TEST_DATASETS
    else:
        dataset_spec = _EXAMPLE_TRAIN_DATASETS

    random.seed(args.seed)
    prompt_builder = PromptBuilder(
        make_prompt_path=args.make_prompt_path,
        qwen_tokenizer_path=args.qwen_tokenizer_path,
        problem_type=args.problem_type,
        node_model=args.node_model,
    )

    all_rows: list[dict[str, Any]] = []
    for domain, spec in dataset_spec.items():
        domain_rows = _normalize_domain_rows(domain, spec, seed=args.seed, prompt_builder=prompt_builder)
        print(f"[{domain}] loaded={len(domain_rows)}")
        all_rows.extend(domain_rows)

    if not all_rows:
        raise RuntimeError("No rows were produced.")

    _validate_rows(all_rows)
    train_rows, val_rows = _split_train_val(all_rows, args.val_ratio, seed=args.seed)
    _validate_rows(train_rows)
    if val_rows:
        _validate_rows(val_rows)

    mode = "test" if args.test else "train"
    base_dir = os.path.join(args.output_dir, mode)
    train_path = os.path.join(base_dir, "train.parquet")
    val_path = os.path.join(base_dir, "val.parquet")
    preview_path = os.path.join(base_dir, f"preview_{args.preview}.json")

    _write_parquet(train_rows, train_path)
    _write_parquet(val_rows, val_path)
    _write_preview_json(train_rows, preview_path, args.preview)

    print(f"[done] train={len(train_rows)} -> {train_path}")
    print(f"[done] val={len(val_rows)}   -> {val_path}")
    print(f"[done] preview={min(args.preview, len(train_rows))} -> {preview_path}")


if __name__ == "__main__":
    main()
