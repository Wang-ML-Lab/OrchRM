from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
from pathlib import Path
from typing import Any

import torch
from peft import PeftConfig, PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

# Add repo root to path so `prompts` package is importable when running as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_single_json_file(p: Path) -> list[dict[str, Any]]:
    suffix = p.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with p.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSONL at line {i} in {p}: {e}") from e
                if not isinstance(obj, dict):
                    raise TypeError(f"Line {i} in {p} is not a JSON object")
                rows.append(obj)
        return rows

    if suffix == ".json":
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            rows = data["items"] if isinstance(data.get("items"), list) else [data]
        elif isinstance(data, list):
            rows = data
        else:
            raise TypeError(f"Unsupported JSON root type in {p}: {type(data)}")
        if not all(isinstance(x, dict) for x in rows):
            raise TypeError(f"JSON entries in {p} must all be objects")
        return rows

    raise ValueError(f"Unsupported file type: {p}. Expected .json or .jsonl")


def _load_json(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")
    if p.is_dir():
        jsonl_files = sorted(p.glob("*.jsonl"))
        json_files = sorted(p.glob("*.json"))
        files = jsonl_files if jsonl_files else json_files
        if not files:
            raise ValueError(f"No .jsonl/.json files found under: {p}")
        if jsonl_files and json_files:
            print(f"[load] ignoring .json files because .jsonl shards are present under {p}", flush=True)
        all_rows: list[dict[str, Any]] = []
        for fp in files:
            all_rows.extend(_load_single_json_file(fp))
        return all_rows
    return _load_single_json_file(p)


def _save_json(data: list[dict[str, Any]], output_path: str) -> None:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".jsonl":
        with p.open("w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return
    if p.suffix.lower() == ".json":
        with p.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return
    raise ValueError("Output path must end with .jsonl or .json")


# ---------------------------------------------------------------------------
# Resume / grouping helpers
# ---------------------------------------------------------------------------

def _sample_key(sample_idx: Any) -> str:
    return str(sample_idx)


def _sample_sort_key(sample: dict[str, Any]) -> tuple[int, int | str]:
    idx = sample.get("sample_idx")
    try:
        return (0, int(idx))
    except (TypeError, ValueError):
        return (1, str(idx))


def _load_existing_output_map(output_path: str) -> tuple[dict[str, dict[str, dict[str, Any]]], list[str]]:
    p = Path(output_path)
    if not p.exists():
        return {}, []
    rows = _load_json(str(p))
    qid_to_samples: dict[str, dict[str, dict[str, Any]]] = {}
    qid_order: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        qid = str(row.get("qid"))
        if qid not in qid_to_samples:
            qid_to_samples[qid] = {}
            qid_order.append(qid)
        for sample in row.get("samples", []):
            if not isinstance(sample, dict):
                continue
            key = _sample_key(sample.get("sample_idx"))
            if key not in qid_to_samples[qid]:
                qid_to_samples[qid][key] = {
                    "sample_idx": sample.get("sample_idx"),
                    "rm_score": sample.get("rm_score"),
                    "is_truncated": sample.get("is_truncated"),
                }
    return qid_to_samples, qid_order


def _materialize_grouped_rows(
    qid_to_samples: dict[str, dict[str, dict[str, Any]]], qid_order: list[str]
) -> list[dict[str, Any]]:
    rows = []
    for qid in qid_order:
        samples = sorted(qid_to_samples.get(qid, {}).values(), key=_sample_sort_key)
        rows.append({"qid": qid, "samples": samples})
    return rows


def _normalize_output_path(output_path: str, input_path: str, dataset_name: str) -> str:
    out = Path(output_path)
    if out.suffix.lower() in {".jsonl", ".json"}:
        return str(out)
    return str(out / f"{dataset_name}_rm_scores.jsonl")


def _dataset_dir_candidates(dataset_name: str) -> list[str]:
    cands = [dataset_name]
    aliases = {
        "browsecomp": ["browsecomp-plus", "browsecomp_validated", "browsecomp-plus_validated"],
        "hotpotqa": ["hotpotqa-distractor", "hotpotqa-distractor_validated"],
        "deepscaler": ["deepscalar", "deepscalar_validated"],
    }
    cands.extend(aliases.get(dataset_name.strip(), []))
    seen, dedup = set(), []
    for c in cands:
        if c not in seen:
            seen.add(c)
            dedup.append(c)
    return dedup


def _find_dataset_input_dir(model_rollout_dir: str, dataset_name: str) -> str | None:
    base = Path(model_rollout_dir)
    for cand in _dataset_dir_candidates(dataset_name):
        p = base / cand
        if p.exists() and p.is_dir():
            return str(p)
    return None


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def _build_rm_input(
    question: str,
    response: str,
    qwen_tokenizer: AutoTokenizer | None,
    llama_tokenizer: AutoTokenizer | None,
    problem_type: str,
    node_model: str,
    dataset_name: str,
) -> str:
    from utils.prompts.make_prompt import make_rm_prompt

    return make_rm_prompt(
        question=question,
        problem_type=problem_type,
        node_model=node_model,
        dataset_name=dataset_name,
        answer=response,
        tokenizer_qwen=qwen_tokenizer,
        tokenizer_llama=llama_tokenizer,
    )


def _extract_question(item: dict[str, Any]) -> str:
    q = item.get("question")
    if isinstance(q, str) and q.strip():
        return q
    raise ValueError("Each input row must contain a non-empty 'question' field.")


def _extract_response(sample: Any) -> str:
    if isinstance(sample, dict):
        resp = sample.get("response")
        if isinstance(resp, str):
            return resp
    raise ValueError("Each sample must be an object with a string 'response' field.")


def _build_question_samples(
    item: dict[str, Any],
    qwen_tokenizer: AutoTokenizer | None,
    llama_tokenizer: AutoTokenizer | None,
    item_idx: int,
    problem_type: str,
    node_model: str,
    dataset_name: str,
) -> tuple[Any, list[dict[str, Any]]]:
    qid = item.get("qid", item.get("id", item_idx))
    question = _extract_question(item)
    samples = item.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Each input row must contain a non-empty 'samples' list.")

    flat: list[dict[str, Any]] = []
    for s_idx, sample in enumerate(samples):
        response = _extract_response(sample)
        rm_input = _build_rm_input(
            question, response, qwen_tokenizer, llama_tokenizer,
            problem_type=problem_type, node_model=node_model, dataset_name=dataset_name,
        )
        sample_idx = sample.get("sample_idx", s_idx) if isinstance(sample, dict) else s_idx
        finish_reason = sample.get("finish_reason") if isinstance(sample, dict) else None
        flat.append({
            "_item_idx": item_idx,
            "_sample_pos": s_idx,
            "qid": qid,
            "text": rm_input,
            "sample_idx": sample_idx,
            "response": response,
            "finish_reason": finish_reason,
        })
    return qid, flat


def _check_truncate(sample: dict[str, Any]) -> bool:
    reason = sample.get("finish_reason")
    return isinstance(reason, str) and reason.lower() in {"length", "max_tokens"}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _tokenize(tokenizer: AutoTokenizer, texts: list[str], max_length: int) -> dict[str, torch.Tensor]:
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        add_special_tokens=False,
    )


def _build_model(model_path: str, max_length: int, load_in_4bit: bool, bf16: bool):
    is_peft = False
    base_model = model_path
    tokenizer_path = model_path
    try:
        peft_cfg = PeftConfig.from_pretrained(model_path)
        base_model = peft_cfg.base_model_name_or_path
        tokenizer_path = base_model
        is_peft = True
    except Exception:
        pass

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    tokenizer.model_max_length = max_length
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
        )

    torch_dtype = torch.bfloat16 if bf16 else torch.float16
    device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() and load_in_4bit else None

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=1,
        torch_dtype=torch_dtype,
        quantization_config=quant_config,
        device_map=device_map,
    )
    if is_peft:
        peft_kwargs: dict[str, Any] = {} if load_in_4bit else {"torch_device": "cpu"}
        model = PeftModel.from_pretrained(model, model_path, **peft_kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.eval()

    if not load_in_4bit:
        device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
        model.to(device)

    return model, tokenizer


# ---------------------------------------------------------------------------
# Core inference worker
# ---------------------------------------------------------------------------

def _inference(
    input_path: str,
    output_path: str,
    model_path: str,
    qwen_model_path: str,
    batch_size: int,
    max_length: int,
    gpu_id: str,
    load_in_4bit: bool,
    bf16: bool,
    drop_truncated: bool,
    problem_type: str,
    node_model: str,
    dataset_name: str,
    start_question_idx: int = 0,
    end_question_idx: int | None = None,
) -> None:
    if gpu_id and torch.cuda.is_available():
        torch.cuda.set_device(int(gpu_id))
        print(f"[worker] bound to cuda:{gpu_id} ({torch.cuda.get_device_name(int(gpu_id))})", flush=True)

    raw = _load_json(input_path)[start_question_idx:end_question_idx]
    if not raw:
        print("No questions to score.")
        _save_json([], output_path)
        return

    model, llama_tokenizer = _build_model(model_path, max_length, load_in_4bit, bf16)
    qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_model_path, trust_remote_code=True, use_fast=False)

    model_device = None if load_in_4bit else next(model.parameters()).device
    qid_to_samples, qid_order = _load_existing_output_map(output_path)
    existing = sum(len(v) for v in qid_to_samples.values())
    if existing:
        print(f"[worker] resume: {existing} existing samples across {len(qid_to_samples)} qids", flush=True)

    stream_jsonl = Path(output_path).suffix.lower() == ".jsonl"
    questions_done = samples_added = skipped_rows = 0

    with torch.no_grad():
        for local_idx, item in enumerate(tqdm(raw, desc="Scoring", unit="q")):
            try:
                qid, flat = _build_question_samples(
                    item, qwen_tokenizer, llama_tokenizer, local_idx,
                    problem_type=problem_type, node_model=node_model, dataset_name=dataset_name,
                )
            except (TypeError, ValueError) as exc:
                skipped_rows += 1
                print(f"[worker] skipping row {start_question_idx + local_idx}: {exc}", flush=True)
                continue

            if drop_truncated:
                flat = [x for x in flat if not _check_truncate(x)]
            if not flat:
                continue

            qid_key = str(qid)
            if qid_key not in qid_to_samples:
                qid_to_samples[qid_key] = {}
                qid_order.append(qid_key)

            existing_keys = set(qid_to_samples[qid_key].keys())
            pending = [x for x in flat if _sample_key(x.get("sample_idx")) not in existing_keys]
            if not pending:
                continue

            q_samples: list[dict[str, Any]] = []
            for i in range(0, len(pending), batch_size):
                batch = pending[i: i + batch_size]
                tok = _tokenize(llama_tokenizer, [str(x["text"]) for x in batch], max_length)
                if model_device is not None:
                    tok = {k: v.to(model_device) for k, v in tok.items()}
                logits = model(**tok).logits.squeeze(-1)
                scores = logits.detach().float().cpu().tolist()
                if isinstance(scores, float):
                    scores = [scores]
                for row, score in zip(batch, scores):
                    q_samples.append({
                        "sample_idx": row["sample_idx"],
                        "rm_score": float(score),
                        "is_truncated": _check_truncate(row),
                    })

            for sample in q_samples:
                key = _sample_key(sample["sample_idx"])
                if key not in qid_to_samples[qid_key]:
                    qid_to_samples[qid_key][key] = sample
                    samples_added += 1
            questions_done += 1

            if stream_jsonl:
                _save_json(_materialize_grouped_rows(qid_to_samples, qid_order), output_path)

    _save_json(_materialize_grouped_rows(qid_to_samples, qid_order), output_path)
    print(f"Done: {questions_done} qids, {samples_added} samples added, {skipped_rows} skipped. -> {output_path}")


# ---------------------------------------------------------------------------
# Multi-GPU parallel dispatch
# ---------------------------------------------------------------------------

def _resolve_total_questions(input_path: str) -> int:
    return len(_load_json(input_path))


def _infer_dataset_name(input_path: str, dataset_name: str) -> str:
    if dataset_name.strip():
        return dataset_name.strip()
    stem = Path(input_path).stem
    return re.sub(r"_[0-9]+_[0-9]+$", "", stem)


def _make_shard_output_path(output_path: str, input_path: str, dataset_name: str, start: int, end: int) -> str:
    p = Path(output_path)
    name = _infer_dataset_name(input_path, dataset_name)
    return str(p.with_name(f"{name}_rm_scores_{start}_{end}{p.suffix}"))


def _merge_shard_outputs(shard_paths: list[str], output_path: str) -> None:
    qid_to_samples: dict[str, dict[str, dict[str, Any]]] = {}
    qid_order: list[str] = []
    for shard in shard_paths:
        for row in _load_json(shard):
            qid_key = str(row.get("qid"))
            if qid_key not in qid_to_samples:
                qid_to_samples[qid_key] = {}
                qid_order.append(qid_key)
            for sample in row.get("samples", []):
                if not isinstance(sample, dict):
                    continue
                key = _sample_key(sample.get("sample_idx"))
                if key not in qid_to_samples[qid_key]:
                    qid_to_samples[qid_key][key] = {
                        "sample_idx": sample.get("sample_idx"),
                        "rm_score": sample.get("rm_score"),
                        "is_truncated": sample.get("is_truncated"),
                    }
    _save_json(_materialize_grouped_rows(qid_to_samples, qid_order), output_path)
    print(f"Merged {len(shard_paths)} shards -> {output_path}")


def _run_parallel(args: argparse.Namespace, input_path: str, output_path: str) -> None:
    gpu_ids = [g.strip() for g in args.gpu_ids.split(",") if g.strip()]
    total = _resolve_total_questions(input_path)
    if total == 0:
        _save_json([], output_path)
        return

    num_workers = min(len(gpu_ids), total)
    shard_size = (total + num_workers - 1) // num_workers
    processes, shard_paths = [], []

    for i in range(num_workers):
        s = i * shard_size
        e = min((i + 1) * shard_size, total)
        if s >= e:
            continue
        shard_out = _make_shard_output_path(output_path, input_path, args.dataset_name, s, e)
        shard_paths.append(shard_out)
        p = mp.Process(
            target=_inference,
            args=(input_path, shard_out, args.model_path, args.qwen_model_path,
                  args.batch_size, args.max_length, gpu_ids[i],
                  args.load_in_4bit, not args.fp16, args.drop_truncated,
                  args.problem_type, args.node_model, args.dataset_name, s, e),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Worker failed with exit code {p.exitcode}")

    _merge_shard_outputs(shard_paths, output_path)


def _run_loop(args: argparse.Namespace) -> None:
    rollout_dirs = [d.strip() for d in args.rollout_dirs.split(",") if d.strip()]
    dataset_map: dict[str, str] = {}
    for entry in args.datasets.split(","):
        entry = entry.strip()
        if ":" in entry:
            k, v = entry.split(":", 1)
            dataset_map[k.strip()] = v.strip()
        elif entry:
            dataset_map[entry] = entry

    found = 0
    for model_dir in rollout_dirs:
        model_name = Path(model_dir).name
        for prompt_data_name, dataset_name in dataset_map.items():
            input_dir = _find_dataset_input_dir(model_dir, dataset_name)
            if input_dir is None:
                print(f"[skip] '{dataset_name}' not found under {model_dir}")
                continue
            found += 1
            dataset_leaf = Path(input_dir).name
            out_path = str(Path(args.output_root) / model_name / dataset_leaf / f"{dataset_name}_rm_scores.jsonl")
            print(f"\n=== {model_name} | {dataset_leaf} ===")
            # Temporarily patch dataset_name for this run
            original = args.dataset_name
            args.dataset_name = dataset_name
            _run_single(args, input_dir, out_path)
            args.dataset_name = original

    if found == 0:
        raise FileNotFoundError(f"No datasets found in any rollout dir: {rollout_dirs}")


def _run_single(args: argparse.Namespace, input_path: str, output_path: str) -> None:
    gpu_ids = [g.strip() for g in args.gpu_ids.split(",") if g.strip()]
    dataset_name = _infer_dataset_name(input_path, args.dataset_name)
    output_path = _normalize_output_path(output_path, input_path, dataset_name)
    print(f"Input:  {input_path}")
    print(f"Model:  {args.model_path}")
    print(f"Output: {output_path}")
    print(f"GPUs:   {gpu_ids}")

    if args.auto_split and len(gpu_ids) > 1:
        _run_parallel(args, input_path, output_path)
    else:
        _inference(
            input_path=input_path,
            output_path=output_path,
            model_path=args.model_path,
            qwen_model_path=args.qwen_model_path,
            batch_size=args.batch_size,
            max_length=args.max_length,
            gpu_id=gpu_ids[0],
            load_in_4bit=args.load_in_4bit,
            bf16=not args.fp16,
            drop_truncated=args.drop_truncated,
            problem_type=args.problem_type,
            node_model=args.node_model,
            dataset_name=args.dataset_name,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run OrchRM inference over rollout files.")

    # Paths
    p.add_argument("--input",        required=False, default="", help="Input JSON/JSONL file or directory of rollout samples.")
    p.add_argument("--output",       required=False, default="", help="Output path (.jsonl/.json) or directory.")
    p.add_argument("--model-path",   required=True,  help="Path to the trained reward model (HF or LoRA adapter dir).")
    p.add_argument("--qwen-model-path", required=True, help="Path to Qwen tokenizer (e.g. Qwen2.5-7B-Instruct).")

    # Prompt config
    p.add_argument("--problem-type", default="harmony_minimal",  help="Prompt problem type (e.g. harmony_minimal, harmony_medium).")
    p.add_argument("--node-model",   default="gpt-120-oss",           help="Sub-agent model name used in prompt formatting.")
    p.add_argument("--dataset-name", default="",                  help="Dataset name for prompt formatting and output naming.")

    # Inference config
    p.add_argument("--gpu-ids",      default="0",   help="Comma-separated GPU IDs, e.g. '0,1,2'.")
    p.add_argument("--batch-size",   type=int, default=64)
    p.add_argument("--max-length",   type=int, default=8192)
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--fp16",         action="store_true", help="Use FP16 instead of BF16.")
    p.add_argument("--drop-truncated", action="store_true", help="Skip samples truncated at max_length.")
    p.add_argument("--auto-split",   action="store_true", default=True,
                   help="Auto-split work across GPUs (default: True). Disable with --no-auto-split.")
    p.add_argument("--no-auto-split", dest="auto_split", action="store_false")

    # Loop-over-models mode
    p.add_argument("--loop",         action="store_true", help="Loop over multiple rollout model dirs.")
    p.add_argument("--rollout-dirs", default="",   help="Comma-separated rollout model dirs (used with --loop).")
    p.add_argument("--datasets",     default="",   help="Comma-separated dataset names or 'prompt_name:dataset_name' pairs (used with --loop).")
    p.add_argument("--output-root",  default="./outputs/rm_scores", help="Root output dir for --loop mode.")

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.loop:
        if not args.rollout_dirs:
            raise ValueError("--rollout-dirs is required with --loop")
        if not args.datasets:
            raise ValueError("--datasets is required with --loop")
        _run_loop(args)
    else:
        if not args.input or not args.output:
            raise ValueError("--input and --output are required (or use --loop mode)")
        _run_single(args, args.input, args.output)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
