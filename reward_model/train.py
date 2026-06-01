from __future__ import annotations

import sys
import yaml
import os
import torch
from datetime import datetime
from pathlib import Path

from datasets import load_dataset, load_from_disk, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig, TrainerCallback
from peft import LoraConfig, TaskType
from trl import RewardTrainer, RewardConfig
from transformers.trainer_utils import get_last_checkpoint

# Default config shipped with the repo — override by passing a path as first CLI arg.
_DEFAULT_CONFIG = str(Path(__file__).resolve().parent / "configs" / "train_lora.yaml")

class SaveAtEpochEndCallback(TrainerCallback):
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir

    def on_epoch_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if model is not None and state.epoch is not None:
            epoch_dir = os.path.join(self.checkpoint_dir, f"epoch-{int(state.epoch)}")
            os.makedirs(epoch_dir, exist_ok=True)
            # Keep epoch checkpoints aligned with Trainer LoRA checkpoint style.
            model.save_pretrained(epoch_dir)
            if tokenizer is not None:
                tokenizer.save_pretrained(epoch_dir)

class EpochProgressCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        if state.epoch is not None:
            total_epochs = int(args.num_train_epochs)
            current_epoch = int(state.epoch) + 1
            print(f"[Epoch {current_epoch}/{total_epochs}] started.")

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch is not None:
            total_epochs = int(args.num_train_epochs)
            current_epoch = int(state.epoch)
            print(f"[Epoch {current_epoch}/{total_epochs}] finished.")

def read_config(path: str) -> dict:
    with open (path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Config file {path} does not contain a valid dictionary.")
    
    return config

def parse_cli_args() -> tuple[str, list[str]]:
    """Parse CLI as: [config_path] [key=value ...]."""
    args = sys.argv[1:]
    if not args:
        return _DEFAULT_CONFIG, []
    if "=" in args[0]:
        return _DEFAULT_CONFIG, args
    return args[0], args[1:]

def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply dot-list overrides like model.base_model=... to config dict."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Expected key=value.")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid override '{item}'. Empty key.")

        value = yaml.safe_load(raw_value)
        parts = key.split(".")
        node = cfg
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                node[p] = {}
            node = node[p]
        node[parts[-1]] = value
    return cfg

def load_pairwise_dataset(cfg: dict) -> DatasetDict:
    data_cfg = cfg.get("data", {})
    tokenized_dir = data_cfg.get("tokenized_dir")
    if tokenized_dir:
        dataset = load_from_disk(tokenized_dir)
        if not isinstance(dataset, DatasetDict):
            raise ValueError(
                f"Tokenized dataset at '{tokenized_dir}' is not a DatasetDict. "
                "Expected splits: train/validation."
            )
        if "train" not in dataset or "validation" not in dataset:
            raise ValueError(
                f"Tokenized dataset at '{tokenized_dir}' must contain both "
                f"'train' and 'validation' splits. Found: {list(dataset.keys())}"
            )
        return dataset

    train_file = data_cfg.get("train_file")
    val_file = data_cfg.get("val_file")

    data_files = {"train": train_file}
    if val_file:
        data_files["validation"] = val_file

    dataset = load_dataset("json", data_files=data_files)

    if "validation" not in dataset:
        split = dataset["train"].train_test_split(test_size=0.1, seed=42)
        dataset = DatasetDict({
            "train": split["train"],
            "validation": split["test"]
        })

    for split in ["train", "validation"]:
        cols = set(dataset[split].column_names)
        has_raw = {"chosen_text", "rejected_text"}.issubset(cols)
        has_tokenized = {"chosen_input_ids", "rejected_input_ids"}.issubset(cols)
        if not (has_raw or has_tokenized):
            raise ValueError(
                f"Dataset split '{split}' must contain either raw columns "
                f"{{'chosen_text','rejected_text'}} or tokenized columns "
                f"{{'chosen_input_ids','rejected_input_ids'}}. Found columns: {cols}"
            )
        
    return dataset

def prepare_reward_dataset(dataset: DatasetDict) -> DatasetDict:
    def _map_fn(ex: dict) -> dict:
        chosen_text = (ex.get("chosen_text") or "").strip()
        rejected_text = (ex.get("rejected_text") or "").strip()
        if not chosen_text or not rejected_text:
            raise ValueError("Raw dataset rows must contain non-empty chosen_text and rejected_text.")
        return {"chosen": chosen_text, "rejected": rejected_text}

    dataset = dataset.map(_map_fn, remove_columns=dataset["train"].column_names)
    return dataset

def normalize_tokenized_dataset_columns(dataset: DatasetDict) -> DatasetDict:
    """Support legacy offline-tokenized column names."""
    legacy_to_new = {
        "input_ids_chosen": "chosen_input_ids",
        "input_ids_rejected": "rejected_input_ids",
    }
    for split in dataset.keys():
        cols = set(dataset[split].column_names)
        rename_map = {old: new for old, new in legacy_to_new.items() if old in cols and new not in cols}
        if rename_map:
            dataset[split] = dataset[split].rename_columns(rename_map)
    return dataset

def build_model_and_tokenizer(cfg: dict):
    model_cfg = cfg.get("model", {})
    base_model = model_cfg.get("base_model")
    max_length = int(model_cfg.get("max_length", 2048))
    use_4bit = model_cfg.get("use_4bit", True)
    bf16 = model_cfg.get("bf16", True)
    fp16 = model_cfg.get("fp16", True)

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    tokenizer.model_max_length = max_length
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16
        )

    torch_dtype = torch.bfloat16 if bf16 else torch.float16 if (fp16 or use_4bit) else torch.float32

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if use_4bit and world_size > 1:
        device_map = {"": local_rank}
    elif use_4bit:
        device_map = "auto"
    else:
        device_map = None

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=1,
        torch_dtype=torch_dtype,
        quantization_config=quant_config,
        device_map=device_map
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    return model, tokenizer
     
def build_lora_config(cfg: dict) -> LoraConfig:
    lora_cfg = cfg.get("lora", {})
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        ),
        bias="none"
    )
    return lora_config

def build_trainer_args(cfg: dict) -> RewardConfig:
    t = cfg.get("train", {})
    m = cfg.get("model", {})
    return RewardConfig(
        output_dir=t.get("rolling_dir"),
        num_train_epochs=float(t.get("num_train_epochs", 1)),
        learning_rate=float(t.get("learning_rate", 1e-5)),
        per_device_train_batch_size=int(t.get("per_device_train_batch_size", 2)),
        per_device_eval_batch_size=int(t.get("per_device_eval_batch_size", 2)),
        gradient_accumulation_steps=int(t.get("gradient_accumulation_steps", 16)),
        logging_steps=int(t.get("logging_steps", 10)),
        eval_strategy=t.get("eval_strategy", "steps"),
        eval_steps=int(t.get("eval_steps", 100)),
        save_strategy=t.get("save_strategy", "steps"),
        save_steps=int(t.get("save_steps", 100)),
        save_total_limit=int(t.get("save_total_limit", 1)),
        max_length=int(m.get("max_length", 2048)),
        weight_decay=float(t.get("weight_decay", 0.0)),
        report_to=t.get("report_to", "none"),
        run_name=t.get("run_name"),
    )

def save_resolved_config(cfg: dict) -> str:
    """Save the final resolved config (with CLI overrides applied)."""
    output_cfg = cfg.get("output", {})
    config_dir = output_cfg.get("config_dir")
    if not config_dir:
        config_dir = os.path.join(cfg.get("train", {}).get("rolling_dir", "outputs"), "configs")
    os.makedirs(config_dir, exist_ok=True)

    run_name = cfg.get("train", {}).get("run_name", "run")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(config_dir, f"{run_name}_{timestamp}.yaml")
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return save_path


def main() -> None:
    config_path, overrides = parse_cli_args()
    cfg = read_config(config_path)
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    print("Loaded configuration:", config_path)
    if overrides:
        print("Applied overrides:", overrides)
    print("Keys:", list(cfg.keys()))
    wandb_cfg = cfg.get("wandb", {})
    if wandb_cfg.get("project"):
        os.environ["WANDB_PROJECT"] = str(wandb_cfg.get("project"))
    if wandb_cfg.get("entity"):
        os.environ["WANDB_ENTITY"] = str(wandb_cfg.get("entity"))

    dataset = load_pairwise_dataset(cfg)
    dataset = normalize_tokenized_dataset_columns(dataset)
    print("Dataset loaded successfully.")
    print("Train size:", len(dataset["train"]))
    print("Validation size:", len(dataset["validation"]))

    train_cols = set(dataset["train"].column_names)
    is_tokenized = {"chosen_input_ids", "rejected_input_ids"}.issubset(train_cols)
    if is_tokenized:
        print("Detected tokenized dataset. Skipping online text preparation/tokenization.")
        print("Train columns:", dataset["train"].column_names)
    else:
        dataset = prepare_reward_dataset(dataset)
        print("Prepared columns:", dataset["train"].column_names)
        print("Sample chosen[:120]:", dataset["train"][0]["chosen"][:120])

    model, tokenizer = build_model_and_tokenizer(cfg)
    print("Model and tokenizer loaded successfully.")
    print("Model: ", cfg.get("model").get("base_model"))
    print("Tokenizer pad_token_id: ", tokenizer.pad_token_id) 

    peft_config = build_lora_config(cfg)
    print("LoRA configuration:", peft_config)
    trainer_args = build_trainer_args(cfg)
    print("Trainer configuration:", trainer_args)

    rolling_dir = cfg.get("train").get("rolling_dir")
    checkpoint_dir = cfg.get("train").get("checkpoint_dir")
    os.makedirs(rolling_dir, exist_ok=True) 
    os.makedirs(checkpoint_dir, exist_ok=True)
    config_snapshot_path = save_resolved_config(cfg)
    print("Saved resolved config to:", config_snapshot_path)

    trainer = RewardTrainer(
        model=model,
        args=trainer_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.add_callback(SaveAtEpochEndCallback(checkpoint_dir=checkpoint_dir))
    trainer.add_callback(EpochProgressCallback())

    resume_if_possible = cfg.get("train").get("resume_if_possible", True)
    last_ckpt = get_last_checkpoint(rolling_dir) if resume_if_possible else None
    if last_ckpt:
        print(f"Resuming training from rolling checkpoint: {last_ckpt}")
        trainer.train(resume_from_checkpoint=last_ckpt)
    else:
        print("No rolling checkpoint found, starting fresh training.")
        trainer.train()

    trainer.save_model(cfg.get("train").get("final_dir"))
    print("Training completed and model saved to:", cfg.get("train").get("final_dir"))
    tokenizer.save_pretrained(cfg.get("train").get("final_dir"))
    print("Tokenizer saved to:", cfg.get("train").get("final_dir"))

if __name__ == "__main__":
    main()
