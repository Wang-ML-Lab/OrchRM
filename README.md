# OrchRM

**OrchRM** is a reward model for multi-agent orchestration systems, trained on preference data across three domains: math reasoning (DeepScaler), multi-hop QA (HotpotQA), and web-based research (BrowseComp+). It is integrated into a GRPO training pipeline to improve a multi-agent system's ability to plan and execute complex tasks.

It is presented in our paper:

> **[Title]** [Authors]
> *[Venue, Year]*
> [[arXiv]]() | [[Paper]]()

## Citation

```bibtex
@article{placeholder,
  title     = {},
  author    = {},
  journal   = {},
  year      = {}
}
```

## Updates

## Repository Structure

```
OrchRM/
├── reward_model/            # RM training, inference, tokenization
│   └── configs/             # training config (YAML)
├── grpo/                    # GRPO training pipeline
│   ├── reward_function.py   # custom reward function (RM server / LLM judge / exact match)
│   ├── make_dataset.py      # build parquet datasets for verl
│   └── scripts/run.sh       # training launch script
├── mas_orchestra/           # multi-agent system (agents, rewards, trainer, prompts)
├── utils/
│   └── prompts/             # prompt templates (system, user, developer, block-level)
│       ├── make_prompt.py   # prompt builder entry point
│       ├── init_archive.py  # agent block archive
│       ├── blocks_harmony/  # CoT, CoT-SC, Reflexion, Debate, WebSearch blocks
│       ├── system_prompts/
│       ├── user_prompts/
│       └── developer_prompts/
└── verl/                    # modified verl fork (GRPO/PPO training backend)
```

## Dataset

The preference dataset used to train OrchRM is publicly available on Hugging Face:

**[tsangkingyeung/OrchRM-datasets](https://huggingface.co/datasets/tsangkingyeung/OrchRM-datasets)**

| Split | Domain | Rows |
|---|---|---|
| `deepscaler_train` | Math (DeepScaler) | 4,587 |
| `deepscaler_val` | Math (DeepScaler) | 531 |
| `hotpotqa_train` | Multi-hop QA (HotpotQA) | 16,932 |
| `hotpotqa_val` | Multi-hop QA (HotpotQA) | 1,828 |
| `browsecomp_train` | Web Research (BrowseComp+) | 3,706 |
| `browsecomp_val` | Web Research (BrowseComp+) | 387 |

**Fields:** `input` (question), `chosen` (preferred response), `rejected` (dispreferred response), `source` (`correct-over-incorrect` | `specialized-over-base`)

```python
from datasets import load_dataset

ds = load_dataset("tsangkingyeung/OrchRM-datasets", split="deepscaler_train")
```

## Installation

```bash
# 1. Install verl
cd verl
pip install --no-deps -e .
pip install ray==2.49.2 --force-reinstall
pip install protobuf==4.25.8 --force-reinstall
cd ..

# 2. Install OrchRM
pip install -e .

# 3. Install remaining dependencies
pip install -r requirements.txt
```

For web-search agent support (BrowseComp+), also install:
```bash
pip install langchain-core langchain-together langchain-community \
    duckduckgo-search tavily-python ddgs langchain_brightdata bs4 \
    pyserini faiss-gpu
pip install git+https://github.com/texttron/tevatron.git
```

## Training the Reward Model

**Step 1 — Tokenize the dataset offline** (optional but recommended for speed):
```bash
python reward_model/tokenize.py \
    --train-path /path/to/train.jsonl \
    --val-path   /path/to/val.jsonl \
    --tokenizer-path Skywork/Skywork-Reward-Llama-3.1-8B \
    --out-dir    /path/to/tokenized_output \
    --max-length 7168
```

**Step 2 — Train with LoRA:**
```bash
# Edit reward_model/configs/train_lora.yaml to set paths, then:
python reward_model/train.py reward_model/configs/train_lora.yaml

# Or pass overrides inline:
python reward_model/train.py reward_model/configs/train_lora.yaml \
    model.base_model=Skywork/Skywork-Reward-Llama-3.1-8B \
    train.rolling_dir=./outputs/orchrm
```

Key config fields in [reward_model/configs/train_lora.yaml](reward_model/configs/train_lora.yaml):
- `model.base_model`: base reward model (e.g. `Skywork/Skywork-Reward-Llama-3.1-8B`)
- `data.tokenized_dir`: path to pre-tokenized dataset (or set `data.train_file` / `data.val_file` for raw JSONL)
- `train.rolling_dir`: output directory for checkpoints

**Step 3 — Run inference:**
```bash
python reward_model/inference.py \
    --input  /path/to/rollout/ \
    --output /path/to/scores/ \
    --model-path      /path/to/trained/orchrm \
    --qwen-model-path Qwen/Qwen2.5-7B-Instruct \
    --dataset-name    math \
    --gpu-ids         0,1
```

## GRPO Training with OrchRM

**Step 1 — Build training parquets:**
```bash
python grpo/make_dataset.py \
    --output-dir        ./datasets/grpo \
    --qwen-tokenizer-path Qwen/Qwen2.5-7B-Instruct \
    --problem-type      harmony_medium
```

**Step 2 — Run GRPO:**
```bash
export MODEL_PATH=/path/to/base/model        # actor model (e.g. Qwen2.5-7B-Instruct)
export RM_MODEL_PATH=/path/to/trained/orchrm # trained reward model

bash grpo/scripts/run.sh \
    ./datasets/grpo/train/train.parquet \
    ./datasets/grpo/train/val.parquet
```

See [grpo/scripts/run.sh](grpo/scripts/run.sh) for all configurable env vars (`LORA_RANK`, `GPU_IDS`, `TRAIN_BATCH_SIZE`, `TOTAL_EPOCHS`, etc.).

## Reward Function

[grpo/reward_function.py](grpo/reward_function.py) supports three reward paths, tried in priority order:

1. **LLM judge** — set `OPENAI_API_KEY` and optionally `LLM_JUDGE_MODEL` (default: `gpt-5-mini`)
2. **RM server** — set `reward_router_address` to a running vLLM classify endpoint backed by OrchRM
3. **Exact match** — fallback when neither above is available

## Acknowledgements

- Multi-agent orchestration framework adapted from [MAS-Orchestra](https://github.com/SalesforceAIResearch/MAS-Orchestra) (Salesforce AI Research)
- Reward model training based on [Skywork-Reward](https://github.com/SkyworkAI/Skywork-Reward)
- GRPO training backend: [verl](https://github.com/volcengine/verl)

## Contact

For questions or feedback, feel free to open a [GitHub Issue](https://github.com/Wang-ML-Lab/OrchRM/issues) or reach out directly:

- **King Yeung Tsang** — kingyeung.tsang@gmail.com