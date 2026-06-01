from __future__ import annotations

import json
from pathlib import Path


def _normalize_dataset_name(dataset_name: str) -> str:
    name = (dataset_name or "").lower()
    if "browsecomp" in name:
        return "browsecomp"
    if "hotpot" in name:
        return "hotpot"
    if "math" in name or "deepscal" in name:
        return "math"
    return name


def make_prompt(question: str, problem_type: str, node_model: str, dataset_name: str, init_archive=""):
    from utils.prompts.init_archive import get_init_archive

    if problem_type == "harmony_medium":
        from utils.prompts.system_prompts.harmony_medium import system_prompt
        from utils.prompts.user_prompts.propose_harmony_medium import base
        from utils.prompts.developer_prompts.harmony_medium import developer_prompt
    elif problem_type == "harmony_minimal":
        from utils.prompts.system_prompts.harmony_minimal import system_prompt
        from utils.prompts.user_prompts.propose_harmony_minimal import base
        from utils.prompts.developer_prompts.harmony_minimal import developer_prompt
    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")

    archive = get_init_archive(init_archive if init_archive else problem_type)

    system_prompt = system_prompt.replace("[MODEL]", node_model)
    user_prompt = base.replace("[QUESTION]", question)

    if problem_type == "harmony_medium":
        developer_prompt = developer_prompt.replace("[COT]",       json.dumps(archive[0]))
        developer_prompt = developer_prompt.replace("[COT_SC]",    json.dumps(archive[1]))
        developer_prompt = developer_prompt.replace("[Reflexion]", json.dumps(archive[2]))
        developer_prompt = developer_prompt.replace("[Debate]",    json.dumps(archive[3]))
        developer_prompt = developer_prompt.replace("[Websearch]", json.dumps(archive[4]))
    elif problem_type == "harmony_minimal":
        developer_prompt = developer_prompt.replace("[COT]",       json.dumps(archive[0]))
        developer_prompt = developer_prompt.replace("[COT_SC]",    json.dumps(archive[1]))
        developer_prompt = developer_prompt.replace("[Reflexion]", json.dumps(archive[2]))
        developer_prompt = developer_prompt.replace("[Debate]",    json.dumps(archive[3]))

    return system_prompt, user_prompt, developer_prompt


def make_prompt_no_examples(question: str, problem_type: str, node_model: str, dataset_name: str, init_archive=""):
    from utils.prompts.init_archive import get_init_archive

    if problem_type == "harmony_medium":
        from utils.prompts.system_prompts.harmony_medium import system_prompt
        from utils.prompts.user_prompts.rm_harmony_medium import base
        from utils.prompts.developer_prompts.harmony_medium import developer_prompt
    elif problem_type == "harmony_minimal":
        from utils.prompts.system_prompts.harmony_minimal import system_prompt
        from utils.prompts.user_prompts.rm_harmony_minimal import base
        from utils.prompts.developer_prompts.harmony_minimal import developer_prompt
    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")

    archive = get_init_archive(init_archive if init_archive else problem_type)

    system_prompt = system_prompt.replace("[MODEL]", node_model)
    user_prompt = base.replace("[QUESTION]", question)

    if problem_type == "harmony_medium":
        developer_prompt = developer_prompt.replace("[COT]",       json.dumps(archive[0]))
        developer_prompt = developer_prompt.replace("[COT_SC]",    json.dumps(archive[1]))
        developer_prompt = developer_prompt.replace("[Reflexion]", json.dumps(archive[2]))
        developer_prompt = developer_prompt.replace("[Debate]",    json.dumps(archive[3]))
        developer_prompt = developer_prompt.replace("[Websearch]", json.dumps(archive[4]))
    elif problem_type == "harmony_minimal":
        developer_prompt = developer_prompt.replace("[COT]",       json.dumps(archive[0]))
        developer_prompt = developer_prompt.replace("[COT_SC]",    json.dumps(archive[1]))
        developer_prompt = developer_prompt.replace("[Reflexion]", json.dumps(archive[2]))
        developer_prompt = developer_prompt.replace("[Debate]",    json.dumps(archive[3]))

    return system_prompt, user_prompt, developer_prompt


def _apply_chat_template(
    system_prompt: str,
    user_prompt: str,
    developer_prompt: str | None,
    model_path: str | None = None,
    tokenizer=None,
):
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    if developer_prompt is not None:
        messages.append({"role": "user", "content": developer_prompt})

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def make_mas_prompt(
    question: str,
    problem_type: str,
    node_model: str,
    dataset_name: str,
    init_archive="",
    tokenizer=None,
    model_path: str | None = None,
):
    system_prompt, user_prompt, developer_prompt = make_prompt_no_examples(
        question, problem_type, node_model, dataset_name, init_archive
    )
    return _apply_chat_template(system_prompt, user_prompt, developer_prompt, model_path, tokenizer)


def make_rm_prompt(
    question: str,
    problem_type: str,
    node_model: str,
    dataset_name: str,
    answer: str,
    init_archive="",
    tokenizer_qwen=None,
    tokenizer_llama=None,
    model_path_qwen: str | None = None,
    model_path_llama: str | None = None,
):
    from utils.prompts.init_archive import get_init_archive

    if problem_type == "harmony_medium":
        from utils.prompts.system_prompts.harmony_medium import system_prompt
        from utils.prompts.user_prompts.rm_harmony_medium import base
        from utils.prompts.developer_prompts.harmony_medium import developer_prompt
    elif problem_type == "harmony_minimal":
        from utils.prompts.system_prompts.harmony_minimal import system_prompt
        from utils.prompts.user_prompts.rm_harmony_minimal import base
        from utils.prompts.developer_prompts.harmony_minimal import developer_prompt
    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")

    archive = get_init_archive(init_archive if init_archive else problem_type)

    system_prompt = system_prompt.replace("[MODEL]", node_model)
    user_prompt = base.replace("[QUESTION]", question)

    if problem_type == "harmony_medium":
        developer_prompt = developer_prompt.replace("[COT]",       json.dumps(archive[0]))
        developer_prompt = developer_prompt.replace("[COT_SC]",    json.dumps(archive[1]))
        developer_prompt = developer_prompt.replace("[Reflexion]", json.dumps(archive[2]))
        developer_prompt = developer_prompt.replace("[Debate]",    json.dumps(archive[3]))
        developer_prompt = developer_prompt.replace("[Websearch]", json.dumps(archive[4]))
    elif problem_type == "harmony_minimal":
        developer_prompt = developer_prompt.replace("[COT]",       json.dumps(archive[0]))
        developer_prompt = developer_prompt.replace("[COT_SC]",    json.dumps(archive[1]))
        developer_prompt = developer_prompt.replace("[Reflexion]", json.dumps(archive[2]))
        developer_prompt = developer_prompt.replace("[Debate]",    json.dumps(archive[3]))

    if tokenizer_qwen is None:
        from transformers import AutoTokenizer
        tokenizer_qwen = AutoTokenizer.from_pretrained(model_path_qwen, trust_remote_code=True, use_fast=False)

    chat_template_qwen = tokenizer_qwen.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
            {"role": "user",   "content": developer_prompt},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )

    if tokenizer_llama is None:
        from transformers import AutoTokenizer
        tokenizer_llama = AutoTokenizer.from_pretrained(model_path_llama, trust_remote_code=True, use_fast=False)

    return tokenizer_llama.apply_chat_template(
        [
            {"role": "user",      "content": chat_template_qwen},
            {"role": "assistant", "content": answer},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )


def main():
    import argparse
    p = argparse.ArgumentParser(description="Test prompt generation and write output JSON.")
    p.add_argument("--question",         default="What are the key challenges in multi-agent coordination?")
    p.add_argument("--problem-type",     default="harmony_medium")
    p.add_argument("--node-model",       default="gpt-4o")
    p.add_argument("--dataset-name",     default="math")
    p.add_argument("--qwen-model-path",  required=True, help="HF path to Qwen tokenizer.")
    p.add_argument("--llama-model-path", required=True, help="HF path to Llama/Skywork tokenizer.")
    p.add_argument("--output",           default="prompts_output.json")
    args = p.parse_args()

    system_prompt, user_prompt, developer_prompt = make_prompt(
        args.question, args.problem_type, args.node_model, args.dataset_name
    )
    chat_template       = _apply_chat_template(system_prompt, user_prompt, developer_prompt, args.qwen_model_path)
    chat_template_llama = _apply_chat_template(system_prompt, user_prompt, developer_prompt, args.llama_model_path)

    out = {
        "system_prompt":       system_prompt,
        "user_prompt":         user_prompt,
        "developer_prompt":    developer_prompt,
        "chat_template":       chat_template,
        "chat_template_llama": chat_template_llama,
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Prompts saved to {args.output}")


if __name__ == "__main__":
    main()
