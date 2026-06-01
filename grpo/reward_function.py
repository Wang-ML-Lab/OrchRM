from __future__ import annotations

import asyncio
import os
import random
import re
from typing import Any

import requests


# ---------------------------------------------------------------------------
# Helpers shared by both reward paths
# ---------------------------------------------------------------------------

def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    return x if isinstance(x, str) else str(x)


def _build_messages(
    response: str,
    rm_prompt: Any,
) -> list[dict[str, str]]:
    """Build RM messages from rm_prompt + current response for message-based prompts."""
    response = _safe_str(response).strip()

    if isinstance(rm_prompt, list):
        messages: list[dict[str, str]] = []
        has_placeholder = False
        for m in rm_prompt:
            if not isinstance(m, dict):
                continue
            role = _safe_str(m.get("role", "user")) or "user"
            content = _safe_str(m.get("content", ""))
            if "{{response}}" in content:
                content = content.replace("{{response}}", response)
                has_placeholder = True
            messages.append({"role": role, "content": content})
        if not has_placeholder:
            messages.append({"role": "assistant", "content": response})
        return messages

    if isinstance(rm_prompt, str) and rm_prompt.strip():
        return [
            {"role": "user", "content": rm_prompt.strip()},
            {"role": "assistant", "content": response},
        ]

    raise ValueError("extra_info.rm_prompt is required and must be list[dict] or str")


def _build_rm_input(
    response: str,
    rm_prompt: Any,
    reward_model_tokenizer: Any,
) -> str:
    """Mirror RM inference: text prompts are concatenated directly with the response."""
    response = _safe_str(response).strip()

    if isinstance(rm_prompt, str) and rm_prompt.strip():
        prompt = rm_prompt.rstrip()
        return f"{prompt}\n\n{response}" if response else prompt

    messages = _build_messages(response=response, rm_prompt=rm_prompt)
    return _render_prompt(messages, reward_model_tokenizer)


def _render_prompt(messages: list[dict[str, str]], reward_model_tokenizer: Any) -> str:
    if reward_model_tokenizer is not None and hasattr(reward_model_tokenizer, "apply_chat_template"):
        try:
            text = reward_model_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            if isinstance(text, str) and text:
                return text
        except Exception:
            pass
    return "\n\n".join([m.get("content", "") for m in messages]).strip()


def _post_json_sync(url: str, payload: dict[str, Any], timeout_s: float = 120.0) -> dict[str, Any]:
    resp = requests.post(url, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def _extract_vllm_classify_score(output: dict[str, Any]) -> float:
    data = output.get("data", [])
    if not data:
        raise ValueError("classify response missing 'data'")
    probs = data[-1].get("probs", [])
    if not probs:
        raise ValueError("classify response missing 'probs'")
    return float(probs[-1])


def _get_llm_judge_concurrency() -> int:
    raw = os.environ.get("LLM_JUDGE_CONCURRENCY", "32").strip()
    try:
        value = int(raw)
    except ValueError:
        return 32
    return value if value > 0 else 1


def _get_llm_judge_max_retries() -> int:
    raw = os.environ.get("LLM_JUDGE_MAX_RETRIES", "8").strip()
    try:
        value = int(raw)
    except ValueError:
        return 8
    return max(1, value)


def _get_llm_judge_retry_base_seconds() -> float:
    raw = os.environ.get("LLM_JUDGE_RETRY_BASE_SECONDS", "1.0").strip()
    try:
        value = float(raw)
    except ValueError:
        return 1.0
    return max(0.1, value)


def _get_llm_judge_max_output_tokens() -> int:
    raw = os.environ.get("LLM_JUDGE_MAX_OUTPUT_TOKENS", "256").strip()
    try:
        value = int(raw)
    except ValueError:
        return 256
    return max(32, value)


# ---------------------------------------------------------------------------
# LLM judge path (OpenAI API — async SDK call)
# ---------------------------------------------------------------------------

def _build_judge_context(rm_prompt: Any) -> str:
    """Extract the task/context string from rm_prompt for the judge."""
    if isinstance(rm_prompt, list):
        parts = [
            _safe_str(m.get("content", ""))
            for m in rm_prompt
            if isinstance(m, dict) and m.get("role") in ("user", "system")
        ]
        return "\n".join(parts).strip()
    if isinstance(rm_prompt, str):
        return rm_prompt.strip()
    return ""


_JUDGE_SYSTEM = (
    "You are an expert evaluator of AI planning outputs. "
    "Your job is to score the quality of a plan produced by an AI agent."
)

_JUDGE_TEMPLATE = """\
## Task / Context
{context}

## Agent Planning Output
{response}

## Evaluation Instructions
Score the planning output above on a scale from 0 to 10:

- 0–2  : Completely wrong, incoherent, or irrelevant
- 3–4  : Partially relevant but with major gaps or errors
- 5–6  : Adequate plan that covers the task but lacks depth or clarity
- 7–8  : Good plan — logical, clear, and mostly complete
- 9–10 : Excellent plan — thorough, well-structured, and fully actionable

Respond with ONLY a single integer (0–10). No explanation.

Score:"""


async def _call_llm_judge_with_client_async(
    client: Any,
    response: str,
    rm_prompt: Any,
    model: str,
    max_retries: int | None = None,
) -> tuple[float, dict[str, int]]:
    context = _build_judge_context(rm_prompt)
    user_content = _JUDGE_TEMPLATE.format(context=context, response=response)

    api_kwargs: dict[str, Any] = {
        "model": model,
        "instructions": _JUDGE_SYSTEM,
        "input": user_content,
        "max_output_tokens": _get_llm_judge_max_output_tokens(),
        "store": False,
    }
    if model.startswith("gpt-5"):
        api_kwargs["reasoning"] = {"effort": "minimal"}

    last_error: Exception | None = None
    max_retries = max_retries or _get_llm_judge_max_retries()
    retry_base_s = _get_llm_judge_retry_base_seconds()
    for attempt in range(1, max_retries + 1):
        try:
            result = await client.responses.create(**api_kwargs)

            raw = (result.output_text or "").strip()
            match = re.search(r"\b(\d+(?:\.\d+)?)\b", raw)
            if not match:
                raise ValueError(f"LLM judge returned unparseable response: {raw!r}")
            score = max(0.0, min(10.0, float(match.group(1))))

            usage = getattr(result, "usage", None)
            token_usage = {
                "input_tokens": getattr(usage, "input_tokens", 0) if usage else 0,
                "output_tokens": getattr(usage, "output_tokens", 0) if usage else 0,
                "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
            }
            return score, token_usage

        except Exception as e:
            last_error = e
            if attempt < max_retries:
                delay_s = min(30.0, retry_base_s * (2 ** (attempt - 1)))
                jitter_s = random.uniform(0.0, min(1.0, delay_s * 0.25))
                await asyncio.sleep(delay_s + jitter_s)

    raise last_error


async def _call_llm_judge_async(
    response: str,
    rm_prompt: Any,
    model: str,
    api_key: str,
    timeout_s: float = 60.0,
    max_retries: int | None = None,
) -> tuple[float, dict[str, int]]:
    """Call the OpenAI Responses API as a judge (async) with retries.

    Returns:
        (score, token_usage) where score is 0–10 and token_usage has
        input_tokens, output_tokens, total_tokens.
    """
    from openai import AsyncOpenAI

    async with AsyncOpenAI(api_key=api_key, timeout=timeout_s) as client:
        return await _call_llm_judge_with_client_async(
            client=client,
            response=response,
            rm_prompt=rm_prompt,
            model=model,
            max_retries=max_retries,
        )


async def _score_llm_judge_batch_async(
    samples: list[dict[str, Any]],
    model: str,
    api_key: str,
    concurrency: int,
    timeout_s: float = 60.0,
) -> list[dict[str, Any]]:
    from openai import AsyncOpenAI

    semaphore = asyncio.Semaphore(concurrency)

    async def score_one(sample: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            try:
                raw_score, token_usage = await _call_llm_judge_with_client_async(
                    client=client,
                    response=sample["response"],
                    rm_prompt=sample["rm_prompt"],
                    model=model,
                    max_retries=_get_llm_judge_max_retries(),
                )
                return {
                    "score": raw_score / 10.0,
                    "llm_judge_score": raw_score,
                    "used_llm_judge": True,
                    "llm_judge_model": model,
                    "llm_judge_input_tokens": token_usage["input_tokens"],
                    "llm_judge_output_tokens": token_usage["output_tokens"],
                    "llm_judge_total_tokens": token_usage["total_tokens"],
                    "data_source": sample["data_source"],
                }
            except Exception as e:
                fallback = 1.0 if sample["ground_truth"] and sample["response"] == sample["ground_truth"] else 0.0
                return {
                    "score": fallback,
                    "llm_judge_score": None,
                    "used_llm_judge": False,
                    "llm_judge_error": str(e),
                    "data_source": sample["data_source"],
                }

    async with AsyncOpenAI(api_key=api_key, timeout=timeout_s) as client:
        return await asyncio.gather(*(score_one(sample) for sample in samples))


# ---------------------------------------------------------------------------
# Main reward entry point (sync — compatible with NaiveRewardManager)
# ---------------------------------------------------------------------------

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    reward_router_address: str | None = None,
    reward_model_tokenizer: Any = None,
    **kwargs,
) -> dict[str, Any]:
    """Reward function for verl GRPO training.

    Priority:
      1. LLM judge  — when OPENAI_API_KEY is set in the environment
      2. RM server  — when reward_router_address is provided (vLLM classify path)
      3. Exact match fallback
    """
    rm_prompt = (extra_info or {}).get("rm_prompt")
    response = _safe_str(solution_str).strip()
    gt = _safe_str(ground_truth).strip()

    # ------------------------------------------------------------------
    # 1. LLM judge path
    # ------------------------------------------------------------------
    openai_api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    llm_judge_model = os.environ.get("LLM_JUDGE_MODEL", "gpt-5-mini")

    if openai_api_key:
        try:
            raw_score, token_usage = asyncio.run(
                _call_llm_judge_async(
                    response=response,
                    rm_prompt=rm_prompt,
                    model=llm_judge_model,
                    api_key=openai_api_key,
                )
            )
            return {
                "score": raw_score / 10.0,                        # normalised to [0, 1] for verl
                "llm_judge_score": raw_score,                      # raw 0–10 for logging
                "used_llm_judge": True,
                "llm_judge_model": llm_judge_model,
                "llm_judge_input_tokens": token_usage["input_tokens"],
                "llm_judge_output_tokens": token_usage["output_tokens"],
                "llm_judge_total_tokens": token_usage["total_tokens"],
                "data_source": data_source,
            }
        except Exception as e:
            fallback = 1.0 if gt and response == gt else 0.0
            return {
                "score": fallback,
                "llm_judge_score": None,
                "used_llm_judge": False,
                "llm_judge_error": str(e),
                "data_source": data_source,
            }

    # ------------------------------------------------------------------
    # 2. RM server path (vLLM classify)
    # ------------------------------------------------------------------
    if reward_router_address:
        try:
            rm_input = _build_rm_input(
                response=response,
                rm_prompt=rm_prompt,
                reward_model_tokenizer=reward_model_tokenizer,
            )

            model_name = None
            if reward_model_tokenizer is not None:
                model_name = getattr(reward_model_tokenizer, "name_or_path", None)

            payload: dict[str, Any] = {"input": rm_input, "use_activation": False}
            if model_name:
                payload["model"] = model_name

            output = _post_json_sync(f"http://{reward_router_address}/classify", payload)
            rm_score = _extract_vllm_classify_score(output)
            return {
                "score": rm_score,
                "rm_score": rm_score,
                "used_rm_server": True,
                "data_source": data_source,
            }
        except Exception as e:
            fallback = 1.0 if gt and response == gt else 0.0
            return {
                "score": fallback,
                "rm_score": None,
                "used_rm_server": False,
                "rm_error": str(e),
                "data_source": data_source,
            }

    # ------------------------------------------------------------------
    # 3. Exact-match fallback
    # ------------------------------------------------------------------
    fallback = 1.0 if gt and response == gt else 0.0
    return {
        "score": fallback,
        "rm_score": None,
        "used_rm_server": False,
        "data_source": data_source,
    }


class OpenAIJudgeRewardManager:
    """Reward manager that scores a whole rollout batch with concurrent OpenAI calls."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or globals()["compute_score"]
        self.reward_fn_key = reward_fn_key

    def __call__(self, data, return_dict: bool = False):
        from collections import defaultdict

        import torch

        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        openai_api_key = os.environ.get("OPENAI_API_KEY", "").strip()

        samples: list[dict[str, Any]] = []
        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
            extra_info["num_turns"] = num_turns
            extra_info["rollout_reward_scores"] = rollout_reward_scores

            samples.append(
                {
                    "index": i,
                    "valid_response_length": int(valid_response_length),
                    "response": _safe_str(response_str).strip(),
                    "ground_truth": _safe_str(ground_truth).strip(),
                    "data_source": data_source,
                    "rm_prompt": extra_info.get("rm_prompt"),
                    "extra_info": extra_info,
                }
            )

        if openai_api_key:
            results = asyncio.run(
                _score_llm_judge_batch_async(
                    samples=samples,
                    model=os.environ.get("LLM_JUDGE_MODEL", "gpt-5-mini"),
                    api_key=openai_api_key,
                    concurrency=_get_llm_judge_concurrency(),
                )
            )
        else:
            results = [
                self.compute_score(
                    data_source=sample["data_source"],
                    solution_str=sample["response"],
                    ground_truth=sample["ground_truth"],
                    extra_info=sample["extra_info"],
                )
                for sample in samples
            ]

        already_print_data_sources = {}
        for sample, result in zip(samples, results, strict=True):
            reward = result["score"] if isinstance(result, dict) else result
            reward_tensor[sample["index"], sample["valid_response_length"] - 1] = reward

            if isinstance(result, dict):
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                reward_extra_info["acc"].append(reward)

            data_source = sample["data_source"]
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[response]", sample["response"])
                print("[ground_truth]", sample["ground_truth"])
                if isinstance(result, dict):
                    for key, value in result.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", reward)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        return reward_tensor

    def _extract_reward_from_rm_scores(self, data, return_dict: bool):
        if "rm_scores" not in data.batch.keys():
            return None
        if return_dict:
            reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
            reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
            return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
        return data.batch["rm_scores"]
