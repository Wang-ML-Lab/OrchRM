"""
Async-safe trace collection for MAS subagent calls.

Usage:
    - Call `init_trace()` before executing a forward() pass
    - Blocks/samplers call `log_subagent_call()` to record each LLM call
    - Call `get_trace()` after execution to retrieve the full trace
"""

import time
import contextvars
from typing import Any

# ContextVar is async-safe: each asyncio Task gets its own copy
_trace_var: contextvars.ContextVar[list] = contextvars.ContextVar("mas_trace", default=None)
_token_var: contextvars.ContextVar[dict] = contextvars.ContextVar("mas_tokens", default=None)


def init_trace():
    """Initialize a fresh trace for the current async context."""
    _trace_var.set([])
    _token_var.set({"prompt": 0, "completion": 0, "total": 0})


def log_subagent_call(
    name: str,
    model: str,
    input_prompt: str,
    output: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    execution_time: float = 0.0,
    extra: dict[str, Any] | None = None,
):
    """Append a subagent call record to the current trace."""
    trace = _trace_var.get()
    tokens = _token_var.get()
    if trace is None:
        return  # trace not initialized, skip silently

    record = {
        "name": name,
        "model": model,
        "input": input_prompt[:2000] if input_prompt else "",  # truncate for storage
        "output": output[:2000] if output else "",
        "token_used": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": prompt_tokens + completion_tokens,
        },
        "execution_time": round(execution_time, 3),
    }
    if extra:
        record.update(extra)
    trace.append(record)

    # Accumulate total tokens
    tokens["prompt"] += prompt_tokens
    tokens["completion"] += completion_tokens
    tokens["total"] += prompt_tokens + completion_tokens


def get_trace() -> tuple[list, dict]:
    """Return (subagent_trace, total_token_used) for the current context."""
    trace = _trace_var.get()
    tokens = _token_var.get()
    return (trace or []), (tokens or {"prompt": 0, "completion": 0, "total": 0})
