from __future__ import annotations

import copy

from utils.prompts.blocks_harmony.cot import COT
from utils.prompts.blocks_harmony.cot_sc import COT_SC
from utils.prompts.blocks_harmony.llm_debate import LLM_debate
from utils.prompts.blocks_harmony.reflexion import Reflexion
from utils.prompts.blocks_harmony.web_search import WebSearch


BLOCK_MAP = {
    "COT": COT,
    "COT_SC": COT_SC,
    "Reflexion": Reflexion,
    "LLM_debate": LLM_debate,
    "WebSearch": WebSearch,
}


DEFAULT_ARCHIVES = {
    "harmony_medium": ["COT", "COT_SC", "Reflexion", "LLM_debate", "WebSearch"],
    "harmony_minimal": ["COT", "COT_SC", "Reflexion", "LLM_debate"],
}


def get_init_archive(init_archive: str | list[str]) -> list[dict]:
    if isinstance(init_archive, list):
        blocks = init_archive
    elif isinstance(init_archive, str):
        if init_archive not in DEFAULT_ARCHIVES:
            raise ValueError(f"Unsupported init_archive: {init_archive}")
        blocks = DEFAULT_ARCHIVES[init_archive]
    else:
        raise TypeError(f"Unsupported init_archive type: {type(init_archive)}")

    missing = [name for name in blocks if name not in BLOCK_MAP]
    if missing:
        raise ValueError(f"Unknown block names in archive: {missing}")

    return [copy.deepcopy(BLOCK_MAP[name]) for name in blocks]
