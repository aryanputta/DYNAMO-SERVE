"""
ShareGPT Trace Loader.

Loads real ChatML conversation traces from the ShareGPT dataset and converts
them into InferenceRequest objects for replay in the benchmark harness.

ShareGPT JSON format::

    [
      {
        "id": "...",
        "conversations": [
          {"from": "human",     "value": "..."},
          {"from": "gpt",       "value": "..."},
          {"from": "human",     "value": "..."},
          ...
        ]
      },
      ...
    ]

Each (human, gpt) exchange becomes one InferenceRequest where:
  - prompt_tokens ≈ len(human_turn.split()) * 1.35  (rough tokenisation)
  - output_tokens ≈ len(gpt_turn.split()) * 1.35
  - arrival_time  is synthesized from a Poisson process

Download::

    wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

Usage::

    from simulator.workload_replay.sharegpt_loader import ShareGPTLoader

    loader = ShareGPTLoader("ShareGPT_V3_unfiltered_cleaned_split.json")
    requests = loader.load(n=1000, arrival_rate_rps=10.0)
"""

from __future__ import annotations

import json
import logging
import math
import random
from pathlib import Path
from typing import Optional

from core.models import InferenceRequest, SLAClass, TenantPriority

logger = logging.getLogger(__name__)

# Approximate tokens per word (GPT-style BPE tokeniser)
TOKENS_PER_WORD = 1.35

# Max token caps (avoid OOM on very long conversations)
MAX_PROMPT_TOKENS  = 32_768
MAX_OUTPUT_TOKENS  = 4_096

# SLA distribution for chat workloads
CHAT_SLA_WEIGHTS = {
    SLAClass.REALTIME:    0.20,
    SLAClass.INTERACTIVE: 0.70,
    SLAClass.BATCH:       0.10,
}


def _word_count_to_tokens(text: str) -> int:
    words = len(text.split())
    return max(1, int(words * TOKENS_PER_WORD))


class ShareGPTLoader:
    """
    Loads ShareGPT conversations and emits InferenceRequest traces.

    Each human turn becomes the prompt; the corresponding GPT turn defines
    the expected output length (used for max_output_tokens).

    Multi-turn conversations produce one request per (human, gpt) pair,
    with the accumulated context carried forward as prefix_hash.
    """

    def __init__(
        self,
        path: str | Path,
        seed: int = 42,
    ) -> None:
        self._path = Path(path)
        self._rng  = random.Random(seed)
        self._data: list[dict] = []

        if self._path.exists():
            self._load()
        else:
            logger.warning(
                "ShareGPT file not found: %s. "
                "Using synthetic fallback. "
                "Download from HuggingFace to use real traces.",
                self._path,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        n: int = 500,
        arrival_rate_rps: float = 5.0,
        sla_weights: Optional[dict] = None,
        num_tenants: int = 10,
        multi_turn: bool = True,
    ) -> list[InferenceRequest]:
        """
        Generate *n* InferenceRequest objects from the loaded ShareGPT data.

        Args:
            n:                Number of requests to emit.
            arrival_rate_rps: Poisson mean arrival rate (requests / sec).
            sla_weights:      Dict mapping SLAClass → fraction (must sum to 1).
            num_tenants:      Number of synthetic tenants to assign.
            multi_turn:       If True, carry conversation prefix as prefix_hash.
        """
        sla_w = sla_weights or CHAT_SLA_WEIGHTS
        requests: list[InferenceRequest] = []
        t = 0.0

        source = self._data if self._data else None
        pairs = self._extract_pairs(source, n, multi_turn)

        for prompt_tokens, output_tokens, prefix_hash in pairs:
            iat = self._rng.expovariate(arrival_rate_rps)
            t  += iat

            sla_class  = self._sample_sla(sla_w)
            tenant_id  = f"tenant_{self._rng.randint(0, num_tenants - 1):02d}"
            priority   = {
                SLAClass.REALTIME:    TenantPriority.HIGH,
                SLAClass.INTERACTIVE: TenantPriority.NORMAL,
                SLAClass.BATCH:       TenantPriority.LOW,
            }[sla_class]

            req = InferenceRequest(
                prompt_tokens=min(prompt_tokens, MAX_PROMPT_TOKENS),
                max_output_tokens=min(output_tokens, MAX_OUTPUT_TOKENS),
                sla_class=sla_class,
                tenant_id=tenant_id,
                priority=priority,
                model_id="llama-3-70b",
                prefix_hash=prefix_hash,
            )
            req.arrival_time = t
            requests.append(req)

            if len(requests) >= n:
                break

        logger.info(
            "ShareGPTLoader: %d requests, %.0f-%.0f prompt tokens, "
            "%.1f%% long-context",
            len(requests),
            min(r.prompt_tokens for r in requests) if requests else 0,
            max(r.prompt_tokens for r in requests) if requests else 0,
            100 * sum(1 for r in requests if r.is_long_context) / max(1, len(requests)),
        )
        return requests

    def stats(self) -> dict:
        """Return dataset statistics."""
        return {
            "conversations": len(self._data),
            "source": str(self._path),
            "loaded": bool(self._data),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        logger.info("Loading ShareGPT traces from %s", self._path)
        with open(self._path, encoding="utf-8") as f:
            raw = json.load(f)
        self._data = raw if isinstance(raw, list) else raw.get("conversations", [])
        logger.info("Loaded %d conversations", len(self._data))

    def _extract_pairs(
        self,
        data: Optional[list[dict]],
        n: int,
        multi_turn: bool,
    ) -> list[tuple[int, int, Optional[str]]]:
        """
        Extract (prompt_tokens, output_tokens, prefix_hash) pairs.

        Falls back to synthetic log-normal pairs if data is not loaded.
        """
        if not data:
            return self._synthetic_pairs(n)

        pairs: list[tuple[int, int, Optional[str]]] = []
        for conv in self._rng.choices(data, k=max(n, len(data))):
            turns = conv.get("conversations", [])
            context_tokens = 0
            prefix_hash: Optional[str] = None

            for i in range(0, len(turns) - 1, 2):
                human = turns[i]
                gpt   = turns[i + 1] if i + 1 < len(turns) else None

                if human.get("from") not in ("human", "user"):
                    continue
                if gpt is None or gpt.get("from") not in ("gpt", "assistant"):
                    continue

                prompt_tok  = _word_count_to_tokens(human.get("value", ""))
                output_tok  = _word_count_to_tokens(gpt.get("value",  ""))
                total_prompt = context_tokens + prompt_tok

                if multi_turn and context_tokens > 0:
                    import hashlib
                    prefix_hash = hashlib.sha256(
                        f"{conv.get('id','')}_turn_{i}".encode()
                    ).hexdigest()[:16]

                pairs.append((total_prompt, output_tok, prefix_hash))
                context_tokens += prompt_tok + output_tok

                if not multi_turn:
                    break

            if len(pairs) >= n:
                break

        return pairs[:n]

    def _synthetic_pairs(self, n: int) -> list[tuple[int, int, Optional[str]]]:
        """Generate realistic chat-like token distributions synthetically."""
        pairs = []
        for _ in range(n):
            # Chat conversations: short prompts, medium outputs
            prompt = max(16, int(self._rng.lognormvariate(
                math.log(512), math.log(3)
            )))
            output = max(1, int(self._rng.lognormvariate(
                math.log(256), math.log(2)
            )))
            pairs.append((
                min(prompt, MAX_PROMPT_TOKENS),
                min(output, MAX_OUTPUT_TOKENS),
                None,
            ))
        return pairs

    def _sample_sla(self, weights: dict) -> SLAClass:
        roll = self._rng.random()
        cum  = 0.0
        for cls, w in weights.items():
            cum += w
            if roll < cum:
                return cls
        return SLAClass.INTERACTIVE
