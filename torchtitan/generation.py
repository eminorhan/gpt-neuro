# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import logging
from typing import Optional


def multinomial_sample_one(
    probs: torch.Tensor, 
    rng: Optional[torch.Generator] = None
) -> torch.Tensor:
    q = torch.empty_like(probs).exponential_(1, generator=rng)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.long)


def logits_to_probs(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
        pivot = v.select(dim=-1, index=-1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def generate_next_token(
    model,
    x: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    rng: Optional[torch.Generator] = None,
) -> torch.Tensor:
    logits = model(x)  # (B, T, vocab_size)
    logits[:, :, -1] = float('-inf')  # do not emit bos token
    probs = logits_to_probs(logits[:, -1, :], temperature, top_k)
    next_token = multinomial_sample_one(probs, rng=rng)
    return next_token


@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    n_neurons: int,
    bos_token: int,
    logger: logging.Logger,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    # ensure batch dimension (T,) --> (B, T)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)

    rng = None
    if seed is not None:
        rng = torch.Generator(input_ids.device).manual_seed(seed)

    generated_tokens = input_ids.clone()

    for i in range(max_new_tokens):

        if i % (n_neurons+1) == 0:
            next_token = torch.tensor(bos_token, dtype=torch.long, device=generated_tokens.device).unsqueeze(0).unsqueeze(0)
        else:
            next_token = generate_next_token(
                model,
                x=generated_tokens,
                temperature=temperature,
                top_k=top_k,
                rng=rng,
            )

        generated_tokens = torch.cat([generated_tokens, next_token], dim=1)
        if i % 1000 == 0:
            logger.info(f"Step {i} of {max_new_tokens}")

    return generated_tokens
