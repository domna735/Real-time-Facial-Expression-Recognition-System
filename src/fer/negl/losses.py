from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def complementary_negative_loss(
    logits: torch.Tensor,
    negative_y: torch.Tensor,
    *,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Negative learning loss (complementary labels) scaffold.

    Given a sampled wrong class `negative_y` for each sample, penalize placing probability mass on it.

    This is a minimal starting point; later we can implement:
    - teacher-guided negative sampling via confusion matrix
    - uncertainty gating
    - class-aware ratios to protect minority recall
    """

    negative_y = negative_y.long()
    log_probs = F.log_softmax(logits, dim=1)
    nll_neg = -log_probs.gather(1, negative_y.view(-1, 1)).squeeze(1)
    if weight is not None:
        w = weight.to(nll_neg.dtype)
        nll_neg = nll_neg * w
        denom = w.sum().clamp_min(1e-12)
        return nll_neg.sum() / denom
    return nll_neg.mean()
