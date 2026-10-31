# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Independent goal-image attention for an Action Expert block."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class GoalCrossAttention(nn.Module):
    """Read pre-Gemma goal tokens and add a zero-initialized residual.

    Initialization uses a private CPU RNG stream, leaving common model weights
    and the caller's RNG unchanged. There is no learned norm, gate, or position
    embedding. Goal tokens remain attached to the visual encoder's graph.
    """

    def __init__(
        self,
        action_width: int,
        goal_width: int,
        *,
        num_heads: int = 8,
        num_kv_heads: int = 1,
        head_dim: int = 64,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if min(num_heads, num_kv_heads, head_dim) < 1 or num_heads % num_kv_heads:
            raise ValueError(
                "Goal attention requires positive dimensions and Hq % Hkv = 0"
            )
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.q_proj = nn.Linear(action_width, num_heads * head_dim, bias=False)
            self.k_proj = nn.Linear(goal_width, num_kv_heads * head_dim, bias=False)
            self.v_proj = nn.Linear(goal_width, num_kv_heads * head_dim, bias=False)
            self.o_proj = nn.Linear(num_heads * head_dim, action_width, bias=False)
            for projection in (self.q_proj, self.k_proj, self.v_proj):
                nn.init.normal_(
                    projection.weight, std=1 / math.sqrt(projection.in_features)
                )
            nn.init.zeros_(self.o_proj.weight)

    @staticmethod
    def _rms(x: torch.Tensor) -> torch.Tensor:
        value = x.float()
        return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)).to(
            x.dtype
        )

    def forward(
        self,
        hidden: torch.Tensor,
        goal_tokens: torch.Tensor | None,
        goal_valid: torch.Tensor | None,
    ) -> torch.Tensor:
        """Attend to valid goal patches; missing or all-invalid goals are identity."""
        if goal_tokens is None:
            return hidden
        if goal_valid is None or goal_valid.shape != goal_tokens.shape[:2]:
            raise ValueError("goal_valid must have shape [batch, goal_patches]")
        if goal_tokens.shape[1] == 0:
            return hidden
        valid = goal_valid.to(device=hidden.device, dtype=torch.bool)
        has_goal = valid.any(-1)
        # A safe zero key prevents all-masked softmax. Invalid image contents,
        # including NaNs in padding, must not enter the projections.
        memory = goal_tokens.to(device=hidden.device, dtype=hidden.dtype)
        memory = torch.where(valid[..., None], memory, torch.zeros_like(memory))
        memory = self._rms(memory)
        safe_valid = valid | (
            (~has_goal)[:, None]
            & (torch.arange(valid.shape[1], device=valid.device) == 0)
        )

        q = F.linear(self._rms(hidden), self.q_proj.weight.to(hidden.dtype))
        k = F.linear(memory, self.k_proj.weight.to(hidden.dtype))
        v = F.linear(memory, self.v_proj.weight.to(hidden.dtype))
        batch = hidden.shape[0]
        q = q.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # Explicit repetition also supports SDPA backends without enable_gqa.
        repeats = self.num_heads // self.num_kv_heads
        attended = F.scaled_dot_product_attention(
            q,
            k.repeat_interleave(repeats, dim=1),
            v.repeat_interleave(repeats, dim=1),
            attn_mask=safe_valid[:, None, None, :],
            dropout_p=0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(batch, hidden.shape[1], -1)
        delta = F.linear(attended, self.o_proj.weight.to(hidden.dtype))
        delta = torch.where(has_goal[:, None, None], delta, torch.zeros_like(delta))
        return hidden + delta
