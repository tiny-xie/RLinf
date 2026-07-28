# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from rlinf.models.embodiment.mlp_policy import get_model
from rlinf.models.embodiment.mlp_policy.mlp_policy import MLPPolicy


def _linear_output_dims(module: nn.Module) -> list[int]:
    return [
        layer.out_features for layer in module.modules() if isinstance(layer, nn.Linear)
    ]


def test_rlt_mlp_hidden_dims_are_loaded_from_config():
    cfg = OmegaConf.create(
        {
            "model_type": "rlt_mlp_policy",
            "z_dim": 8,
            "proprio_dim": 3,
            "action_dim": 2,
            "num_action_chunks": 4,
            "ref_num_action_chunks": 6,
            "add_q_head": True,
            "actor_hidden_dims": [32, 64, 128],
            "critic_hidden_dims": [16, 24, 32],
        }
    )

    model = get_model(cfg)

    assert model.actor_hidden_dims == (32, 64, 128)
    assert _linear_output_dims(model.backbone) == [32, 64, 128]
    assert model.actor_mean.in_features == 128
    assert model.actor_logstd.in_features == 128
    assert model.critic_hidden_dims == (16, 24, 32)
    for q_head in model.q_head.qs:
        assert _linear_output_dims(q_head.net) == [16, 24, 32, 1]

    obs = {
        "z_rl": torch.randn(2, 8),
        "proprio": torch.randn(2, 3),
        "ref_chunk": torch.randn(2, 6, 2),
    }
    actions, logprobs, values = model.sac_forward(obs, deterministic=True)
    q_values = model.sac_q_forward(obs, actions)

    assert actions.shape == (2, 8)
    assert logprobs.shape == (2, 8)
    assert values is None
    assert q_values.shape == (2, 2)


@pytest.mark.parametrize(
    ("config_name", "hidden_dims"),
    [
        ("actor_hidden_dims", []),
        ("critic_hidden_dims", [256, 0, 256]),
    ],
)
def test_mlp_hidden_dims_must_be_nonempty_and_positive(config_name, hidden_dims):
    kwargs = {
        "actor_hidden_dims": [256],
        "critic_hidden_dims": [256],
        config_name: hidden_dims,
    }

    with pytest.raises(ValueError, match=config_name):
        MLPPolicy(
            obs_dim=4,
            action_dim=2,
            num_action_chunks=1,
            add_value_head=False,
            add_q_head=False,
            **kwargs,
        )
