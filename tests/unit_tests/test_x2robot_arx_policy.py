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

import numpy as np
import pytest

pytest.importorskip("einops")
pytest.importorskip("openpi")

from rlinf.models.embodiment.openpi.policies.arx_policy import ArxInputs  # noqa: E402


def _images():
    return {
        camera: np.zeros((8, 8, 3), dtype=np.uint8)
        for camera in ArxInputs.EXPECTED_CAMERAS
    }


def test_arx_rollout_disables_training_only_augmentation():
    state = np.arange(6 * 32, dtype=np.float32).reshape(6, 32)
    transform = ArxInputs(
        mode="sm2sm",
        state_history_size=3,
        state_future_size=2,
        random_drop_master=1.0,
        random_drop_history=1.0,
        random_drop_future=1.0,
        random_pos_offset=1.0,
    )

    result = transform({"state": state.copy(), "images": _images(), "prompt": "task"})

    assert result["state"].shape == (6, 32)
    np.testing.assert_array_equal(result["state"][:, 14:28], state[:, 14:28])


def test_arx_sft_augmentation_supports_32d_sequence_state():
    state = np.arange(6 * 32, dtype=np.float32).reshape(6, 32)
    transform = ArxInputs(
        mode="sm2sm",
        state_history_size=3,
        state_future_size=2,
        random_drop_master=1.0,
    )

    result = transform(
        {
            "state": state,
            "images": _images(),
            "prompt": "task",
            "actions": np.zeros((20, 28), dtype=np.float32),
        }
    )

    assert result["actions"].shape == (20, 32)
    expected_master = np.repeat(result["state"][3:4, :14], 6, axis=0)
    np.testing.assert_array_equal(result["state"][:, 14:28], expected_master)
