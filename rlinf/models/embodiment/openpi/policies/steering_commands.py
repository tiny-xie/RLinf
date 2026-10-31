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

"""Sample text steering commands from dataset annotations before tokenization."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
from openpi import transforms

STEERING_STYLES = (
    "task",
    "subtask",
    "atomic_motion",
    "gripper_trace",
    "point",
    "combination",
)


@dataclasses.dataclass(frozen=True)
class SampleSteeringCommand(transforms.DataTransformFn):
    """Uniformly sample a candidate on every dataset access.

    ``steering_commands`` is a JSON string containing a list of objects with
    ``style`` and ``text``. This transform belongs to dataset-only repacking;
    inference continues to accept a caller-provided ``prompt``.
    """

    def __call__(self, data: dict) -> dict:
        if "steering_commands" not in data:
            raise ValueError("Steering SFT requires the steering_commands annotation.")
        candidates = json.loads(data["steering_commands"])
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("steering_commands must be a nonempty JSON list.")
        for command in candidates:
            if (
                not isinstance(command, dict)
                or command.get("style") not in STEERING_STYLES
                or not isinstance(command.get("text"), str)
                or not command["text"].strip()
            ):
                raise ValueError("Each steering command needs a valid style and text.")
        result = {
            key: value for key, value in data.items() if key != "steering_commands"
        }
        result["prompt"] = candidates[int(np.random.randint(len(candidates)))]["text"]
        return result
