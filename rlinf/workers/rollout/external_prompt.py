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

"""Pause evaluation at each action chunk for externally supplied text prompts."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class ExternalPromptController:
    """Write observable RGB/state requests and wait for a matching response.

    The controller changes only language, never actions, resets, task success,
    or simulator state. Every call requires a fresh response for every batch
    member. There is no oracle, automatic planner, or repeated-prompt fallback.
    """

    def __init__(self, directory: str | Path, rank: int, timeout_s: float = 1800):
        self.directory = Path(directory) / f"worker_{rank}"
        self.directory.mkdir(parents=True, exist_ok=False)
        self.timeout_s = float(timeout_s)
        if self.timeout_s <= 0:
            raise ValueError("External prompt timeout must be positive.")
        self.sequence = 0
        self.session = uuid.uuid4().hex

    def apply(self, observations: dict[str, Any]) -> dict[str, Any]:
        """Save the current batch, wait for fresh commands and replace its text."""
        images = _array(observations["main_images"])
        wrist = _array(observations["wrist_images"])
        tasks = list(observations["task_descriptions"])
        states = _array(observations["states"])
        batch = len(tasks)
        if (
            batch == 0
            or len(images) != batch
            or len(wrist) != batch
            or len(states) != batch
        ):
            raise ValueError(
                "External prompt observations must have matching batch sizes."
            )
        sequence = self.sequence
        request_id = f"{self.session}:{sequence}"
        stem = f"chunk_{sequence:06d}"
        board = Image.new("RGB", (448, batch * 256), "#152030")
        views = []
        for index in range(batch):
            paths = {}
            for column, name, values in [(0, "main", images), (1, "wrist", wrist)]:
                pixels = values[index]
                if pixels.shape[0] == 3:
                    pixels = np.moveaxis(pixels, 0, -1)
                if np.issubdtype(pixels.dtype, np.floating):
                    pixels = np.rint(pixels * 255)
                if pixels.ndim != 3 or pixels.shape[-1] != 3:
                    raise ValueError("External prompt views must be RGB HWC or CHW.")
                image = Image.fromarray(pixels.astype(np.uint8)).resize(
                    (224, 224), Image.Resampling.BILINEAR
                )
                path = self.directory / f"{stem}_env_{index}_{name}.png"
                image.save(path)
                board.paste(image, (column * 224, index * 256 + 32))
                paths[name] = str(path)
            ImageDraw.Draw(board).text(
                (4, index * 256 + 3),
                f"env {index} | chunk {sequence}\n{tasks[index][:72]}",
                fill="white",
            )
            views.append(paths)
        board_path = self.directory / f"{stem}_board.png"
        board.save(board_path)
        request = {
            "request_id": request_id,
            "chunk_index": sequence,
            "batch_size": batch,
            "task_descriptions": tasks,
            "states": states.tolist(),
            "views": views,
            "board": str(board_path),
            "pixel_coordinates": "main view, integer [x,y] in 0..255; origin top left",
        }
        request_path = self.directory / f"{stem}_request.json"
        temporary = request_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(request, indent=2) + "\n")
        temporary.replace(request_path)
        response_path = self.directory / f"{stem}_response.json"
        deadline = time.monotonic() + self.timeout_s
        while not response_path.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"No fresh external prompt response for {request_path}"
                )
            time.sleep(0.1)
        response = json.loads(response_path.read_text())
        if response.get("request_id") != request_id:
            raise ValueError(
                "External prompt response has a stale or mismatched request_id."
            )
        prompts = response.get("prompts")
        if (
            not isinstance(prompts, list)
            or len(prompts) != batch
            or any(not isinstance(text, str) or not text.strip() for text in prompts)
        ):
            raise ValueError("Supply one nonempty text prompt per environment.")
        self.sequence += 1
        result = dict(observations)
        result["task_descriptions"] = prompts
        return result
