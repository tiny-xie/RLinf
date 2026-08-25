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

"""Per-frame LeRobot recorder for the X2Robot always-on upload stream."""

from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rlinf.data.storage.lerobot.compat import add_frame_to_dataset
from rlinf.data.storage.lerobot.writer import LeRobotDatasetWriter

logger = logging.getLogger(__name__)

IMAGE_NAMES = ("left_wrist_view", "face_view", "right_wrist_view")


class X2RobotLeRobotRecorder:
    """Stream synchronized 20 Hz observations into a LeRobot dataset.

    The upload socket enqueues compressed JPEG payloads. A background thread
    decodes and writes them so robot communication is not blocked by image or
    video I/O. Every source frame becomes one LeRobot frame and keeps its own
    ``intervene_flag``; labels are never expanded or reduced at policy-chunk
    granularity.
    """

    def __init__(
        self,
        data_path: str,
        *,
        task_description: str,
        image_height: int,
        image_width: int,
        fps: int = 20,
        queue_size: int = 4000,
    ) -> None:
        if queue_size <= 0:
            raise ValueError("LeRobot recorder queue_size must be positive")
        self.data_path = Path(data_path).expanduser()
        self.task_description = str(task_description)
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.fps = int(fps)
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._thread = threading.Thread(
            target=self._run,
            name="x2robot-lerobot-writer",
            daemon=True,
        )
        self._started = False
        self._error: BaseException | None = None
        self._writer: LeRobotDatasetWriter | None = None
        self._episode_frames = 0
        self._episode_takeover_frames = 0

    @property
    def error(self) -> BaseException | None:
        """Return the first background writer failure, if any."""

        return self._error

    def start(self) -> None:
        """Start the background writer."""

        if self._started:
            return
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        self._started = True
        self._thread.start()

    def append_step(self, header: dict[str, Any], blobs: list[bytes]) -> None:
        """Queue one synchronized header plus three JPEG payloads."""

        if len(blobs) != len(IMAGE_NAMES):
            raise ValueError(f"expected three X2Robot images, got {len(blobs)}")
        self._raise_if_failed()
        self._queue.put(("step", dict(header), tuple(bytes(blob) for blob in blobs)))

    def finish_episode(self, header: dict[str, Any]) -> None:
        """Queue an episode boundary after all preceding step records."""

        self._raise_if_failed()
        self._queue.put(("episode_end", dict(header), None))

    def close(self) -> None:
        """Flush queued frames and save an unfinished active episode."""

        if not self._started:
            return
        if self._error is not None:
            self._started = False
            self._raise_if_failed()
        self._queue.put(("stop", None, None))
        self._thread.join(timeout=60.0)
        self._started = False
        if self._thread.is_alive():
            raise RuntimeError("timed out while closing X2Robot LeRobot recorder")
        self._raise_if_failed()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("X2Robot LeRobot recorder failed") from self._error

    def _features(self) -> dict[str, dict[str, Any]]:
        image_feature = {
            "dtype": "image",
            "shape": (self.image_height, self.image_width, 3),
            "names": ["height", "width", "channel"],
        }
        return {
            "state": {
                "dtype": "float32",
                "shape": (32,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (28,),
                "names": ["actions"],
            },
            "intervene_flag": {
                "dtype": "bool",
                "shape": (1,),
                "names": ["intervene_flag"],
            },
            "mode": {
                "dtype": "int64",
                "shape": (1,),
                "names": ["mode"],
            },
            "done": {
                "dtype": "bool",
                "shape": (1,),
                "names": ["done"],
            },
            "is_success": {
                "dtype": "bool",
                "shape": (1,),
                "names": ["is_success"],
            },
            **{name: dict(image_feature) for name in IMAGE_NAMES},
        }

    def _ensure_writer(self) -> LeRobotDatasetWriter:
        if self._writer is not None:
            return self._writer
        writer = LeRobotDatasetWriter()
        writer.create(
            repo_id=str(self.data_path),
            robot_type="x2robot",
            fps=self.fps,
            features=self._features(),
            image_writer_threads=4,
            image_writer_processes=0,
        )
        self._writer = writer
        return writer

    def _run(self) -> None:
        try:
            while True:
                kind, header, blobs = self._queue.get()
                if kind == "step":
                    self._write_step(header, blobs)
                elif kind == "episode_end":
                    self._save_episode(header, incomplete=False)
                elif kind == "stop":
                    self._save_episode(
                        {"type": "shutdown", "success": None}, incomplete=True
                    )
                    if self._writer is not None:
                        self._writer.finalize()
                    return
                else:
                    raise ValueError(f"unknown LeRobot recorder item {kind!r}")
        except BaseException as exc:  # noqa: BLE001 - surfaced via public API
            self._error = exc
            logger.exception("X2Robot LeRobot writer failed")

    def _decode_image(self, blob: bytes) -> np.ndarray:
        if blob:
            image = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
        else:
            image = None
        if image is None:
            return np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape[:2] != (self.image_height, self.image_width):
            image = cv2.resize(
                image,
                (self.image_width, self.image_height),
                interpolation=cv2.INTER_AREA,
            )
        return np.ascontiguousarray(image, dtype=np.uint8)

    @staticmethod
    def _observation_state(header: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        slave = np.asarray(
            header["follow1_pos"] + header["follow2_pos"], dtype=np.float32
        )
        master = np.asarray(
            header["master1_pos"] + header["master2_pos"], dtype=np.float32
        )
        action = np.concatenate([slave, master])
        if action.shape != (28,):
            raise ValueError(
                f"X2Robot upload state must be slave14 + master14, got {action.shape}"
            )
        state = np.zeros((32,), dtype=np.float32)
        state[:28] = action
        return state, action

    def _write_step(self, header: dict[str, Any], blobs: tuple[bytes, ...]) -> None:
        writer = self._ensure_writer()
        state, action = self._observation_state(header)
        is_takeover = bool(header.get("is_takeover", False))
        mode = int(header.get("running_mode", header.get("mode", 1)))
        frame: dict[str, Any] = {
            "state": state,
            "actions": action,
            "intervene_flag": np.array([is_takeover], dtype=bool),
            "mode": np.array([mode], dtype=np.int64),
            "done": np.array([False], dtype=bool),
            "is_success": np.array([False], dtype=bool),
            "task": self.task_description,
        }
        for name, blob in zip(IMAGE_NAMES, blobs):
            frame[name] = self._decode_image(blob)
        add_frame_to_dataset(writer.dataset, frame)
        self._episode_frames += 1
        self._episode_takeover_frames += int(is_takeover)

    def _save_episode(self, header: dict[str, Any], *, incomplete: bool) -> None:
        if self._writer is None or self._episode_frames == 0:
            return
        dataset = self._writer.dataset
        episode_buffer = dataset.episode_buffer
        success = bool(header.get("success", False))
        for index in range(self._episode_frames):
            episode_buffer["is_success"][index] = np.array([success], dtype=bool)
        episode_buffer["done"][-1] = np.array([True], dtype=bool)
        dataset.save_episode()
        logger.info(
            "saved X2Robot LeRobot episode: path=%s frames=%d "
            "takeover_frames=%d incomplete=%s",
            self.data_path,
            self._episode_frames,
            self._episode_takeover_frames,
            incomplete,
        )
        self._episode_frames = 0
        self._episode_takeover_frames = 0
