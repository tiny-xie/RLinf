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

"""Independent per-frame action tracing for real-world RLT rollouts."""

from __future__ import annotations

import atexit
import json
import os
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch


class RLTActionTraceRecorder:
    """Write VLA, Stage2, and optional human actions to a separate JSONL file.

    This recorder is deliberately independent of replay ingestion and LeRobot
    episode collection. Each JSONL row represents one environment frame and
    also retains the chunk's ``z_rl`` and Stage2 proprioceptive state. Both
    policy candidates are retained even when reference control is active;
    ``actor_active`` identifies which policy was routed. The human action is
    ``null`` for frames without intervention.

    Args:
        save_dir: Root directory for the standalone action traces.
        rank: Environment worker rank.
        stage_id: Pipeline stage owned by the environment worker.
        num_envs: Number of environments represented by each action batch.
        action_dim: Per-frame action dimension.
        resume: Append to an existing trace file instead of starting a new one.
    """

    def __init__(
        self,
        *,
        save_dir: str,
        rank: int,
        stage_id: int,
        num_envs: int,
        action_dim: int,
        resume: bool = False,
    ) -> None:
        self.num_envs = int(num_envs)
        self.action_dim = int(action_dim)
        if self.num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {self.num_envs}.")
        if self.action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {self.action_dim}.")

        trace_dir = Path(save_dir) / f"rank_{rank}" / f"stage_{stage_id}"
        trace_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = trace_dir / "actions.jsonl"
        self.trace_path.open("a" if resume else "w", encoding="utf-8").close()

        self._chunk_index = 0
        self._frame_indices = [0] * self.num_envs
        self._episode_indices = [0] * self.num_envs
        self._executor: ThreadPoolExecutor | None = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"rlt_action_trace_rank_{rank}_stage_{stage_id}",
        )
        self._futures: list[Future] = []
        self._closed = False
        atexit.register(self.close)

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray | None:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _reshape_actions(
        self,
        name: str,
        value: Any,
        *,
        min_chunk_size: int,
        required: bool,
    ) -> np.ndarray | None:
        array = self._to_numpy(value)
        if array is None:
            if required:
                raise ValueError(f"{name} cannot be None.")
            return None
        if array.ndim == 0 or array.shape[0] != self.num_envs:
            raise ValueError(
                f"{name} must have num_envs={self.num_envs} as its leading "
                f"dimension, got shape={array.shape}."
            )
        per_env_size = int(array[0].size)
        if per_env_size % self.action_dim != 0:
            raise ValueError(
                f"{name} cannot be split into {self.action_dim}D frame actions: "
                f"shape={array.shape}."
            )
        reshaped = array.reshape(self.num_envs, -1, self.action_dim)
        if reshaped.shape[1] < min_chunk_size:
            raise ValueError(
                f"{name} has only {reshaped.shape[1]} frames, expected at least "
                f"{min_chunk_size}."
            )
        return reshaped[:, :min_chunk_size].astype(np.float32, copy=False)

    def _reshape_flags(
        self,
        name: str,
        value: Any,
        *,
        chunk_size: int | None = None,
        default: bool = False,
    ) -> np.ndarray:
        array = self._to_numpy(value)
        if array is None:
            width = 1 if chunk_size is None else chunk_size
            return np.full((self.num_envs, width), default, dtype=bool)
        if array.ndim == 0:
            array = np.full((self.num_envs, 1), bool(array), dtype=bool)
        elif array.shape[0] != self.num_envs:
            raise ValueError(
                f"{name} must have num_envs={self.num_envs} as its leading "
                f"dimension, got shape={array.shape}."
            )
        array = np.asarray(array, dtype=bool).reshape(self.num_envs, -1)
        if chunk_size is None:
            return array
        if array.shape[1] < chunk_size:
            raise ValueError(
                f"{name} has only {array.shape[1]} frames, expected at least "
                f"{chunk_size}."
            )
        return array[:, :chunk_size]

    def _reshape_feature(self, name: str, value: Any) -> np.ndarray:
        array = self._to_numpy(value)
        if array is None:
            raise ValueError(f"{name} cannot be None.")
        if array.ndim == 0 or array.shape[0] != self.num_envs:
            raise ValueError(
                f"{name} must have num_envs={self.num_envs} as its leading "
                f"dimension, got shape={array.shape}."
            )
        return array.reshape(self.num_envs, -1).astype(np.float32, copy=False)

    def record_chunk(
        self,
        *,
        z_rl: Any,
        stage2_state: Any,
        vla_actions: Any,
        small_model_actions: Any,
        actor_switch: Any,
        human_actions: Any,
        human_flags: Any,
        terminations: Any,
        truncations: Any,
    ) -> None:
        """Queue one routed action chunk for standalone per-frame export."""
        if self._closed:
            raise RuntimeError("Cannot record actions after the recorder is closed.")

        z_rl_features = self._reshape_feature("z_rl", z_rl)
        stage2_states = self._reshape_feature("stage2_state", stage2_state)
        terminal_flags = self._reshape_flags(
            "terminations", terminations, default=False
        )
        chunk_size = int(terminal_flags.shape[1])
        truncated_flags = self._reshape_flags(
            "truncations", truncations, chunk_size=chunk_size, default=False
        )
        vla_chunk = self._reshape_actions(
            "vla_actions",
            vla_actions,
            min_chunk_size=chunk_size,
            required=True,
        )
        small_model_chunk = self._reshape_actions(
            "small_model_actions",
            small_model_actions,
            min_chunk_size=chunk_size,
            required=True,
        )
        human_chunk = self._reshape_actions(
            "human_actions",
            human_actions,
            min_chunk_size=chunk_size,
            required=False,
        )
        intervention_flags = self._reshape_flags(
            "human_flags", human_flags, chunk_size=chunk_size, default=False
        )
        actor_flags = self._reshape_flags(
            "actor_switch", actor_switch, default=False
        ).all(axis=1)

        if human_chunk is None and intervention_flags.any():
            raise ValueError(
                "human_flags marks an intervention, but human_actions is None."
            )

        records: list[dict[str, Any]] = []
        for env_idx in range(self.num_envs):
            for step_idx in range(chunk_size):
                intervened = bool(intervention_flags[env_idx, step_idx])
                done = bool(
                    terminal_flags[env_idx, step_idx]
                    or truncated_flags[env_idx, step_idx]
                )
                records.append(
                    {
                        "chunk_index": self._chunk_index,
                        "frame_index": self._frame_indices[env_idx],
                        "frame_in_chunk": step_idx,
                        "episode_index": self._episode_indices[env_idx],
                        "env_index": env_idx,
                        "actor_active": bool(actor_flags[env_idx]),
                        "intervene_flag": intervened,
                        "terminated": bool(terminal_flags[env_idx, step_idx]),
                        "truncated": bool(truncated_flags[env_idx, step_idx]),
                        "z_rl": z_rl_features[env_idx].tolist(),
                        "stage2_state": stage2_states[env_idx].tolist(),
                        "vla_action": vla_chunk[env_idx, step_idx].tolist(),
                        "small_model_action": small_model_chunk[
                            env_idx, step_idx
                        ].tolist(),
                        "human_action": (
                            human_chunk[env_idx, step_idx].tolist()
                            if intervened and human_chunk is not None
                            else None
                        ),
                    }
                )
                self._frame_indices[env_idx] += 1
                if done:
                    self._episode_indices[env_idx] += 1
                    break

        self._chunk_index += 1
        if records:
            payload = "".join(
                json.dumps(record, ensure_ascii=False) + os.linesep
                for record in records
            )
            assert self._executor is not None
            self._futures.append(self._executor.submit(self._append_payload, payload))
            self._drain_futures()

    def _append_payload(self, payload: str) -> None:
        with self.trace_path.open("a", encoding="utf-8") as trace_file:
            trace_file.write(payload)
            trace_file.flush()

    def _drain_futures(self) -> None:
        remaining = []
        for future in self._futures:
            if future.done():
                future.result()
            else:
                remaining.append(future)
        self._futures = remaining

    def close(self) -> None:
        """Flush pending action traces and stop the background writer."""
        if self._closed:
            return
        self._closed = True
        for future in self._futures:
            future.result()
        self._futures = []
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
