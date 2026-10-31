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

"""Load external LIBERO task definitions without changing installed suites."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from libero.libero.benchmark import Benchmark, Task
from libero.libero.envs.bddl_utils import get_problem_info


class CustomTaskBenchmark(Benchmark):
    """Expose manifest tasks through LIBERO's benchmark interface.

    Paths are relative to the manifest. Initial states are trusted local torch
    files, loaded once per distinct path on the CPU. BDDL defines success.
    """

    def __init__(self, manifest_path: str | Path) -> None:
        super().__init__(task_order_index=0)
        path = Path(manifest_path).expanduser().resolve(strict=True)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        entries = manifest.get("tasks", [])
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Custom task manifest must contain tasks: {path}")
        if manifest.get("order_required", False) is not False:
            raise ValueError("Custom LIBERO tasks do not support ordered goals.")
        self.name = "libero_custom"
        self.tasks: list[Task] = []
        self._states_by_path: dict[Path, Any] = {}
        self._task_state_paths: list[Path] = []
        self._task_subgoals: list[list[dict[str, Any]]] = []
        names: set[str] = set()
        for entry in entries:
            name = entry["key"]
            if not isinstance(name, str) or not name or name in names:
                raise ValueError(f"Custom task keys must be unique strings: {name!r}")
            names.add(name)
            if entry.get("order_required", False) is not False:
                raise ValueError(f"Ordered custom task is unsupported: {name}")
            bddl = (path.parent / entry["bddl_file"]).resolve(strict=True)
            states_path = (path.parent / entry["initial_states"]).resolve(strict=True)
            problem = get_problem_info(str(bddl))
            language = problem["language_instruction"]
            if entry["prompt"] != language:
                raise ValueError(f"Manifest prompt differs from BDDL language: {name}")
            if states_path not in self._states_by_path:
                states = torch.load(states_path, map_location="cpu", weights_only=False)
                if not hasattr(states, "__len__") or len(states) == 0:
                    raise ValueError(f"Empty custom initial-state pool: {states_path}")
                self._states_by_path[states_path] = states
            self._task_state_paths.append(states_path)
            self._task_subgoals.append(entry.get("subgoals", []))
            self.tasks.append(
                Task(
                    name=name,
                    language=language,
                    problem=problem["problem_name"],
                    problem_folder="",
                    bddl_file=str(bddl),
                    init_states_file=str(states_path),
                )
            )
        self.n_tasks = len(self.tasks)
        if manifest.get("task_count", self.n_tasks) != self.n_tasks:
            raise ValueError("Manifest task_count differs from the number of tasks.")

    def get_task_bddl_file_path(self, i: int) -> str:
        """Return the resolved BDDL path for a manifest-local task index."""
        return self.tasks[i].bddl_file

    def get_task_init_states(self, i: int) -> Any:
        """Return the cached initial-state pool for a manifest-local task index."""
        return self._states_by_path[self._task_state_paths[i]]

    def get_task_subgoals(self, i: int) -> list[dict[str, Any]]:
        """Return atomic prompts and predicates in manifest execution order."""
        return self._task_subgoals[i]
