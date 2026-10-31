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

"""Evaluate atomic subgoals with optional robot homing and visual targets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from libero.libero.envs.predicates import eval_predicate_fn


def validate_atomic_subgoals(
    subgoals: list[dict[str, Any]],
    max_steps_per_subgoal: int,
    native_goals: list,
    instruction_mode: str = "atomic",
) -> list[tuple[str, ...]]:
    """Validate the complete decomposition before spawning simulator workers."""
    if instruction_mode not in ("atomic", "full"):
        raise ValueError("atomic_sequence.instruction_mode must be 'atomic' or 'full'.")
    if (
        not isinstance(max_steps_per_subgoal, int)
        or isinstance(max_steps_per_subgoal, bool)
        or max_steps_per_subgoal < 1
    ):
        raise ValueError("atomic_sequence.max_steps_per_subgoal must be positive.")
    if not isinstance(subgoals, list) or not subgoals:
        raise ValueError("Atomic sequence evaluation requires manifest subgoals.")
    predicates = []
    for goal in subgoals:
        if not isinstance(goal, dict):
            raise ValueError("Atomic subgoals must be dictionaries.")
        predicate = goal.get("predicate", [])
        prompt = goal.get("prompt", "")
        if (
            not isinstance(prompt, str)
            or not prompt.strip()
            or not isinstance(predicate, list)
            or len(predicate) != 3
            or not all(isinstance(x, str) for x in predicate)
            or predicate[0].lower() != "on"
        ):
            raise ValueError("Atomic subgoals require a prompt and an On predicate.")
        predicates.append(tuple(x.lower() for x in predicate))
    if len(set(predicates)) != len(predicates) or sorted(predicates) != sorted(
        tuple(x) for x in native_goals
    ):
        raise ValueError("Atomic subgoals must match the complete BDDL goal.")
    return predicates


def load_goal_robot_poses(
    path: str, predicates: list[tuple[str, ...]]
) -> list[dict[str, float]]:
    """Load named robot-joint targets for each predicate, without fallback."""
    library = json.loads(Path(path).read_text())
    if library.get("version") != 1:
        raise ValueError("Goal robot pose library must have version 1.")
    entries = {}
    for entry in library["goals"]:
        key = tuple(part.lower() for part in entry["predicate"])
        if key in entries:
            raise ValueError(f"Duplicate goal robot pose: {key}.")
        pose = entry["joint_positions"]
        if not pose or not all(np.isfinite(float(value)) for value in pose.values()):
            raise ValueError(f"Invalid goal robot pose: {key}.")
        entries[key] = {name: float(value) for name, value in pose.items()}
    poses = []
    for predicate in predicates:
        key = tuple(part.lower() for part in predicate)
        if key not in entries:
            raise ValueError(f"No demonstration robot pose for visual goal {key}.")
        poses.append(entries[key])
    return poses


def render_subgoal_images(
    env: Any,
    predicates: list[tuple[str, ...]],
    robot_goal_poses: list[dict[str, float]] | None = None,
) -> list[np.ndarray]:
    """Render cumulative On goals, then restore the complete simulator state.

    By default targets retain the current robot pose and untouched objects.
    Explicit robot poses change only the arm/gripper during rendering. They
    may be reconstructed from demonstration terminal TCP states rather than
    original joint measurements; this does not guarantee pixel-identical goals.
    Each successive view retains earlier goals. No policy actions or physics
    steps are executed while constructing these oracle visual targets.
    """
    sim = env.sim
    state = env.get_sim_state().copy()
    ctrl = sim.data.ctrl.copy()
    warmstart = sim.data.qacc_warmstart.copy()
    images = []
    try:
        if robot_goal_poses is not None:
            if len(robot_goal_poses) != len(predicates) or len(env.robots) != 1:
                raise ValueError("Each visual subgoal needs one single-robot pose.")
            robot = env.robots[0]
            expected_joints = set(robot.robot_model.joints + robot.gripper.joints)
            for pose in robot_goal_poses:
                if set(pose) != expected_joints:
                    raise ValueError(
                        "Goal pose must contain exactly the arm/gripper joints."
                    )
        for stage, (predicate, source, target) in enumerate(predicates):
            rob = env.env
            obj = rob.objects_dict[source]
            plate = rob.objects_dict[target]
            if predicate != "on" or len(obj.joints) != 1:
                raise ValueError(
                    "Visual subgoals require an On goal with a free object."
                )
            start, end = sim.model.get_joint_qpos_addr(obj.joints[0])
            if end - start != 7:
                raise ValueError("Visual subgoal objects must have a free joint.")
            pose = sim.data.qpos[start:end].copy()
            offset = sim.data.get_body_xpos(
                plate.root_body
            ).copy() - sim.data.get_body_xpos(obj.root_body)
            for height in np.linspace(0.12, 0, 241):
                placed = pose.copy()
                placed[:3] += offset + [0, 0, height]
                sim.data.qpos[start:end] = placed
                sim.forward()
                if eval_predicate_fn(
                    predicate,
                    rob.object_states_dict[source],
                    rob.object_states_dict[target],
                ):
                    break
            else:
                raise ValueError(
                    f"Could not construct visual On target: {source} on {target}."
                )
            if not all(
                eval_predicate_fn(
                    p, rob.object_states_dict[a], rob.object_states_dict[b]
                )
                for p, a, b in predicates[: len(images) + 1]
            ):
                raise ValueError(
                    "Visual target placement invalidated an earlier subgoal."
                )
            if robot_goal_poses is not None:
                for name, value in robot_goal_poses[stage].items():
                    joint_id = sim.model.joint_name2id(name)
                    limits = sim.model.jnt_range[joint_id]
                    if not np.isfinite(value) or (
                        sim.model.jnt_limited[joint_id]
                        and not limits[0] - 1e-6 <= value <= limits[1] + 1e-6
                    ):
                        raise ValueError(f"Goal pose is outside joint limits: {name}.")
                    sim.data.set_joint_qpos(name, value)
                sim.forward()
            env._post_process()
            env._update_observables(force=True)
            obs = rob._get_observations()
            if "agentview_image" not in obs:
                raise ValueError("Visual subgoals require the agentview RGB camera.")
            images.append(obs["agentview_image"].copy())
    finally:
        env.set_state(state)
        sim.data.ctrl[:] = ctrl
        sim.forward()
        sim.data.qacc_warmstart[:] = warmstart
        env._post_process()
        env._update_observables(force=True)
    return images


class AtomicSequenceEnv:
    """Reset the robot between manifest goals without resetting scene objects.

    Goal-image mode renders terminal targets or caches marked transfer-start
    images (G1), and never substitutes gripper-release actions. Homing can be disabled independently.
    The native BDDL conjunction still defines episode success. Stage budgets
    include automatic gripper-opening actions. A stage boundary discards the
    rest of the current policy chunk until the next observation is consumed.
    Atomic instructions follow manifest order; full instructions allow either
    goal to be completed first and remain unchanged throughout the episode.
    """

    def __init__(
        self,
        env: Any,
        subgoals: list[dict[str, Any]],
        max_steps_per_subgoal: int,
        instruction_mode: str = "atomic",
        goal_image: bool = False,
        reset_robot_between_subgoals: bool = True,
        goal_robot_pose_path: str | None = None,
        refresh_goal_images: bool = False,
        goal_image_mode: str = "terminal",
    ) -> None:
        if len(env.robots) != 1:
            raise ValueError("Atomic sequence evaluation requires one robot.")
        self.env = env
        self.subgoals = subgoals
        self.predicates = validate_atomic_subgoals(
            subgoals,
            max_steps_per_subgoal,
            env.env.parsed_problem["goal_state"],
            instruction_mode,
        )
        if goal_image and instruction_mode != "atomic":
            raise ValueError("Visual subgoals require atomic instruction mode.")
        if goal_image_mode not in {"terminal", "g1"}:
            raise ValueError("goal_image_mode must be 'terminal' or 'g1'.")
        if goal_image_mode == "g1" and (
            not goal_image
            or not refresh_goal_images
            or goal_robot_pose_path is not None
            or reset_robot_between_subgoals
        ):
            raise ValueError(
                "G1 requires goal images and stage refresh, without robot poses or homing."
            )
        if not goal_image and (goal_robot_pose_path or refresh_goal_images):
            raise ValueError(
                "Robot goal poses and goal refresh require goal-image mode."
            )
        self.goal_image_mode = goal_image_mode
        self.goal_image = goal_image
        self.robot_goal_poses = (
            load_goal_robot_poses(goal_robot_pose_path, self.predicates)
            if goal_robot_pose_path is not None
            else None
        )
        self.refresh_goal_images = refresh_goal_images
        self.reset_robot_between_subgoals = reset_robot_between_subgoals
        self.instruction_mode = instruction_mode
        self.max_steps_per_subgoal = max_steps_per_subgoal
        self._active = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def reset(self) -> Any:
        """Reset the full episode; begin the sequence after initial settling."""
        self._active = False
        return self.env.reset()

    def begin_atomic_sequence(self) -> tuple[dict, str]:
        """Capture the settled robot pose and select the configured instruction."""
        robot = self.env.robots[0]
        self._qpos_indices = (
            robot._ref_joint_pos_indexes + robot._ref_gripper_joint_pos_indexes
        )
        self._qvel_indices = (
            robot._ref_joint_vel_indexes + robot._ref_gripper_joint_vel_indexes
        )
        self._ctrl_indices = (
            robot._ref_joint_actuator_indexes
            + robot._ref_joint_gripper_actuator_indexes
        )
        self._initial_qpos = self.env.sim.data.qpos[self._qpos_indices].copy()
        self._initial_ctrl = self.env.sim.data.ctrl[self._ctrl_indices].copy()
        self._initial_gripper_action = robot.gripper.current_action.copy()
        self._stage = 0
        self._stage_steps = 0
        self._completed_goals: set[int] = set()
        self._releasing_goal: int | None = None
        self._waiting_for_policy = False
        self._finished = False
        self._active = True
        if self.goal_image:
            if self.refresh_goal_images:
                self._goal_images = [None] * len(self.predicates)
                self._refresh_goal_image()
            else:
                self._goal_images = render_subgoal_images(
                    self.env, self.predicates, self.robot_goal_poses
                )
        self._last_obs = self._refresh_obs()
        return self._observation(), self._task_description()

    def _refresh_goal_image(self) -> None:
        """Render the active target from the current scene at a stage boundary."""
        if self.goal_image_mode == "g1":
            from rlinf.envs.sim.libero.g1_eval import render_g1_subgoal_image

            self._goal_images[self._stage] = render_g1_subgoal_image(
                self.env, self.predicates[self._stage]
            )
            return
        poses = (
            self.robot_goal_poses[self._stage : self._stage + 1]
            if self.robot_goal_poses is not None
            else None
        )
        self._goal_images[self._stage] = render_subgoal_images(
            self.env, self.predicates[self._stage : self._stage + 1], poses
        )[0]

    def _observation(self) -> dict:
        if self.goal_image:
            return {**self._last_obs, "goal_image": self._goal_images[self._stage]}
        return self._last_obs

    def _task_description(self) -> str:
        if self.instruction_mode == "full":
            return self.env.language_instruction
        return self.subgoals[self._stage]["prompt"]

    def resume_atomic_sequence(self) -> None:
        """Allow a fresh policy chunk to execute the current stage."""
        self._waiting_for_policy = False

    def _refresh_obs(self) -> dict:
        self.env.sim.forward()
        self.env._post_process()
        self.env._update_observables(force=True)
        return self.env.env._get_observations()

    def restore_robot(self) -> dict:
        """Restore the captured arm/gripper pose without advancing scene physics."""
        robot = self.env.robots[0]
        # Rebuild controller and robot history buffers, then restore the captured
        # pose rather than robosuite's nominal robot.init_qpos.
        robot.reset(deterministic=True)
        sim = self.env.sim
        sim.data.qpos[self._qpos_indices] = self._initial_qpos
        sim.data.qvel[self._qvel_indices] = 0
        sim.data.qacc_warmstart[self._qvel_indices] = 0
        sim.data.ctrl[self._ctrl_indices] = self._initial_ctrl
        robot.gripper.current_action = self._initial_gripper_action.copy()
        sim.forward()
        robot.controller.update(force=True)
        robot.controller.update_initial_joints(
            sim.data.qpos[robot._ref_joint_pos_indexes].copy()
        )
        robot.controller.reset_goal()
        return self._refresh_obs()

    def _goal_holds(self, index: int) -> bool:
        predicate, source, target = self.predicates[index]
        states = self.env.env.object_states_dict
        return bool(eval_predicate_fn(predicate, states[source], states[target]))

    def step(self, action: np.ndarray) -> tuple:
        """Execute or release the current goal, preserving native joint success."""
        if not self._active:
            return self.env.step(action)
        info = {}
        reward = 0.0
        if not self._finished and not self._waiting_for_policy:
            if self._releasing_goal is not None and not self.goal_image:
                action = np.zeros_like(action)
                action[-1] = -1
            self._last_obs, reward, _, info = self.env.step(action)
            self._stage_steps += 1
            if self._releasing_goal is not None and not self._goal_holds(
                self._releasing_goal
            ):
                self._releasing_goal = None
            if self._releasing_goal is None:
                candidates = (
                    [self._stage]
                    if self.instruction_mode == "atomic"
                    else [
                        i
                        for i in range(len(self.predicates))
                        if i not in self._completed_goals
                    ]
                )
                self._releasing_goal = next(
                    (i for i in candidates if self._goal_holds(i)), None
                )
            if self._releasing_goal is not None:
                source = self.predicates[self._releasing_goal][1]
                grasped = self.env.env._check_grasp(
                    self.env.robots[0].gripper, self.env.env.objects_dict[source]
                )
                if not grasped:
                    self._completed_goals.add(self._releasing_goal)
                    if len(self._completed_goals) < len(self.subgoals):
                        if self.reset_robot_between_subgoals:
                            self._last_obs = self.restore_robot()
                        self._stage += 1
                        if self.goal_image and self.refresh_goal_images:
                            self._refresh_goal_image()
                        self._stage_steps = 0
                        self._releasing_goal = None
                        self._waiting_for_policy = True
                    else:
                        self._finished = True
            if self._stage_steps >= self.max_steps_per_subgoal:
                self._finished = True
        info["atomic_task_description"] = self._task_description()
        if self.goal_image:
            info["goal_image_stage"] = self._stage
        return self._observation(), reward, self.env.check_success(), info
