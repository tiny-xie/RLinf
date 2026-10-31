# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Scene-aware pick/lift and released-place goal-image evaluation for LIBERO."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from libero.libero.envs.predicates import eval_predicate_fn
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from rlinf.envs.sim.libero.atomic_eval import AtomicSequenceEnv

GOAL_PROMPT = "achieve the state shown in the goal image"


def load_pick_place_templates(path: str, predicates: list[tuple]) -> list[dict]:
    """Require finite, complete cup-relative targets for each requested atom."""
    library = json.loads(Path(path).read_text())
    if library.get("version") != 1:
        raise ValueError("Pick/place template library must have version 1.")
    entries = {}
    shapes = {
        "eef_position_in_cup": (3,),
        "eef_rotation_in_cup": (3, 3),
        "arm_qpos": (7,),
        "gripper_qpos": (2,),
    }
    for entry in library["goals"]:
        key = tuple(x.lower() for x in entry["predicate"])
        if key in entries:
            raise ValueError(f"Duplicate pick/place template: {key}")
        for phase in ("pick", "place"):
            for name, shape in shapes.items():
                value = np.asarray(entry[phase][name], dtype=float)
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f"Invalid {phase} template field: {name}")
            rotation = np.asarray(entry[phase]["eef_rotation_in_cup"])
            if not np.allclose(
                rotation.T @ rotation, np.eye(3), atol=1e-5
            ) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5):
                raise ValueError("Template orientation must be a rotation matrix.")
        entries[key] = entry
    if any(tuple(p) not in entries for p in predicates):
        raise ValueError("Missing pick/place template for a requested predicate.")
    return [entries[tuple(p)] for p in predicates]


def _set_target_arm(env: Any, target: dict, cup: Any) -> None:
    """Solve joint-limited IK for a demonstration gripper pose relative to a cup."""
    sim, robot = env.sim, env.robots[0]
    cpos = sim.data.get_body_xpos(cup.root_body).copy()
    crot = sim.data.get_body_xmat(cup.root_body).copy()
    desired_pos = cpos + crot @ np.asarray(target["eef_position_in_cup"])
    desired_rot = crot @ np.asarray(target["eef_rotation_in_cup"])
    indices = robot._ref_joint_pos_indexes
    limits = np.array(
        [
            sim.model.jnt_range[sim.model.joint_name2id(name)]
            for name in robot.robot_model.joints
        ]
    )
    site = robot.eef_site_id

    def residual(qpos):
        sim.data.qpos[indices] = qpos
        sim.forward()
        delta_rot = desired_rot @ sim.data.site_xmat[site].reshape(3, 3).T
        return np.r_[
            sim.data.site_xpos[site] - desired_pos,
            Rotation.from_matrix(delta_rot).as_rotvec() * 0.2,
        ]

    seeds = [target["arm_qpos"], sim.data.qpos[indices].copy()]
    for seed in seeds:
        result = least_squares(
            residual,
            np.clip(seed, limits[:, 0] + 1e-6, limits[:, 1] - 1e-6),
            bounds=(limits[:, 0], limits[:, 1]),
            max_nfev=160,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        error = residual(result.x)
        if np.linalg.norm(error[:3]) < 0.001 and np.linalg.norm(error[3:]) < 0.002:
            break
    else:
        raise ValueError(
            f"Could not reconstruct reachable goal gripper pose (residual={error})."
        )
    sim.data.qpos[robot._ref_gripper_joint_pos_indexes] = target["gripper_qpos"]
    sim.forward()


def render_pick_place_goal(
    env: Any,
    predicate: tuple,
    phase: str,
    template: dict,
    initial_cup_qpos: np.ndarray,
    lift_height: float,
) -> np.ndarray:
    """Render one current-scene target without changing the physical environment.

    Pick keeps the current cup XY and its initial upright orientation, lifting
    it by lift_height. Place centers it on the target plate. Other objects stay
    at their actual current poses. The arm/gripper are reconstructed by IK.
    """
    if (
        phase not in ("pick", "place")
        or not np.isfinite(lift_height)
        or lift_height <= 0
    ):
        raise ValueError("Expected pick/place phase and positive lift height.")
    sim, rob = env.sim, env.env
    state, ctrl, warmstart = (
        env.get_sim_state().copy(),
        sim.data.ctrl.copy(),
        sim.data.qacc_warmstart.copy(),
    )
    try:
        _, source, target = predicate
        cup, plate = rob.objects_dict[source], rob.objects_dict[target]
        start, end = sim.model.get_joint_qpos_addr(cup.joints[0])
        if end - start != 7:
            raise ValueError("Pick/place target requires a free cup joint.")
        pose = sim.data.qpos[start:end].copy()
        pose[3:] = initial_cup_qpos[3:]
        if phase == "pick":
            pose[2] = initial_cup_qpos[2] + lift_height
            sim.data.qpos[start:end] = pose
            sim.forward()
        else:
            sim.data.qpos[start:end] = pose
            sim.forward()
            offset = sim.data.get_body_xpos(
                plate.root_body
            ).copy() - sim.data.get_body_xpos(cup.root_body)
            for height in np.linspace(0.12, 0, 241):
                placed = pose.copy()
                placed[:3] += offset + [0, 0, height]
                sim.data.qpos[start:end] = placed
                sim.forward()
                if eval_predicate_fn(
                    "on", rob.object_states_dict[source], rob.object_states_dict[target]
                ):
                    break
            else:
                raise ValueError("Could not construct the placement target.")
        _set_target_arm(env, template[phase], cup)
        if phase == "pick" and not rob._check_grasp(env.robots[0].gripper, cup):
            raise ValueError(
                "Reconstructed pick goal does not grasp the requested cup."
            )
        if phase == "place" and rob._check_grasp(env.robots[0].gripper, cup):
            raise ValueError("Reconstructed place goal has not released the cup.")
        env._post_process()
        env._update_observables(force=True)
        obs = rob._get_observations()
        if "agentview_image" not in obs:
            raise ValueError(
                "Pick/place goal generation requires the agentview camera."
            )
        return obs["agentview_image"].copy()
    finally:
        env.set_state(state)
        sim.data.ctrl[:] = ctrl
        sim.forward()
        sim.data.qacc_warmstart[:] = warmstart
        env._post_process()
        env._update_observables(force=True)


class PickPlaceSequenceEnv(AtomicSequenceEnv):
    """Run pick/place pairs without homing, using physics-checked phase switches.

    A pick needs grasp contact on the named cup and a sustained object-height
    gain. A place needs the On predicate and release. A switch pauses the old
    action chunk. Success requires all ordered phases and the native conjunction.
    """

    def __init__(
        self,
        env: Any,
        subgoals: list[dict],
        max_steps_per_subgoal: int,
        template_path: str,
        lift_threshold: float = 0.04,
        goal_lift_height: float = 0.06,
        hold_steps: int = 3,
    ) -> None:
        super().__init__(
            env,
            subgoals,
            max_steps_per_subgoal,
            goal_image=True,
            reset_robot_between_subgoals=False,
        )
        if not (
            np.isfinite(lift_threshold)
            and np.isfinite(goal_lift_height)
            and 0 < lift_threshold <= goal_lift_height
        ):
            raise ValueError("Require 0 < lift_threshold <= goal_lift_height.")
        if (
            not isinstance(hold_steps, int)
            or isinstance(hold_steps, bool)
            or hold_steps < 1
        ):
            raise ValueError("hold_steps must be a positive integer.")
        self.templates = load_pick_place_templates(template_path, self.predicates)
        self.lift_threshold, self.goal_lift_height = lift_threshold, goal_lift_height
        self.hold_steps = hold_steps

    def begin_atomic_sequence(self) -> tuple[dict, str]:
        """Capture initial cup heights and render the first pick target."""
        self._stage, self._stage_steps, self._hold_count = 0, 0, 0
        self._completed_stages = 0
        self._finished, self._timed_out, self._waiting_for_policy = False, False, False
        self._active = True
        self._initial_cups, self._initial_heights = {}, {}
        for _, source, _ in self.predicates:
            cup = self.env.env.objects_dict[source]
            self._initial_cups[source] = self.env.sim.data.get_joint_qpos(
                cup.joints[0]
            ).copy()
            self._initial_heights[source] = float(
                self.env.sim.data.get_body_xpos(cup.root_body)[2]
            )
        self._refresh_goal_image()
        self._last_obs = self._refresh_obs()
        return self._observation(), GOAL_PROMPT

    def _task_description(self) -> str:
        return GOAL_PROMPT

    def _refresh_goal_image(self) -> None:
        atom = self._stage // 2
        phase = "pick" if self._stage % 2 == 0 else "place"
        predicate = self.predicates[atom]
        self._current_goal = render_pick_place_goal(
            self.env,
            predicate,
            phase,
            self.templates[atom],
            self._initial_cups[predicate[1]],
            self.goal_lift_height,
        )

    def _observation(self) -> dict:
        return {**self._last_obs, "goal_image": self._current_goal}

    def check_success(self) -> bool:
        """Require completed phase sequence as well as the current native goal."""
        return bool(
            self._active
            and self._completed_stages == 2 * len(self.predicates)
            and self.env.check_success()
        )

    def step(self, action: np.ndarray) -> tuple:
        """Execute one physical action unless waiting for a newly conditioned chunk."""
        if not self._active:
            return self.env.step(action)
        reward, info = 0.0, {}
        if not self._finished and not self._waiting_for_policy:
            self._last_obs, reward, _, info = self.env.step(action)
            self._stage_steps += 1
            atom = self._stage // 2
            source = self.predicates[atom][1]
            cup = self.env.env.objects_dict[source]
            grasped = self.env.env._check_grasp(self.env.robots[0].gripper, cup)
            if self._stage % 2 == 0:
                height = (
                    self.env.sim.data.get_body_xpos(cup.root_body)[2]
                    - self._initial_heights[source]
                )
                holds = grasped and height >= self.lift_threshold
            else:
                holds = self._goal_holds(atom) and not grasped
            self._hold_count = self._hold_count + 1 if holds else 0
            if self._hold_count >= self.hold_steps:
                self._completed_stages += 1
                if self._completed_stages == 2 * len(self.predicates):
                    self._finished = True
                else:
                    self._stage += 1
                    self._stage_steps, self._hold_count = 0, 0
                    self._refresh_goal_image()
                    self._waiting_for_policy = True
            elif self._stage_steps >= self.max_steps_per_subgoal:
                self._timed_out, self._finished = True, True
        info.update(
            atomic_task_description=GOAL_PROMPT,
            goal_image_stage=self._stage,
            pick_place_phase=self._stage % 2,
            pick_place_completed_stages=self._completed_stages,
            pick_place_total_stages=2 * len(self.predicates),
            pick_place_timed_out=self._timed_out,
            pick_place_native_success=bool(self.env.check_success()),
        )
        return self._observation(), reward, self.check_success(), info
