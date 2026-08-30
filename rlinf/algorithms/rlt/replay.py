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

from dataclasses import dataclass
from typing import Any

import torch

from rlinf.algorithms.rlt.transition import (
    RLT_OBS_KEYS,
    extract_rlt_obs_from_forward_inputs,
)
from rlinf.data.embodied_io_struct import (
    ChunkStepResult,
    EmbodiedRolloutResult,
    EnvOutput,
)


def _slice_obs(obs: dict[str, Any], start: int, stop: int) -> dict[str, Any]:
    return {
        key: None if value is None else value[start:stop] for key, value in obs.items()
    }


def _stack_single_observations(
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    if not observations:
        return {}
    stacked = {}
    for key in observations[0]:
        values = [obs[key] for obs in observations]
        first_non_none = next((value for value in values if value is not None), None)
        if first_non_none is None:
            stacked[key] = None
        elif isinstance(first_non_none, torch.Tensor):
            stacked[key] = torch.cat(values, dim=0).cpu().contiguous()
        else:
            stacked[key] = [item for value in values for item in value]
    return stacked


@dataclass(kw_only=True)
class RLTReplayWindow:
    """One fixed-horizon replay row before Stage 1 feature backfill."""

    curr_anchor: int
    next_anchor: int
    actions: torch.Tensor
    rewards: torch.Tensor
    terminations: torch.Tensor
    truncations: torch.Tensor
    dones: torch.Tensor
    human_actions: torch.Tensor
    human_flags: torch.Tensor
    intervene_flags: torch.Tensor
    versions: torch.Tensor
    record_transition: torch.Tensor


@dataclass(kw_only=True)
class RLTReplayBuildRequest:
    """Raw anchors and aligned windows sent for post-rollout feature backfill."""

    anchor_obs: dict[str, Any]
    raw_anchor_positions: list[int]
    cached_anchor_features: dict[str, torch.Tensor]
    cached_anchor_positions: list[int]
    anchor_count: int
    windows: list[RLTReplayWindow]
    action_dim: int
    chunk_len: int
    max_episode_length: int

    @property
    def batch_size(self) -> int:
        """Return the supported logical batch size for routed communication."""
        return 1

    def materialize(
        self, anchor_features: dict[str, torch.Tensor]
    ) -> EmbodiedRolloutResult:
        """Convert raw windows into replay-ready RLT transitions."""
        destination = EmbodiedRolloutResult(max_episode_length=self.max_episode_length)
        for window in self.windows:
            curr_obs = {
                key: anchor_features[key][window.curr_anchor : window.curr_anchor + 1]
                .detach()
                .clone()
                for key in RLT_OBS_KEYS
            }
            next_obs = {
                key: anchor_features[key][window.next_anchor : window.next_anchor + 1]
                .detach()
                .clone()
                for key in RLT_OBS_KEYS
            }

            ref_chunk = curr_obs["ref_chunk"]
            ref_actions = ref_chunk.reshape(1, -1, self.action_dim).clone()
            # The raw window and this reference share the same start frame, so
            # frame-rate takeover rewrites the corresponding leading reference.
            ref_actions[:, : self.chunk_len] = torch.where(
                window.human_flags.unsqueeze(-1),
                window.human_actions.to(ref_actions.dtype),
                ref_actions[:, : self.chunk_len],
            )
            curr_obs["ref_chunk"] = ref_actions.reshape_as(ref_chunk)

            destination.append_step_result(
                ChunkStepResult(
                    actions=window.actions.reshape(1, -1),
                    rewards=window.rewards,
                    terminations=window.terminations,
                    truncations=window.truncations,
                    dones=window.dones,
                    versions=window.versions,
                    forward_inputs={
                        "record_transition": window.record_transition,
                    },
                )
            )
            destination.mark_last_step_with_intervene_flags(window.intervene_flags)
            destination.append_transitions(curr_obs, next_obs)
        return destination


class RLTRawReplayBuilder:
    """Collect a frame trace and create overlapping replay windows after rollout.

    This builder intentionally stores raw observations. Stage 1 features are
    computed once, in batches, only after robot interaction has finished.
    """

    def __init__(
        self,
        *,
        chunk_len: int,
        action_dim: int,
        stride: int,
        max_episode_length: int,
        intervention_only: bool = False,
    ) -> None:
        self.chunk_len = chunk_len
        self.action_dim = action_dim
        self.stride = stride
        self.max_episode_length = max_episode_length
        self.intervention_only = intervention_only

        self._state_obs: dict[int, dict[str, Any]] = {}
        self._transition_obs: dict[int, dict[str, Any]] = {}
        self._state_features: dict[int, dict[str, torch.Tensor]] = {}
        self._transition_features: dict[int, dict[str, torch.Tensor]] = {}
        self._actions: list[torch.Tensor] = []
        self._rewards: list[torch.Tensor] = []
        self._terminations: list[torch.Tensor] = []
        self._truncations: list[torch.Tensor] = []
        self._dones: list[torch.Tensor] = []
        self._human_actions: list[torch.Tensor] = []
        self._human_flags: list[torch.Tensor] = []
        self._intervene_flags: list[torch.Tensor] = []
        self._versions: list[torch.Tensor] = []
        self._record_transition: list[torch.Tensor] = []

    @staticmethod
    def _frame_chunk(
        value: Any,
        *,
        chunk_len: int,
        feature_dim: int | None,
    ) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        value = value.detach().cpu().contiguous()
        shape = (1, chunk_len) if feature_dim is None else (1, chunk_len, feature_dim)
        return value.reshape(shape)

    def cache_initial(self, obs: dict[str, Any], rollout_result: Any) -> None:
        """Cache the raw observation before the first executed action."""
        self._state_obs[0] = obs
        self._transition_obs[0] = obs
        features = extract_rlt_obs_from_forward_inputs(rollout_result.forward_inputs)
        self._state_features[0] = features
        self._transition_features[0] = features

    def append_chunk(
        self,
        *,
        action_rollout_result: Any,
        boundary_rollout_result: Any,
        env_output: Any,
        chunk_actions: torch.Tensor,
        rewards: torch.Tensor,
        observations: list[dict[str, Any]],
        final_obs: dict[str, Any] | None,
    ) -> None:
        """Append an executed chunk without running the Stage 1 model."""
        chunk_start = len(self._actions)
        executed_len = len(observations)

        self._state_features[chunk_start] = extract_rlt_obs_from_forward_inputs(
            action_rollout_result.forward_inputs
        )

        actions = self._frame_chunk(
            chunk_actions,
            chunk_len=executed_len,
            feature_dim=self.action_dim,
        )
        reward_chunk = self._frame_chunk(
            rewards,
            chunk_len=executed_len,
            feature_dim=None,
        )
        terminations = self._frame_chunk(
            env_output.terminations,
            chunk_len=executed_len,
            feature_dim=None,
        ).to(torch.bool)
        truncations = self._frame_chunk(
            env_output.truncations,
            chunk_len=executed_len,
            feature_dim=None,
        ).to(torch.bool)
        dones = torch.logical_or(terminations, truncations)

        if env_output.intervene_flags is None:
            human_flags = torch.zeros((1, executed_len), dtype=torch.bool)
            human_actions = torch.zeros_like(actions)
        else:
            human_flags = self._frame_chunk(
                env_output.intervene_flags,
                chunk_len=executed_len,
                feature_dim=None,
            ).to(torch.bool)
            if env_output.intervene_actions is None:
                raise ValueError(
                    "intervene_flags were provided without intervene_actions."
                )
            human_actions = self._frame_chunk(
                env_output.intervene_actions,
                chunk_len=executed_len,
                feature_dim=self.action_dim,
            )

        effective_actions = torch.where(
            human_flags.unsqueeze(-1), human_actions.to(actions.dtype), actions
        )
        route_flags = action_rollout_result.intervene_flags
        if route_flags is None:
            intervene_flags = human_flags
        else:
            route_flags = self._frame_chunk(
                route_flags,
                chunk_len=executed_len,
                feature_dim=None,
            ).to(torch.bool)
            intervene_flags = torch.logical_or(route_flags, human_flags)

        versions = action_rollout_result.versions
        if versions is None:
            versions = torch.zeros((1, 1), dtype=torch.float32)
        else:
            versions = versions.detach().cpu().contiguous()
        record_transition = action_rollout_result.forward_inputs.get(
            "record_transition"
        )
        if record_transition is None:
            record_transition = torch.ones((1, 1), dtype=torch.bool)
        else:
            record_transition = (
                record_transition.detach().cpu().to(torch.bool).reshape(1, -1)[:, :1]
            )

        for offset in range(executed_len):
            frame = chunk_start + offset
            self._actions.append(effective_actions[:, offset])
            self._rewards.append(reward_chunk[:, offset])
            self._terminations.append(terminations[:, offset])
            self._truncations.append(truncations[:, offset])
            self._dones.append(dones[:, offset])
            self._human_actions.append(human_actions[:, offset])
            self._human_flags.append(human_flags[:, offset])
            self._intervene_flags.append(intervene_flags[:, offset])
            self._versions.append(versions)
            self._record_transition.append(record_transition)

            next_frame = frame + 1
            self._state_obs[next_frame] = observations[offset]
            self._transition_obs[next_frame] = observations[offset]

        chunk_end = chunk_start + executed_len
        self._state_features[chunk_end] = extract_rlt_obs_from_forward_inputs(
            boundary_rollout_result.forward_inputs
        )
        self._transition_features[chunk_end] = extract_rlt_obs_from_forward_inputs(
            boundary_rollout_result.forward_inputs,
            transition=True,
        )
        if bool(dones[:, -1].any()) and final_obs is not None:
            self._transition_obs[chunk_end] = final_obs

    def _window_tensor(self, frames: list[torch.Tensor], start: int) -> torch.Tensor:
        return torch.stack(
            [frames[start + offset][0] for offset in range(self.chunk_len)], dim=0
        ).unsqueeze(0)

    def build_request(self) -> RLTReplayBuildRequest:
        """Build fixed-length, stride-spaced windows and deduplicated anchors."""
        anchor_rows: list[dict[str, Any]] = []
        raw_anchor_positions: list[int] = []
        cached_feature_rows: list[dict[str, torch.Tensor]] = []
        cached_anchor_positions: list[int] = []
        anchor_indices: dict[tuple[str, int], int] = {}

        def add_anchor(kind: str, frame: int) -> int:
            key = (kind, frame)
            if key in anchor_indices:
                return anchor_indices[key]
            source = self._state_obs if kind == "state" else self._transition_obs
            feature_source = (
                self._state_features if kind == "state" else self._transition_features
            )
            anchor_idx = len(anchor_indices)
            anchor_indices[key] = anchor_idx
            if frame in feature_source:
                cached_feature_rows.append(feature_source[frame])
                cached_anchor_positions.append(anchor_idx)
            else:
                anchor_rows.append(_slice_obs(source[frame], 0, 1))
                raw_anchor_positions.append(anchor_idx)
            return anchor_indices[key]

        windows = []
        available = len(self._actions)
        start = 0
        while start + self.chunk_len <= available:
            done_before_end = [
                offset
                for offset in range(self.chunk_len - 1)
                if bool(self._dones[start + offset][0])
            ]
            if done_before_end:
                start += done_before_end[0] + 1
                continue

            terminal = bool(self._dones[start + self.chunk_len - 1][0])
            end = start + self.chunk_len
            intervene_flags = self._window_tensor(self._intervene_flags, start)
            if self.intervention_only and not bool(intervene_flags.any()):
                start = end if terminal else start + self.stride
                continue

            curr_anchor = add_anchor("state", start)
            next_anchor = add_anchor("transition" if terminal else "state", end)
            windows.append(
                RLTReplayWindow(
                    curr_anchor=curr_anchor,
                    next_anchor=next_anchor,
                    actions=self._window_tensor(self._actions, start),
                    rewards=self._window_tensor(self._rewards, start),
                    terminations=self._window_tensor(self._terminations, start),
                    truncations=self._window_tensor(self._truncations, start),
                    dones=self._window_tensor(self._dones, start),
                    human_actions=self._window_tensor(self._human_actions, start),
                    human_flags=self._window_tensor(self._human_flags, start),
                    intervene_flags=intervene_flags,
                    versions=self._versions[start],
                    record_transition=self._record_transition[start],
                )
            )
            start = end if terminal else start + self.stride

        return RLTReplayBuildRequest(
            anchor_obs=_stack_single_observations(anchor_rows),
            raw_anchor_positions=raw_anchor_positions,
            cached_anchor_features={
                key: torch.cat(
                    [features[key] for features in cached_feature_rows], dim=0
                )
                .detach()
                .cpu()
                .contiguous()
                for key in RLT_OBS_KEYS
            }
            if cached_feature_rows
            else {},
            cached_anchor_positions=cached_anchor_positions,
            anchor_count=len(anchor_indices),
            windows=windows,
            action_dim=self.action_dim,
            chunk_len=self.chunk_len,
            max_episode_length=self.max_episode_length,
        )


class RLTEnvReplaySession:
    """Keep RLT-specific raw-trace state out of the generic env loop."""

    def __init__(
        self,
        *,
        stage_num: int,
        chunk_len: int,
        action_dim: int,
        stride: int,
        max_episode_length: int,
        use_training_pipeline: bool,
        intervention_only: bool = False,
    ) -> None:
        self.stage_num = stage_num
        self.chunk_len = chunk_len
        self.action_dim = action_dim
        self.stride = stride
        self.max_episode_length = max_episode_length
        self.use_training_pipeline = use_training_pipeline
        self.intervention_only = intervention_only
        self.builders: list[RLTRawReplayBuilder] = []
        self.pending_chunks: list[dict[str, Any] | None] = []
        self.accumulated_results = (
            [
                EmbodiedRolloutResult(max_episode_length=max_episode_length)
                for _ in range(stage_num)
            ]
            if not use_training_pipeline
            else []
        )

    @staticmethod
    def _prepare_observations(
        observations: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    ) -> list[dict[str, Any]]:
        return [EnvOutput(obs=obs).to_dict()["obs"] for obs in observations]

    @staticmethod
    def _extend_result(
        destination: EmbodiedRolloutResult,
        source: EmbodiedRolloutResult,
    ) -> None:
        for field_name in (
            "actions",
            "intervene_flags",
            "rewards",
            "terminations",
            "truncations",
            "dones",
            "prev_logprobs",
            "prev_values",
            "versions",
            "forward_inputs",
            "curr_obs",
            "next_obs",
        ):
            getattr(destination, field_name).extend(getattr(source, field_name))

    def start_epoch(self) -> None:
        """Reset raw builders while retaining non-pipeline accumulated replay."""
        self.builders = [
            RLTRawReplayBuilder(
                chunk_len=self.chunk_len,
                action_dim=self.action_dim,
                stride=self.stride,
                max_episode_length=self.max_episode_length,
                intervention_only=self.intervention_only,
            )
            for _ in range(self.stage_num)
        ]
        self.pending_chunks = [None] * self.stage_num

    def process_boundary(
        self,
        *,
        stage_id: int,
        env_output: EnvOutput,
        rollout_result: Any,
        rewards: torch.Tensor | None,
    ) -> None:
        """Finish the preceding raw chunk at a normal rollout boundary."""
        pending = self.pending_chunks[stage_id]
        if pending is None:
            self.builders[stage_id].cache_initial(
                env_output.to_dict()["obs"], rollout_result
            )
            return
        if rewards is None:
            raise ValueError("Executed RLT chunk is missing frame rewards.")
        prepared_output = env_output.to_dict()
        self.builders[stage_id].append_chunk(
            action_rollout_result=pending["rollout_result"],
            boundary_rollout_result=rollout_result,
            env_output=env_output,
            chunk_actions=pending["chunk_actions"],
            rewards=rewards,
            observations=self._prepare_observations(pending["obs_list"]),
            final_obs=prepared_output["final_obs"],
        )
        self.pending_chunks[stage_id] = None

    def record_executed_chunk(
        self,
        *,
        stage_id: int,
        rollout_result: Any,
        chunk_step_payload: dict[str, Any],
    ) -> None:
        """Retain raw frames until the following boundary supplies rewards."""
        self.pending_chunks[stage_id] = {
            "rollout_result": rollout_result,
            "chunk_actions": chunk_step_payload["chunk_actions"],
            "obs_list": chunk_step_payload["obs_list"],
        }

    def build_requests(self) -> list[RLTReplayBuildRequest]:
        """Create one post-rollout replay request per pipeline stage."""
        return [builder.build_request() for builder in self.builders]

    @staticmethod
    def _split_single_request(
        request: RLTReplayBuildRequest,
        sizes: list[int],
    ) -> list[RLTReplayBuildRequest]:
        return [request]

    def accept_result(
        self,
        *,
        stage_id: int,
        live_result: EmbodiedRolloutResult,
        replay_result: EmbodiedRolloutResult,
    ) -> None:
        """Discard temporary legacy rows and retain materialized sliding rows."""
        live_result.clear()
        destination = (
            live_result
            if self.use_training_pipeline
            else self.accumulated_results[stage_id]
        )
        self._extend_result(destination, replay_result)

    def final_results(
        self, live_results: list[EmbodiedRolloutResult]
    ) -> list[EmbodiedRolloutResult]:
        """Return accumulated results for the runner's normal actor handoff."""
        return live_results if self.use_training_pipeline else self.accumulated_results

    def exchange_replay(
        self,
        *,
        worker: Any,
        group_name: str,
        request_channel: Any,
        result_channel: Any,
        live_results: list[EmbodiedRolloutResult],
    ) -> None:
        """Exchange post-rollout feature requests without leaking protocol details."""
        for stage_id, request in enumerate(self.build_requests()):
            worker.send_to(
                group_name=group_name,
                channel=request_channel,
                data=request,
                tag="rlt_replay_requests",
                route_key=stage_id,
                batch_size=1,
                split_fn=self._split_single_request,
            )
        for stage_id in range(self.stage_num):
            replay_result = worker.recv_from(
                group_name=group_name,
                channel=result_channel,
                tag="rlt_replay_results",
                route_key=stage_id,
                batch_size=1,
                infer_batch_size_fn=lambda _: 1,
            )
            self.accept_result(
                stage_id=stage_id,
                live_result=live_results[stage_id],
                replay_result=replay_result,
            )


def build_rlt_replay(
    request: RLTReplayBuildRequest,
    *,
    feature_model: Any,
    feature_batch_size: int,
) -> EmbodiedRolloutResult:
    """Backfill unique raw anchors and materialize replay transitions."""
    if feature_batch_size <= 0:
        raise ValueError(
            f"RLT feature_batch_size must be positive, got {feature_batch_size}."
        )
    if not request.windows:
        return EmbodiedRolloutResult(max_episode_length=request.max_episode_length)

    raw_anchor_count = next(
        (
            int(value.shape[0]) if isinstance(value, torch.Tensor) else len(value)
            for value in request.anchor_obs.values()
            if value is not None
        ),
        0,
    )
    feature_parts = {key: [] for key in RLT_OBS_KEYS}
    with torch.no_grad():
        for start in range(0, raw_anchor_count, feature_batch_size):
            micro_obs = _slice_obs(
                request.anchor_obs,
                start,
                min(start + feature_batch_size, raw_anchor_count),
            )
            features = feature_model.extract_rlt_obs(micro_obs)
            for key in RLT_OBS_KEYS:
                feature_parts[key].append(features[key].detach().cpu())
    raw_features = {
        key: torch.cat(parts, dim=0).contiguous() if parts else None
        for key, parts in feature_parts.items()
    }
    anchor_features = {}
    for key in RLT_OBS_KEYS:
        rows: list[torch.Tensor | None] = [None] * request.anchor_count
        cached = request.cached_anchor_features.get(key)
        if cached is not None:
            for row_idx, anchor_idx in enumerate(request.cached_anchor_positions):
                rows[anchor_idx] = cached[row_idx : row_idx + 1]
        raw = raw_features[key]
        if raw is not None:
            for row_idx, anchor_idx in enumerate(request.raw_anchor_positions):
                rows[anchor_idx] = raw[row_idx : row_idx + 1]
        if any(row is None for row in rows):
            raise ValueError(f"Missing backfilled RLT anchor features for '{key}'.")
        anchor_features[key] = torch.cat(rows, dim=0).contiguous()
    return request.materialize(anchor_features)


def _split_single_replay_result(result: Any, sizes: list[int]) -> list[Any]:
    return [result]


async def serve_rlt_replay_requests(
    *,
    worker: Any,
    group_name: str,
    input_channel: Any,
    output_channel: Any,
    feature_model: Any,
    feature_batch_size: int,
    stage_num: int,
) -> None:
    """Backfill replay anchors after online interaction has finished."""
    for stage_id in range(stage_num):
        request = await worker.recv_from(
            group_name=group_name,
            channel=input_channel,
            tag="rlt_replay_requests",
            route_key=stage_id,
            async_op=True,
            batch_size=1,
            infer_batch_size_fn=lambda value: value.batch_size,
        ).async_wait()
        replay_result = build_rlt_replay(
            request,
            feature_model=feature_model,
            feature_batch_size=feature_batch_size,
        )
        worker.send_to(
            group_name=group_name,
            channel=output_channel,
            data=replay_result,
            tag="rlt_replay_results",
            route_key=stage_id,
            async_op=True,
            batch_size=1,
            split_fn=_split_single_replay_result,
        )
