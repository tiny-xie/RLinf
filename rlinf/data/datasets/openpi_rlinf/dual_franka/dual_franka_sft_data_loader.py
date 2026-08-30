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

"""Dual-Franka tcp_rot6d SFT data loader for the openpi_rlinf path.

The loader owns dataset access, distributed sampling, and collation into the
openpi_rlinf ``Observation`` type. Input semantics come from the canonical
RLinf OpenPI config selected by ``config_name``::

    repack LeRobot keys
    -> DualFrankaTcpRot6dInputs (state reorder, camera map, pad)
    -> RigidBodyDeltaActions (body-frame SE(3); grippers absolute)
    -> quantile Normalize
    -> ModelTransformFactory (resize + tokenize)
    -> collate -> (Observation, actions [B, H, D])
"""

from __future__ import annotations

import dataclasses
import multiprocessing
import typing

import numpy as np
import torch
from openpi.transforms import DataTransformFn, compose
from torch.utils.data.distributed import DistributedSampler

from rlinf.data.datasets.openpi_rlinf.dual_franka.dual_franka_sft_dataset import (
    DualFrankaSftDataset,
)
from rlinf.data.storage.lerobot import resolve_lerobot_repo_id
from rlinf.models.embodiment.openpi_rlinf.pi0_model.model import Observation
from rlinf.models.embodiment.openpi_rlinf.transforms_pipeline import (
    build_openpi_transforms,
)
from rlinf.utils.logging import get_logger

logger = get_logger()

__all__ = [
    "DualFrankaSftRepack",
    "DualFrankaSftDataConfig",
    "DualFrankaSftDataLoader",
    "build_dual_franka_sft_dataloader",
    "build_dual_franka_sft_transform",
    "collate_dual_franka_sft_items",
    "create_dual_franka_sft_data_loader",
    "prepare_dual_franka_online_sft_batch",
]

_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")

_REPACK_KEYS = {
    "observation/image": "image",
    "observation/extra_view_image-0": "extra_view_image-0",
    "observation/extra_view_image-1": "extra_view_image-1",
    "observation/state": "state",
    "actions": "actions",
}


class DualFrankaSftRepack(DataTransformFn):
    """Map raw dual-Franka LeRobot keys to canonical OpenPI input keys."""

    def __call__(self, frame: dict[str, typing.Any]) -> dict[str, typing.Any]:
        data: dict[str, typing.Any] = {}
        for destination, source in _REPACK_KEYS.items():
            if source not in frame:
                raise KeyError(
                    f"dual_franka SFT frame missing {source!r} "
                    f"(needed for {destination!r}); "
                    f"available keys={sorted(frame.keys())}"
                )
            data[destination] = np.asarray(frame[source])

        prompt = frame.get("prompt", frame.get("task"))
        if prompt is None:
            raise ValueError(
                "dual_franka SFT frame is missing both 'prompt' and 'task'; "
                "ensure the LeRobot dataset contains task annotations."
            )
        if isinstance(prompt, bytes):
            prompt = prompt.decode("utf-8")
        elif not isinstance(prompt, str):
            prompt = prompt.item() if hasattr(prompt, "item") else str(prompt)
        data["prompt"] = prompt
        return data


def build_dual_franka_sft_transform(
    input_transforms: typing.Sequence[typing.Callable],
) -> typing.Callable[[dict[str, typing.Any]], dict[str, typing.Any]]:
    """Build the canonical raw-LeRobot-to-OpenPI SFT transform."""
    return compose([DualFrankaSftRepack(), *input_transforms])


def _take_online_batch_item(value: typing.Any, index: int) -> typing.Any:
    """Select one item from a collated online batch and make it NumPy-safe."""
    if isinstance(value, torch.Tensor):
        return value[index].detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value[index]
    if isinstance(value, (list, tuple)):
        return value[index]
    return value


def prepare_dual_franka_online_sft_batch(
    batch: typing.Mapping[str, typing.Any],
    transform: typing.Callable[[dict[str, typing.Any]], dict[str, typing.Any]],
) -> tuple[Observation, torch.Tensor]:
    """Transform a collated online-LeRobot batch into OpenPI SFT inputs.

    Online DAgger moves the collated raw batch to the actor device before the
    model adapter runs.  The OpenPI data transforms are NumPy transforms, so
    this function selects each sample and explicitly stages only that sample
    back to CPU before applying the same transform/collate path as offline SFT.
    """
    actions = batch.get("actions")
    if actions is None:
        raise KeyError("online dual_franka SFT batch is missing 'actions'.")
    if not hasattr(actions, "shape") or len(actions.shape) < 1:
        raise ValueError("online dual_franka SFT 'actions' must be batched.")

    batch_size = int(actions.shape[0])
    if batch_size <= 0:
        raise ValueError("Cannot prepare an empty online dual_franka SFT batch.")

    required_keys = tuple(_REPACK_KEYS.values())
    transformed_items = []
    for index in range(batch_size):
        frame = {
            key: _take_online_batch_item(batch[key], index)
            for key in required_keys
            if key in batch
        }
        if "prompt" in batch:
            frame["prompt"] = _take_online_batch_item(batch["prompt"], index)
        if "task" in batch:
            frame["task"] = _take_online_batch_item(batch["task"], index)
        transformed_items.append(transform(frame))

    return collate_dual_franka_sft_items(transformed_items)


class _TransformedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: DualFrankaSftDataset,
        transform: typing.Callable[[typing.Any], typing.Any],
    ) -> None:
        self._dataset = dataset
        self._transform = transform

    def __getitem__(self, idx: int) -> typing.Any:
        return self._transform(self._dataset[idx])

    def __len__(self) -> int:
        return len(self._dataset)


def collate_dual_franka_sft_items(
    items: typing.Sequence[typing.Mapping[str, typing.Any]],
) -> tuple[Observation, torch.Tensor]:
    """Collate transformed items into ``(Observation, actions)``."""
    if not items:
        raise ValueError("Cannot collate an empty dual_franka SFT batch.")

    images = {
        key: torch.from_numpy(
            np.stack([np.asarray(item["image"][key]) for item in items])
        )
        for key in _IMAGE_KEYS
    }
    image_masks = {
        key: torch.from_numpy(
            np.stack(
                [np.asarray(item["image_mask"][key], dtype=np.bool_) for item in items]
            )
        )
        for key in _IMAGE_KEYS
    }
    batch = {
        "image": images,
        "image_mask": image_masks,
        "state": torch.from_numpy(
            np.stack([np.asarray(item["state"], dtype=np.float32) for item in items])
        ),
        "tokenized_prompt": torch.from_numpy(
            np.stack(
                [np.asarray(item["tokenized_prompt"], dtype=np.int64) for item in items]
            )
        ).long(),
        "tokenized_prompt_mask": torch.from_numpy(
            np.stack(
                [
                    np.asarray(item["tokenized_prompt_mask"], dtype=np.bool_)
                    for item in items
                ]
            )
        ),
    }
    actions = torch.from_numpy(
        np.stack([np.asarray(item["actions"], dtype=np.float32) for item in items])
    )
    return Observation.from_dict(batch), actions


@dataclasses.dataclass(frozen=True)
class DualFrankaSftDataConfig:
    """Resolved metadata exposed by :class:`DualFrankaSftDataLoader`."""

    repo_id: str
    action_dim: int
    action_horizon: int
    max_token_len: int


def _worker_init_fn(worker_id: int) -> None:
    del worker_id


def create_dual_franka_sft_data_loader(
    *,
    data_path: str,
    model_path: str,
    config_name: str,
    assets_dir: str,
    asset_id: str,
    action_dim: int,
    action_horizon: int,
    max_token_len: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    fps: int | None,
    tolerance_s: float,
    dist_rank: int,
    dist_world_size: int,
    data_kwargs: dict | None = None,
) -> "DualFrankaSftDataLoader":
    """Build a dual-franka SFT loader yielding ``(Observation, actions)``."""
    dataset = DualFrankaSftDataset(
        data_path=data_path,
        action_horizon=action_horizon,
        fps=fps,
        tolerance_s=tolerance_s,
    )

    input_transforms, _ = build_openpi_transforms(
        model_path,
        config_name,
        data_kwargs=data_kwargs,
        norm_stats_dir=assets_dir,
        norm_stats_asset_id=asset_id,
    )
    source = _TransformedDataset(
        dataset,
        build_dual_franka_sft_transform(input_transforms),
    )

    sampler = DistributedSampler(
        source,
        num_replicas=dist_world_size,
        rank=dist_rank,
        shuffle=shuffle,
        seed=seed,
        drop_last=True,
    )

    mp_context = multiprocessing.get_context("spawn") if num_workers > 0 else None
    generator = torch.Generator()
    generator.manual_seed(seed)

    logger.info(
        "dual_franka SFT data loader: batch_size=%d, num_workers=%d, "
        "action_horizon=%d, transforms=%s",
        batch_size,
        num_workers,
        action_horizon,
        config_name,
    )

    torch_loader = torch.utils.data.DataLoader(
        typing.cast(torch.utils.data.Dataset, source),
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        multiprocessing_context=mp_context,
        persistent_workers=num_workers > 0,
        collate_fn=collate_dual_franka_sft_items,
        worker_init_fn=_worker_init_fn,
        drop_last=True,
        generator=generator,
        pin_memory=True,
    )

    data_config = DualFrankaSftDataConfig(
        repo_id=str(dataset.dataset_root),
        action_dim=action_dim,
        action_horizon=action_horizon,
        max_token_len=max_token_len,
    )
    return DualFrankaSftDataLoader(torch_loader, data_config, sampler)


class DualFrankaSftDataLoader:
    """Infinite ``(Observation, actions)`` loop over dual-franka SFT data."""

    def __init__(
        self,
        torch_loader: torch.utils.data.DataLoader,
        data_config: DualFrankaSftDataConfig,
        sampler: DistributedSampler | None = None,
    ) -> None:
        self._torch_loader = torch_loader
        self._data_config = data_config
        self._sampler = sampler
        self._epoch = 0

    def data_config(self) -> DualFrankaSftDataConfig:
        """Return the resolved dual-Franka data configuration."""
        return self._data_config

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        """Return the underlying PyTorch dataloader."""
        return self._torch_loader

    def __iter__(
        self,
    ) -> typing.Iterator[tuple[Observation, torch.Tensor]]:
        while True:
            if self._sampler is not None:
                self._sampler.set_epoch(self._epoch)
                self._epoch += 1
            yield from self._torch_loader

    def __len__(self) -> int:
        return len(self._torch_loader)


def build_dual_franka_sft_dataloader(
    cfg: typing.Any,
    world_size: int,
    rank: int,
    data_paths: typing.Any,
    eval_dataset: bool = False,
) -> tuple[DualFrankaSftDataLoader, DualFrankaSftDataConfig]:
    """Build the dual-franka openpi_rlinf loader for the SFT worker."""
    from omegaconf import OmegaConf

    data_path = resolve_lerobot_repo_id(data_paths)
    if data_path is None:
        raise ValueError("openpi_rlinf dual_franka SFT requires data.train_data_paths.")

    model_cfg = cfg.actor.model
    data_cfg = cfg.data
    openpi_cfg = model_cfg.openpi

    # When train_data_paths names the LeRobot home, resolve the optional repo id
    # to the concrete dataset directory before constructing LeRobotDataset.
    repo_id = OmegaConf.select(data_cfg, "repo_id", default=None)
    if repo_id:
        from pathlib import Path

        from rlinf.data.storage.lerobot import default_hf_lerobot_home

        candidate = default_hf_lerobot_home() / str(repo_id)
        nested = Path(data_path) / str(repo_id)
        if (candidate / "meta" / "info.json").is_file():
            data_path = str(candidate)
        elif (nested / "meta" / "info.json").is_file():
            data_path = str(nested)

    data_kwargs = OmegaConf.select(model_cfg, "openpi_data", default=None)
    if data_kwargs is not None:
        data_kwargs = OmegaConf.to_container(data_kwargs, resolve=True)

    fps = OmegaConf.select(data_cfg, "fps", default=None)
    fps = int(fps) if fps is not None else None
    tolerance_s = float(OmegaConf.select(data_cfg, "tolerance_s", default=1e-4))

    loader = create_dual_franka_sft_data_loader(
        data_path=str(data_path),
        model_path=str(model_cfg.model_path),
        config_name=str(openpi_cfg.config_name),
        assets_dir=str(openpi_cfg.assets_dir),
        asset_id=str(openpi_cfg.asset_id),
        action_dim=int(openpi_cfg.model_action_dim),
        action_horizon=int(model_cfg.num_action_chunks),
        max_token_len=int(openpi_cfg.max_token_len),
        batch_size=(
            int(cfg.actor.eval_batch_size)
            if eval_dataset
            else int(cfg.actor.micro_batch_size)
        ),
        num_workers=int(data_cfg.num_workers),
        shuffle=not eval_dataset,
        seed=int(cfg.actor.seed),
        fps=fps,
        tolerance_s=tolerance_s,
        dist_rank=rank,
        dist_world_size=world_size,
        data_kwargs=typing.cast(dict | None, data_kwargs),
    )
    return loader, loader.data_config()
