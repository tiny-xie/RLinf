# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Mark current LIBERO scenes using simulator-grounded source and target boxes."""

from typing import Any

import numpy as np

from rlinf.utils.visual_prompt import render_g1_prompt


def visible_object_box(mask: np.ndarray) -> list[int]:
    """Bound visible object components, discarding segmentation speckles.

    Retain components at least one percent as large as the largest component,
    with a minimum of two pixels, then pad by two pixels. Separate visible
    portions of a mug (such as its handle) remain part of the box.
    """
    from scipy.ndimage import label

    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("Object masks must be two-dimensional boolean arrays.")
    components, _ = label(mask, structure=np.ones((3, 3), dtype=bool))
    counts = np.bincount(components.ravel())
    counts[0] = 0
    threshold = max(2, int(np.ceil(counts.max() * 0.01)))
    retained = np.flatnonzero(counts >= threshold)
    ys, xs = np.nonzero(np.isin(components, retained))
    if not len(xs):
        raise ValueError(
            "No visible object component remains after filtering speckles."
        )
    height, width = mask.shape
    return [
        max(0, int(xs.min()) - 2),
        max(0, int(ys.min()) - 2),
        min(width - 1, int(xs.max()) + 2),
        min(height - 1, int(ys.max()) + 2),
    ]


def render_g1_subgoal_image(env: Any, predicate: tuple[str, str, str]) -> np.ndarray:
    """Mark the current transfer without moving the robot or scene objects.

    Boxes enclose visible object geoms, with two pixels of padding. Simulator
    segmentation provides oracle object localization; no detector is evaluated.
    Outlines are drawn in policy image coordinates, then returned in LIBERO's
    raw camera convention for the existing observation export.
    """
    import mujoco
    from robosuite import macros
    from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING

    from rlinf.envs.sim.libero.utils import get_libero_image

    if len(predicate) != 3 or predicate[0].lower() != "on":
        raise ValueError("G1 requires an On(source, target) subgoal.")
    rob = env.env
    env._post_process()
    env._update_observables(force=True)
    obs = rob._get_observations()
    if "agentview_image" not in obs:
        raise ValueError("G1 requires an enabled agentview RGB camera.")
    rgb = get_libero_image(obs)
    height, width = rgb.shape[:2]
    sim = env.sim
    segmentation = sim.render(
        width=width, height=height, camera_name="agentview", segmentation=True
    )
    # Match the RGB sensor's convention before applying RLinf's 180-degree turn.
    convention = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
    segmentation = segmentation[::convention][::-1, ::-1]
    model = sim.model
    boxes = []
    for name in predicate[1:]:
        if name not in rob.objects_dict:
            raise ValueError(f"G1 object {name!r} is not present in this scene.")
        root = model.body_name2id(rob.objects_dict[name].root_body)
        bodies = {root}
        for body in range(model.nbody):
            parent = body
            while parent and parent not in bodies:
                parent = int(model.body_parentid[parent])
            if parent in bodies:
                bodies.add(body)
        geoms = np.flatnonzero(np.isin(model.geom_bodyid, list(bodies)))
        mask = (segmentation[..., 0] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(
            segmentation[..., 1], geoms
        )
        try:
            boxes.append(visible_object_box(mask))
        except ValueError as exc:
            raise ValueError(
                f"G1 object {name!r} is not visible in agentview."
            ) from exc
    marked = render_g1_prompt(rgb, boxes[0], boxes[1])
    return marked[::-1, ::-1].copy()
