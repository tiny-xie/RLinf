Collect Dual-Arm YAM Demonstrations
===================================

Use RLinf's native YAM environment to teleoperate two follower arms from two
motorized leader arms and write 14-D joint-space demonstrations directly in
LeRobot format. The runtime depends on RLinf plus a pinned, compatible ``i2rt``
build; it does not require a checkout of a separate YAM application repository.

.. warning::

   The collection command on this page opens three cameras and four SocketCAN
   devices on the first environment reset, and it can move both follower arms.
   Clear the workspace, make the physical emergency stop reachable, release both
   teaching-handle buttons, and keep a trained operator beside the station.
   Reading this page or running the unit-test command below does not open hardware.

Overview
--------

The current example covers native environment wiring and demonstration
collection. YAM-specific policy transforms, SFT, and deployment configs are a
separate integration stage.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      No policy required for collection

   .. grid-item-card:: Algorithms
      :text-align: center

      Leader-follower teleoperation · DAgger-ready intervention API

   .. grid-item-card:: Tasks
      :text-align: center

      Dual-arm joint-space demonstrations

   .. grid-item-card:: Hardware
      :text-align: center

      2 followers · 2 motorized leaders · 3 RealSense cameras

| **You'll do:** install the runtime → configure one YAM station → export calibration values → collect successful episodes → inspect both RLinf and LeRobot outputs.
| **Prerequisites:** :doc:`Installation <../../start/installation>` · Linux SocketCAN · three Intel RealSense cameras · a validated ``i2rt`` build · a physical emergency stop.

How RLinf Organizes the Station
-------------------------------

RLinf keeps resource scheduling separate from device control:

.. code-block:: text

   Hydra YAML
     -> NodeHardwareConfig(type="DualYam")
     -> DualYamConfig
     -> DualYamRobot.enumerate()             # configuration only
     -> one DualYamHWInfo per complete station
     -> component_placement selects hardware rank 0
     -> WorkerInfo.hardware_infos
     -> RealWorldEnv
     -> create_dual_yam_joint_env()
     -> optional DualYamLeaderIntervention
     -> DualYamJointEnv
     -> YamControlRuntime                    # only follower-command writer
     -> lazy i2rt backend

One ``DualYamConfig`` describes one complete station: the left/right followers,
left/right leaders, and every camera. It becomes one scheduler hardware rank.
Therefore ``placement: 0`` means "the first complete YAM station"; it does not
mean node rank 0 or CAN interface ``can0``. See
:doc:`Placement <../../concepts/placement>` for the distinction between node and
hardware ranks.

The scheduler only validates configuration and resource ownership. It does not
import ``i2rt``, probe a CAN bus, or open a camera. The assigned environment
worker receives ``DualYamHWInfo`` through ``WorkerInfo``. Real hardware is opened
later, during the first ``reset()``:

.. code-block:: text

   open all cameras -> warm every camera -> connect left follower -> hold
                    -> connect right follower -> hold -> validate feedback
                    -> connect leaders only when intervention is enabled

Policy-only use leaves leader intervention disabled and never opens either
leader CAN interface. The collection config enables the leader wrapper. All
four CAN interfaces and all cameras must currently be accessible from the same
node as the environment worker.

Configuration Boundaries
~~~~~~~~~~~~~~~~~~~~~~~~

Keep three kinds of settings in their respective YAML sections:

.. list-table::
   :header-rows: 1
   :widths: 34 30 36

   * - Configuration
     - Location
     - Owns
   * - Physical station
     - ``cluster.node_groups[].hardware``
     - Node ownership, CAN names, arm/gripper variants, per-device mass and compensation, raw gripper stops, and camera serials.
   * - Task/runtime
     - ``env.eval.override_cfg``
     - Task text, control frequency, RLinf joint/slew limits, timeouts, image size, and leader episode behavior.
   * - Process placement
     - ``cluster.component_placement``
     - Which worker exclusively receives a complete station resource.

Installation
------------

Create the YAM embodied environment using RLinf's standard installer:

.. code-block:: bash

   bash requirements/install.sh embodied --env yam
   source .venv/bin/activate

The installer adds RLinf's embodied dependencies, RealSense/OpenCV, LeRobot,
and the pinned official ``i2rt`` SDK declared in
``requirements/embodied/envs/yam.txt``. It does not require or clone a YAM
application repository, and there is no separate wheel variable or SDK install
path. Modules still import ``i2rt`` lazily, so scheduler-only and dummy imports
remain hardware-free.

The compatible build must support the options used by
``get_yam_robot()`` and must shut down its CAN/control threads reliably. The
adapter deliberately refuses unsupported per-device numeric damping or friction
overrides instead of mutating private SDK arrays.

Before launching RLinf, configure persistent SocketCAN names for the four USB-CAN
adapters and bring those interfaces up using your machine's normal system setup.
The example defaults are:

.. list-table::
   :header-rows: 1
   :widths: 30 28 42

   * - Role
     - Default interface
     - Optional override
   * - Left follower
     - ``can_left``
     - ``YAM_LEFT_FOLLOWER_CAN``
   * - Right follower
     - ``can_right``
     - ``YAM_RIGHT_FOLLOWER_CAN``
   * - Left leader
     - ``can_lead_l``
     - ``YAM_LEFT_LEADER_CAN``
   * - Right leader
     - ``can_lead_r``
     - ``YAM_RIGHT_LEADER_CAN``

You can inspect an interface without commanding a robot:

.. code-block:: bash

   ip -details link show can_left
   ip -details link show can_right
   ip -details link show can_lead_l
   ip -details link show can_lead_r

Configure the Station
---------------------

The reusable environment defaults are in
``examples/embodiment/config/env/realworld_dual_yam_joint.yaml``. The complete
single-node collection example is
``examples/embodiment/config/realworld_dual_yam_collect_data.yaml``.

Export the required per-station values before starting collection:

.. code-block:: bash

   # Raw follower motor positions, ordered [closed, open].
   # The values may be ascending or descending; do not sort them.
   export YAM_LEFT_GRIPPER_CLOSED_RAD=<measured-left-closed>
   export YAM_LEFT_GRIPPER_OPEN_RAD=<measured-left-open>
   export YAM_RIGHT_GRIPPER_CLOSED_RAD=<measured-right-closed>
   export YAM_RIGHT_GRIPPER_OPEN_RAD=<measured-right-open>

   # RealSense serial numbers.
   export YAM_TOP_CAMERA_SERIAL=<top-serial>
   export YAM_LEFT_CAMERA_SERIAL=<left-serial>
   export YAM_RIGHT_CAMERA_SERIAL=<right-serial>

Do not estimate the gripper stops. They are raw motor-radian endpoints, not the
normalized ``[0, 1]`` action. Their order carries the motor direction, so a
valid ``[closed, open]`` pair can be decreasing.

If your CAN aliases differ from the defaults, export the four optional variables
from the table above. All four resolved names must be unique. Camera names and
serials must also be unique.

Per-device tuning is kept beside each device in the hardware configuration:

.. code-block:: yaml

   left_leader:
     ee_mass: null                 # use the pinned i2rt model
     gravity_comp_factor: null     # use the selected arm model's defaults
     grav_comp_kd: null
     coulomb_friction: null
     use_coulomb_friction: false
     bilateral_kp: 0.0
     gripper_invert: false

``ee_mass`` and ``gravity_comp_factor`` affect gravity support;
``grav_comp_kd`` is gravity-compensation damping; ``coulomb_friction`` and
``use_coulomb_friction`` control Coulomb-friction compensation.
``bilateral_kp`` instead controls leader feedback toward the measured follower
pose and is not a gravity-compensation setting. Keep it at ``0.0`` for initial
collection. Leaving an optional value at ``null`` preserves the pinned SDK
model. Numeric ``grav_comp_kd`` and ``coulomb_friction`` values require an SDK
constructor that explicitly accepts those fields.

The base config's joint limits are the nominal YAM v1 limits. Replace them with
a narrower, validated workspace envelope when the station or task requires it.
For another YAM arm variant, replace both ``arm_type`` and the RLinf joint limits;
startup rejects configured limits outside the SDK's limits.

Validate Without Hardware
-------------------------

Run the YAM unit tests before a supervised hardware session. They use mocks and
must not open CAN or cameras:

.. code-block:: bash

   pytest -q \
     tests/unit_tests/test_yam_hardware.py \
     tests/unit_tests/test_yam_runtime.py \
     tests/unit_tests/test_yam_env.py \
     tests/unit_tests/test_yam_intervention.py \
     tests/unit_tests/test_yam_imports.py \
     tests/unit_tests/test_yam_examples.py

This checks registration, configuration conversion, 14-D ordering, joint and
slew limiting, stale/non-finite handling, intervention ownership, cleanup, and
lazy ``i2rt`` imports. It is not a substitute for low-speed hardware acceptance.

Collect Demonstrations
----------------------

.. danger::

   The next command is the hardware launch point. On its first ``reset()``, it
   opens cameras, connects both followers, and then connects both leaders.
   Do not run it as a configuration-only check.

From the RLinf repository root, start a 50-episode collection:

.. code-block:: bash

   bash examples/embodiment/collect_data.sh \
     realworld_dual_yam_collect_data

The supplied config collects 50 ``pick_block`` episodes. To keep the launcher in
RLinf's existing config-name style, copy or edit the following fields in
``realworld_dual_yam_collect_data.yaml`` when creating another recipe:

.. code-block:: yaml

   runner:
     num_data_episodes: 50
   env:
     eval:
       override_cfg:
         task_description: pick_block

What this does:

1. launches RLinf's generic ``collect_real_data.py`` entry point;
2. asks the scheduler for one complete ``DualYam`` resource;
3. constructs ``RealWorldEnv`` and the registered ``DualYamJointEnv-v1`` task;
4. enables motorized-leader intervention and button-controlled episodes;
5. writes successful demonstrations directly to RLinf replay and LeRobot data.

There is no ``--convert`` step and no runtime clone/import of a YAM application
repository.

Teaching-Handle Controls
~~~~~~~~~~~~~~~~~~~~~~~~

Release both buttons while leaders connect. The first sample establishes the
idle electrical level for each handle. A button on either handle controls the
whole dual-arm station.

.. list-table::
   :header-rows: 1
   :widths: 25 32 43

   * - Control
     - State
     - Result
   * - Top / first button
     - Synchronization off
     - Smoothly engage from the measured follower pose to the current leader pose, then let both leaders command both followers.
   * - Top / first button
     - Synchronization on
     - Hold both followers and return both leaders to gravity-compensation idle.
   * - Record / second button
     - Waiting before an episode
     - Start a new recorded episode at the current pose.
   * - Record / second button
     - Recording
     - End the episode as a success with reward ``1`` and release active bilateral feedback.
   * - Teaching trigger
     - Default mapping
     - Released is gripper ``1`` (open); pressed is ``0`` (closed). Set ``gripper_invert: true`` per leader to reverse it.

The example uses ``sync_on_reset: false``: press the top button when you are
ready to take control. It also uses ``unsynced_action_source: hold``, so the
collector's placeholder zero action can never send the followers toward zero.
Button events are rising-edge-triggered and debounced.

Observation and Action Contract
-------------------------------

Every state and action uses the same absolute 14-D layout:

.. code-block:: text

   [left_q0, ..., left_q5, left_gripper,
    right_q0, ..., right_q5, right_gripper]

Arm joints are radians. Grippers are normalized to ``0=closed, 1=open``.
Observations contain measured follower positions. Before dispatch, commands are
checked for the exact shape and finite values, clipped to configured hard limits,
limited by ``max_joint_delta`` relative to the measured pose, and clipped to the
gripper range. During collection, the wrapper reports the accepted target as
``intervene_action`` so the recorded expert action matches what RLinf accepted.

Camera and dataset names change at three deliberate boundaries:

.. list-table::
   :header-rows: 1
   :widths: 24 36 40

   * - Boundary
     - Keys
     - Meaning
   * - YAM Gym observation
     - ``frames.top_rgb``, ``frames.left_rgb``, ``frames.right_rgb``
     - Named RGB frames; ``state.joint_position`` is the 14-D measured state.
   * - ``RealWorldEnv``
     - ``main_images``, ``extra_view_images``
     - ``top_rgb`` becomes the main image. Remaining names are sorted, so index 0 is ``left_rgb`` and index 1 is ``right_rgb`` for the supplied config.
   * - RLinf LeRobot writer
     - ``image``, ``extra_view_image-0``, ``extra_view_image-1``
     - Literal dataset feature names for top, left, and right respectively. State/action features are ``state`` and ``actions``.

Keep the supplied camera names if downstream transforms depend on that ordering.
The current generic collector preserves view order but not the semantic
``left_rgb``/``right_rgb`` names in the final LeRobot columns. A future YAM
policy dataconfig must map the literal dataset keys explicitly.

Output Layout
-------------

``collect_data.sh`` creates a fresh ``logs/<timestamp>/`` directory. The same
successful episode is written to two destinations:

.. code-block:: text

   logs/<timestamp>/
   |-- demos/                         # RLinf TrajectoryReplayBuffer (.pt)
   `-- collected_data/
       `-- rank_0/
           `-- id_0/                 # LeRobot shard for this run
               |-- meta/info.json
               |-- meta/episodes.jsonl
               |-- meta/tasks.jsonl
               |-- meta/stats.json
               |-- data/...
               `-- videos/...        # layout depends on the installed LeRobot version

The example uses ``finalize_interval: 0`` so all episodes stay in one shard and
metadata is finalized during a clean shutdown. Set ``resume: true`` only when
reusing an explicit ``save_dir``; a resumed run writes a new ``id_N`` shard and
does not overwrite finalized shards.

The LeRobot frames include ``state``, ``actions``, ``image``,
``extra_view_image-0``, ``extra_view_image-1``, ``done``, ``is_success``,
``intervene_flag``, and ``segment_id``. The task string is stored through
LeRobot's task metadata. See :doc:`Data Collection <../../guides/data_collection>`
for the generic writer behavior.

Implementation Map
------------------

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - File
     - Responsibility
   * - ``rlinf/scheduler/hardware/robots/dual_yam.py``
     - Defines one complete station resource, converts nested Hydra config, validates CAN/camera uniqueness, and enumerates without hardware I/O.
   * - ``rlinf/envs/realworld/yam/types.py``
     - Defines the 14-D state/action contract, typed state objects, command results, and backend protocols.
   * - ``rlinf/envs/realworld/yam/config.py``
     - Validates task-level timing, limits, camera timeouts, and leader-intervention behavior.
   * - ``rlinf/envs/realworld/yam/i2rt_backend.py``
     - Isolates the lazy ``i2rt`` import and adapts follower, leader, handle, health, and cleanup APIs.
   * - ``rlinf/envs/realworld/yam/mock_backend.py``
     - Supplies hardware-free follower/leader implementations for dummy use and tests.
   * - ``rlinf/envs/realworld/yam/control_runtime.py``
     - Owns all transports, serializes follower writes, validates every command, engages leaders smoothly, holds on failures, and closes in safe order.
   * - ``rlinf/envs/realworld/yam/dual_yam_joint_env.py``
     - Implements the Gym action/observation spaces, lazy startup, camera processing, step pacing, and resource cleanup.
   * - ``rlinf/envs/realworld/yam/leader_intervention.py``
     - Implements dual-leader synchronization, buttons, episode control, policy/hold ownership, and ``intervene_action`` reporting.
   * - ``rlinf/envs/realworld/yam/tasks/__init__.py``
     - Registers ``DualYamJointEnv-v1``, validates ``main_image_key``, and installs the optional intervention wrapper.
   * - ``rlinf/envs/realworld/yam/__init__.py``
     - Exposes the public YAM API and imports the task registration.
   * - ``examples/embodiment/config/env/realworld_dual_yam_joint.yaml``
     - Reusable real-world Gym/task defaults and explicit RLinf safety limits.
   * - ``examples/embodiment/config/realworld_dual_yam_collect_data.yaml``
     - Complete one-station scheduler, teleoperation, and direct LeRobot collection recipe.
   * - ``requirements/install.sh`` (``--env yam``)
     - Builds one complete YAM environment, including camera, LeRobot, and the pinned i2rt SDK, without requiring a YAM application repository.
   * - ``requirements/embodied/envs/yam.txt``
     - Pins the official i2rt commit and declares the native runtime's camera and configuration dependencies.
   * - ``requirements/embodied/envs/yam-build-constraints.txt``
     - Keeps i2rt's ruckig source-build constraint local to the YAM environment.
   * - ``examples/embodiment/collect_data.sh``
     - Existing, unchanged generic collection launcher used by the YAM recipe; it selects the config by name and creates the timestamped log directory.
   * - ``rlinf/envs/realworld/__init__.py``
     - Imports the YAM task package so Gym registration is available through RLinf's real-world environment entry point.
   * - ``rlinf/scheduler/__init__.py``, ``rlinf/scheduler/hardware/__init__.py``, and ``rlinf/scheduler/hardware/robots/__init__.py``
     - Re-export the YAM scheduler types and load ``DualYam`` into the hardware-policy registry.
   * - ``tests/unit_tests/test_yam_hardware.py``
     - Covers registry conversion, station enumeration, ownership conflicts, and direction-preserving gripper calibration.
   * - ``tests/unit_tests/test_yam_runtime.py``
     - Covers lazy connection, command safety, stale feedback, role-specific i2rt modes, and cleanup.
   * - ``tests/unit_tests/test_yam_env.py``
     - Covers the dummy Gym observation/action contract, 14-D ordering, and idempotent close.
   * - ``tests/unit_tests/test_yam_intervention.py``
     - Covers policy/leader ownership, synchronization failure cleanup, and episode termination behavior.
   * - ``tests/unit_tests/test_yam_imports.py``
     - Guards the hardware-free import path and verifies that ``i2rt`` remains lazy.
   * - ``tests/unit_tests/test_yam_examples.py``
     - Guards the public YAML contract, direct LeRobot settings, and bundled pinned-SDK/no-application-repository installation rule.

Known Limits
------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Area
     - Current boundary
   * - Control representation
     - Absolute 14-D joint space only. TCP/Cartesian actions require a separately versioned environment and dataset contract.
   * - Cameras
     - RealSense RGB only; ``enable_depth`` must remain ``false``.
   * - Placement
     - Followers, motorized leaders, and cameras must be local to one environment worker node. Remote per-arm controllers are not implemented.
   * - Intervention
     - Synchronization and intervention apply to both arms together. There is no independent left/right intervention mask yet.
   * - Models
     - This example collects data. A YAM-specific model dataconfig, normalization statistics, SFT recipe, and policy deployment example are not included yet.
   * - SDK tuning
     - Gravity factor and friction enablement use public SDK options. Numeric per-device damping/friction overrides require a compatible ``i2rt`` constructor and are rejected otherwise.
   * - Failure response
     - Software performs best-effort measured-pose hold and cleanup; it is not a physical emergency stop.
   * - Validation
     - Unit tests use mocks. Validate the pinned SDK and station at low speed, one arm at a time, before bimanual collection.
