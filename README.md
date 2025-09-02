# IMU G1 Teleoperation

Compact Isaac Lab project for IMU-driven teleoperation of a G1 humanoid. The operator’s body motion is mapped from wearable IMUs to the robot, with a learning-assisted controller providing stabilization and smooth control. This repository is packaged as an Isaac Lab extension for rapid iteration.

## Recent updates

- Added Play env variant: flat ground, centered spawn, no auto-resets (viewer-friendly).
- Added `scripts/joint_sweep.py` to verify joint actuation and test live key input (arrows/numpad).
- Training config and rewards tuned; supports TensorBoard for live curves.

## Quick Start

- Install Isaac Lab by following the official installation guide: [Isaac Lab Installation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
- Install this project (editable mode) using a Python interpreter that has Isaac Lab available:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not in your active env
    python -m pip install -e source/IMU_G1_Teleoperation
    ```

- List available tasks:

    ```bash
    python scripts/list_envs.py
    ```

- Try a task with dummy agents to validate setup:

    ```bash
    python scripts/zero_agent.py --task=<TASK_NAME>
    python scripts/random_agent.py --task=<TASK_NAME>
    ```

- Train or play using rl_games:

    ```bash
    python scripts/rl_games/train.py --task=<TASK_NAME>
    python scripts/rl_games/play.py --task=<TASK_NAME>
    ```

- Joint sweep (flat ground, centered, infinite run):

    ```bash
    # viewer sanity-check of joint control
    python scripts/joint_sweep.py --task=Template-Imu-G1-Teleoperation-Play-v0 --duration -1 --upper_arms_static
    ```

- TensorBoard (optional):

    ```bash
    tensorboard --logdir logs/rl_games
    ```

## Repository

- `scripts/`: utilities for listing environments, dummy agents, and rl_games entry points.
- `source/IMU_G1_Teleoperation/`: Isaac Lab extension with tasks and configuration.

Status: active development; APIs and configs may change.

## Known notes

- In early tests, some “shoulder” joint name guesses mapped to leg joints. Use `scripts/joint_sweep.py` logs ([MAP], [JOINT]) to confirm indices on your setup.
- Play env (`Template-Imu-G1-Teleoperation-Play-v0`) uses flat ground, centered spawn, and disables time-out and fall resets for interactive sessions.

## TODO (post-push)

- Refine reward shaping: larger tracking std, add upright term, soften early penalties.
- Expand velocity commands to full XY+yaw with a short curriculum; ramp difficulty.
- Verify and correct joint name → action index mapping for arms; expose a small mapping helper.
- Optional torque-mode action for experimentation (keep position control default).
- IMU device input path and simple teleop bridge.
- Gamepad/keyboard teleop for selected joints with clear on-screen or console feedback.