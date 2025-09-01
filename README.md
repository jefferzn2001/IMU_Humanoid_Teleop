# IMU G1 Teleoperation

Compact Isaac Lab project for IMU-driven teleoperation of a G1 humanoid. The operator’s body motion is mapped from wearable IMUs to the robot, with a learning-assisted controller providing stabilization and smooth control. This repository is packaged as an Isaac Lab extension for rapid iteration.

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

## Repository

- `scripts/`: utilities for listing environments, dummy agents, and rl_games entry points.
- `source/IMU_G1_Teleoperation/`: Isaac Lab extension with tasks and configuration.

Status: active development; APIs and configs may change.