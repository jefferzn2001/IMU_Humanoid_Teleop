# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""Simple joint sweep to verify actuator control in the environment.

This script opens the manager-based env (Play variant by default) and sends
sinusoidal joint position targets across ALL action dimensions so you can
visually confirm joints are moving in Isaac Sim.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import time

import numpy as np
import torch

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Joint sweep for actuator sanity check.")
parser.add_argument("--task", type=str, default="Template-Imu-G1-Teleoperation-Play-v0", help="Task ID to run.")
parser.add_argument("--amp", type=float, default=0.4, help="Action amplitude in [-1,1] (scaled inside env).")
parser.add_argument("--freq", type=float, default=0.2, help="Sine frequency in Hz.")
parser.add_argument("--duration", type=float, default=-1.0, help="Seconds to run the sweep (<=0 means infinite).")
parser.add_argument("--upper_arms_static", action="store_true", help="Freeze upper arm joints (no sweep).")
# append AppLauncher args (e.g., device, headless)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear hydra args
sys.argv = [sys.argv[0]] + hydra_args

# launch app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import IMU_G1_Teleoperation.tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(env_cfg, agent_cfg):  # agent_cfg unused
    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()

    dt = env.unwrapped.step_dt
    num_envs = env.unwrapped.num_envs
    # infer action dimension from gym space
    act_shape = env.action_space.shape  # typically (num_envs, act_dim)
    if len(act_shape) == 2:
        act_dim = act_shape[1]
    else:
        act_dim = act_shape[0]
    print(f"[INFO] num_envs={num_envs}, action_dim={act_dim}, dt={dt:.4f}s")

    t = 0.0
    start_time = time.time()

    # per-joint phase offsets for a nice wave (repeat pattern)
    phases = np.linspace(0.0, 2.0 * np.pi, num=act_dim, endpoint=False)

    # indices to freeze (upper arms) if requested
    robot = env.unwrapped.scene["robot"]
    upper_arm_expr = [
        ".*_shoulder_pitch_joint",
        ".*_shoulder_roll_joint",
        ".*_shoulder_yaw_joint",
        ".*_elbow_pitch_joint",
        ".*_elbow_roll_joint",
    ]
    freeze_mask = np.zeros(act_dim, dtype=bool)
    try:
        # if action maps directly to joint_names, we can mark by regex name membership
        joint_names = robot.joint_names
        if len(joint_names) == act_dim:
            import re

            for i, name in enumerate(joint_names):
                if any(re.match(expr, name) for expr in upper_arm_expr):
                    freeze_mask[i] = True
    except Exception:
        pass

    # resolve shoulder joint indices (pitch joints)
    robot = env.unwrapped.scene["robot"]
    try:
        left_pitch_ids, _ = robot.find_joints(["left_shoulder_pitch_joint"], preserve_order=True)
        left_roll_ids, _ = robot.find_joints(["left_shoulder_roll_joint"], preserve_order=True)
    except Exception:
        left_pitch_ids, left_roll_ids = [], []

    # attempt to locate action indices for these joints (when action covers all joints)
    pitch_act_idx = None
    roll_act_idx = None
    try:
        joint_names = robot.joint_names
        if len(joint_names) == act_dim:
            for i, n in enumerate(joint_names):
                if n == "left_shoulder_pitch_joint":
                    pitch_act_idx = i
                if n == "left_shoulder_roll_joint":
                    roll_act_idx = i
        print(f"[MAP] action_dim={act_dim}, pitch_idx={pitch_act_idx}, roll_idx={roll_act_idx}")
        if pitch_act_idx is None or roll_act_idx is None:
            shoulder_like = [n for n in joint_names if "shoulder" in n]
            print(f"[MAP] shoulder-like joints detected: {shoulder_like}")
    except Exception as e:
        print(f"[MAP] mapping detection failed: {e}")

    # keyboard setup (Omniverse) — event-based (no polling)
    # import after AppLauncher to ensure availability
    pressed = {}
    try:
        import carb
        import omni

        input_iface = carb.input.acquire_input_interface()
        appwin = omni.appwindow.get_default_app_window()
        keyboard = appwin.get_keyboard()
    except Exception:
        input_iface = None
        keyboard = None

    # offsets applied to left shoulder pitch/roll
    left_pitch_offset = 0.0
    left_roll_offset = 0.0

    # debug print on key events
    def _on_key(event, *args, **kwargs):
        nonlocal left_pitch_offset, left_roll_offset
        try:
            if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                pressed[event.input.name] = True
                print(f"[KEY] {event.input.name} pressed")
            if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
                pressed[event.input.name] = False
                print(f"[KEY] {event.input.name} released")
        except Exception:
            pass
        return True

    # subscribe minimal logger
    if input_iface is not None and keyboard is not None:
        _kbd_sub = input_iface.subscribe_to_keyboard_events(keyboard, _on_key)

    while simulation_app.is_running():
        elapsed = time.time() - start_time
        if args_cli.duration > 0.0 and elapsed >= args_cli.duration:
            break

        # sinusoid per joint
        theta = 2.0 * np.pi * args_cli.freq * t
        action_1d = args_cli.amp * np.sin(theta + phases)
        if args_cli.upper_arms_static:
            # fallback: if we recognize standard action layout (legs 9, feet 4, arms 24)
            if act_dim == 37:
                action_1d[13:] = 0.0
            else:
                action_1d[freeze_mask] = 0.0

        # apply shoulder offsets directly into action vector if indices known
        if pitch_act_idx is not None:
            action_1d[pitch_act_idx] += left_pitch_offset
        if roll_act_idx is not None:
            action_1d[roll_act_idx] += left_roll_offset
        # tile to all envs (num_envs, act_dim)
        actions = np.tile(action_1d.astype(np.float32), (num_envs, 1))

        # step (env expects torch tensor on sim device)
        actions_t = torch.from_numpy(actions).to(env.unwrapped.device)
        obs, _, _, _, _ = env.step(actions_t)

        # arrow/numpad controls for left shoulder (event-based state)
        if input_iface is not None and keyboard is not None:
            step = 0.03
            up = pressed.get("UP", False) or pressed.get("NUMPAD_8", False)
            down = pressed.get("DOWN", False) or pressed.get("NUMPAD_2", False)
            left = pressed.get("LEFT", False) or pressed.get("NUMPAD_4", False)
            right = pressed.get("RIGHT", False) or pressed.get("NUMPAD_6", False)

            if up:
                left_pitch_offset += step
                print(f"[CMD] left_shoulder_pitch += {step:.3f} -> {left_pitch_offset:.3f}")
            if down:
                left_pitch_offset -= step
                print(f"[CMD] left_shoulder_pitch -= {step:.3f} -> {left_pitch_offset:.3f}")
            if left:
                left_roll_offset += step
                print(f"[CMD] left_shoulder_roll += {step:.3f} -> {left_roll_offset:.3f}")
            if right:
                left_roll_offset -= step
                print(f"[CMD] left_shoulder_roll -= {step:.3f} -> {left_roll_offset:.3f}")

            # always print current joint positions for visibility
            if len(left_pitch_ids) > 0:
                curr = robot.data.joint_pos[:, left_pitch_ids]
                print(f"[JOINT] left_shoulder_pitch curr={curr[0, 0].item():.3f}")
            if len(left_roll_ids) > 0:
                curr = robot.data.joint_pos[:, left_roll_ids]
                print(f"[JOINT] left_shoulder_roll  curr={curr[0, 0].item():.3f}")

        # real-time pacing
        t += dt
        sleep_time = dt - (time.time() - start_time - elapsed)
        if sleep_time > 0.0 and args_cli.device is None:  # if not forcing a different device cadence
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()


