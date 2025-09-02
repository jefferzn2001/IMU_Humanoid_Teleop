# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab.envs import mdp

##
# Pre-defined configs
##

from IMU_G1_Teleoperation.robots import G1_CFG
from isaaclab.envs.mdp import JointPositionActionCfg, UniformVelocityCommandCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
##
# Scene definition
##


@configclass
class ImuG1TeleoperationSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # default: provide both entries; Play variant disables terrain and uses ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    flat_ground = AssetBaseCfg(
        prim_path="/World/FlatGround",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robot
    robot: ArticulationCfg = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/G1")
    
    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Map normalized actions to G1 joints (position targets)."""

    # Legs + torso
    legs = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            ".*_hip_yaw_joint",
            ".*_hip_roll_joint",
            ".*_hip_pitch_joint",
            ".*_knee_joint",
            "torso_joint",
        ],
        scale=0.5,
        use_default_offset=True,
    )

    # Ankles
    feet = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            ".*_ankle_pitch_joint",
            ".*_ankle_roll_joint",
        ],
        scale=0.5,
        use_default_offset=True,
    )

    # Arms + fingers
    arms = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            ".*_shoulder_pitch_joint",
            ".*_shoulder_roll_joint",
            ".*_shoulder_yaw_joint",
            ".*_elbow_pitch_joint",
            ".*_elbow_roll_joint",
            ".*_five_joint",
            ".*_three_joint",
            ".*_six_joint",
            ".*_four_joint",
            ".*_zero_joint",
            ".*_one_joint",
            ".*_two_joint",
        ],
        scale=0.5,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
        )
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        up_proj = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        # exposes sampled commands (so policy can see desired vx, vy if you want)
        commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    stand_reset = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-0.05, 0.05),
            },
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (-0.1, 0.1),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.05, 0.05),
            },
        },
    )

    zeroish_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "position_range": (-0.02, 0.02),
            "velocity_range": (-0.02, 0.02),
        },
    )

@configclass
class RewardsCfg:
    # Track commanded forward velocity (vx, vy) — use exponential kernel
    track_lin = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={
            "std": 0.25,
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Keep stable/upright-ish
    ang_smooth = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    vertical_v = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    track_ang = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.25,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # Regularizers
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_speed = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.002,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fell_low = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.55, "asset_cfg": SceneEntityCfg("robot")},
    ) 


##
# Environment configuration
##


@configclass
class CommandsCfg:
    base_velocity = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 5.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-2.0, 2.0),
        ),
    )


@configclass
class ImuG1TeleoperationEnvCfg(ManagerBasedRLEnvCfg):
    scene: ImuG1TeleoperationSceneCfg = ImuG1TeleoperationSceneCfg(num_envs=512, env_spacing=4.0)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: CommandsCfg = CommandsCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 10.0
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        # align sim material with terrain
        if getattr(self.scene, "terrain", None) is not None:
            self.sim.physics_material = self.scene.terrain.physics_material


@configclass
class ImuG1TeleoperationEnvCfg_PLAY(ImuG1TeleoperationEnvCfg):
    """Play variant: single env, relaxed resets, viewer-friendly."""

    def __post_init__(self) -> None:
        super().__post_init__()
        # smaller scene for play
        self.scene.num_envs = 1
        self.scene.env_spacing = 0.0
        # disable obs corruption for clean viewing
        self.observations.policy.enable_corruption = False
        # relax terminations: avoid rapid resets on fall
        self.terminations.fell_low = None
        # disable time-out resets for interactive sessions
        self.terminations.time_out = None
        # bring camera closer to the robot at center
        self.viewer.eye = (5.0, 0.0, 2.0)
        # use flat ground only
        self.scene.terrain = None
        # soften vertical velocity penalty to reduce twitchiness when recovering
        if hasattr(self.rewards, "vertical_v"):
            self.rewards.vertical_v.weight = -0.02