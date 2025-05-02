from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg

import os

QUADCOPTER_ASSET_PATH = os.path.dirname(__file__)
QUADCOPTER_USD_PATH = os.path.join(QUADCOPTER_ASSET_PATH, "racing_quadcopter_v1.usd")
COPTER_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=QUADCOPTER_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
        activate_contact_sensors=True,
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(
            drive_type="force",
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
    ),
    actuators={
        "dummy": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
