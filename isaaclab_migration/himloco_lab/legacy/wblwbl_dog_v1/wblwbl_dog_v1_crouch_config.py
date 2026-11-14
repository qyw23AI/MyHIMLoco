"""Low-height training profile for the WBLWBL Dog V1."""

from .wblwbl_dog_v1_config import WblwblDogV1Cfg, WblwblDogV1CfgPPO


class WblwblDogV1CrouchCfg(WblwblDogV1Cfg):
    class rewards(WblwblDogV1Cfg.rewards):
        base_height_target = 0.10
        clearance_height_target = -0.13

        class scales(WblwblDogV1Cfg.rewards.scales):
            tracking_lin_vel = 1.5
            base_height = -5.0


class WblwblDogV1CrouchCfgPPO(WblwblDogV1CfgPPO):
    class runner(WblwblDogV1CfgPPO.runner):
        experiment_name = "wblwbl_dog_v1_crouch"
