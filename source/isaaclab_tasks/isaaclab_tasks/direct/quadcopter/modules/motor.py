import torch
from ..quadcopter_cfg import QuadcopterEnvCfg
class MotorModel:
    def __init__(self, num_envs, device, dt, domain_randomization_cfg:QuadcopterEnvCfg.domain_randomization.motor):
        self.domain_randomization_cfg = domain_randomization_cfg
        self.num_envs = num_envs
        self.device = device
        self.dt = dt
        # --------- IRIS ---------
        # self.rotor_param_dict = {
        #     "directions": torch.tensor([1.0, 1.0, -1.0, -1.0], device=self.device),
        #     "force_constants": torch.tensor([5.84e-06, 5.84e-06, 5.84e-06, 5.84e-06], device=self.device),
        #     "max_rotation_velocities": torch.tensor([1108.0, 1108.0, 1108.0, 1108.0], device=self.device),
        #     "moment_constants": torch.tensor([0.6, 0.6, 0.6, 0.6], device=self.device),
        #     "tau_up": torch.tensor([0.0125, 0.0125, 0.0125, 0.0125], device=self.device),
        #     "tau_down": torch.tensor([0.025, 0.025, 0.025, 0.025], device=self.device),
        # }
        # ----------UAV_LIDAR -------
        self.rotor_param_dict = {
            "directions": torch.tensor([1.0, 1.0, -1.0, -1.0], device=self.device),
            "force_constants": torch.tensor([4.33948e-07, 4.33948e-07, 4.33948e-07, 4.33948e-07], device=self.device),
            "max_rotation_velocities": torch.tensor([3800.0, 3800.0, 3800.0, 3800.0], device=self.device),
            "moment_constants": torch.tensor([0.00932, 0.00932, 0.00932, 0.00932], device=self.device),
            "tau_up": torch.tensor([0.0125, 0.0125, 0.0125, 0.0125], device=self.device),
            "tau_down": torch.tensor([0.025, 0.025, 0.025, 0.025], device=self.device),
        }
        self.rotor_directions = self.rotor_param_dict["directions"].repeat(self.num_envs, 1)
        self.rotor_force_constants = self.rotor_param_dict["force_constants"].repeat(self.num_envs, 1)
        self.rotor_max_rotation_velocities = self.rotor_param_dict["max_rotation_velocities"].repeat(self.num_envs, 1)
        self.rotor_moment_constants = self.rotor_param_dict["moment_constants"].repeat(self.num_envs, 1)
        self.rotor_tau_up = self.rotor_param_dict["tau_up"].repeat(self.num_envs, 1)
        self.rotor_tau_down = self.rotor_param_dict["tau_down"].repeat(self.num_envs, 1)
        self.rotor_thrust = torch.zeros(self.num_envs, 4, 3, device=self.device)
        self.rotor_moment = torch.zeros(self.num_envs, 4, 3, device=self.device)
        self.rotor_zero_moment = torch.zeros(self.num_envs, 4, 3, device=self.device)
        self.rotor_velocity = torch.zeros(self.num_envs, 4, device=self.device)
        self.rotor_commands = torch.zeros(self.num_envs, 4, device=self.device)

    def calculate_rotor_dynamic(self, cmds):
        target_velocity = torch.clamp(cmds, 0, 1) * self.rotor_max_rotation_velocities
        alpha_up = torch.exp(-self.dt / self.rotor_tau_up)
        alpha_down = torch.exp(-self.dt / self.rotor_tau_down)
        alpha = torch.where(target_velocity > self.rotor_velocity, alpha_up, alpha_down)
        self.rotor_velocity.add_(alpha * (target_velocity - self.rotor_velocity)).clamp_(torch.tensor(0., device=self.device), self.rotor_max_rotation_velocities)
        
        thrusts = self.rotor_velocity * abs(self.rotor_velocity) * self.rotor_force_constants
        moments = (thrusts * self.rotor_moment_constants) * -self.rotor_directions
        self.rotor_thrust[:, :, -1] = thrusts
        self.rotor_moment[:, :, -1] = moments

    def domain_random_percentage(self, value, percentage):
        return value * (1 + torch.rand_like(value, device=self.device) * 2 * percentage - percentage)

    def reset(self, env_ids=None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.rotor_velocity[env_ids] = torch.zeros_like(self.rotor_velocity[env_ids])
        self.rotor_thrust[env_ids] = torch.zeros_like(self.rotor_thrust[env_ids])
        self.rotor_moment[env_ids] = torch.zeros_like(self.rotor_moment[env_ids])
        self.rotor_commands[env_ids] = torch.zeros_like(self.rotor_commands[env_ids])
        
        # domain randomization
        if self.domain_randomization_cfg.enable:
            self.rotor_force_constants[env_ids] = self.domain_random_percentage(self.rotor_param_dict["force_constants"], self.domain_randomization_cfg.scale.force_constants)
            self.rotor_max_rotation_velocities[env_ids] = self.domain_random_percentage(self.rotor_param_dict["max_rotation_velocities"], self.domain_randomization_cfg.scale.max_rotation_velocities)
            self.rotor_moment_constants[env_ids] = self.domain_random_percentage(self.rotor_param_dict["moment_constants"], self.domain_randomization_cfg.scale.moment_constants)
            self.rotor_tau_up[env_ids] = self.domain_random_percentage(self.rotor_param_dict["tau_up"], self.domain_randomization_cfg.scale.tau_up)
            self.rotor_tau_down[env_ids] = self.domain_random_percentage(self.rotor_param_dict["tau_down"], self.domain_randomization_cfg.scale.tau_down)
        
        