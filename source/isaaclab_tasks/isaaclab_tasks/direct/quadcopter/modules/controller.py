import torch

def deg2rad(deg):
    return deg / 180.0 * 3.14159265358979323846

class RateController:
    def __init__(self, num_envs, device):
        self.num_envs = num_envs
        self.device = device
        self.prev_rate_error = torch.zeros(self.num_envs, 3, device=self.device)
        self._rate_int = torch.zeros(self.num_envs, 3, device=self.device)
        self._lim_int = torch.tensor([0.3, 0.3, 0.3], device=self.device)
        
        # iris
        # self._gain_p = torch.tensor([0.15, 0.15, 0.2], device=self.device)
        # self._gain_i = torch.tensor([0.2, 0.2, 0.1], device=self.device)
        # self._gain_d = torch.tensor([0.003, 0.003, 0.0], device=self.device)
        
        # uav_lidar
        self._gain_p = torch.tensor([0.028, 0.04, 0.06], device=self.device)
        self._gain_i = torch.tensor([0.2, 0.2, 0.06], device=self.device)
        self._gain_d = torch.tensor([0.0006, 0.0008, 0.0], device=self.device)
        
        # uav_zxw
        # self._gain_p = torch.tensor([0.15, 0.15, 0.2], device=self.device)
        # self._gain_i = torch.tensor([0.2, 0.2, 0.1], device=self.device)
        # self._gain_d = torch.tensor([0.003, 0.003, 0.0], device=self.device)
        
        self.mixer = torch.tensor(
            [
                [-0.70711, -0.70711, 1.0, 1.000000],
                [0.70711, 0.70711, 1.000000, 1.000000],
                [0.70711, -0.70711, -1.0, 1.000000],
                [-0.70711, 0.70711, -1.000000, 1.000000],
            ]
        ).to(self.device)

    def update_integral(self, rate_error, dt):
        i_factor = rate_error / deg2rad(400.0)
        i_factor = torch.clamp(1.0 - i_factor * i_factor, 0.0, 1.0)
        rate_i = self._rate_int + i_factor * self._gain_i * rate_error * dt
        if torch.any(torch.isnan(rate_i)) or torch.any(torch.isinf(rate_i)):
            return
        self._rate_int = torch.clip(rate_i, -self._lim_int, self._lim_int)

    def run(self, target_rate, target_thrust, current_rate, dt):
        rate_error = target_rate - current_rate
        derivative_error = (rate_error - self.prev_rate_error) / dt
        self.prev_rate_error = rate_error
        torque = (
            self._gain_p * rate_error
            + self._rate_int
            - self._gain_d * derivative_error
        )
        self.update_integral(rate_error, dt)
        angacc_thrust = torch.cat([torque, target_thrust], dim=1)
        cmd = (self.mixer @ angacc_thrust.T).T
        return cmd

    def reset(self, env_ids=None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.prev_rate_error[env_ids] = torch.zeros_like(self.prev_rate_error[env_ids])
        self._rate_int[env_ids] = torch.zeros_like(self._rate_int[env_ids])