import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.utils import resolve_nn_activation

class ActorCriticCascade(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        lidar_input_dim,      # 激光雷达输入维度
        mlp1_state_dim,            # 状态信息维度
        mlp2_state_dim,
        num_critic_obs_dim,  # Critic输入维度
        num_actions,
        
        mlp1_hidden_dims=[128, 64],
        mlp2_hidden_dims=[256, 128, 64],
        mlp1_out_dim=3,
        mlp2_out_dim=64,
        
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        super().__init__()
        activation = resolve_nn_activation(activation)

        # 第一个MLP：输入为lidar和mlp1_state的拼接
        mlp1_in_dim = lidar_input_dim + mlp1_state_dim
        mlp1_layers = [nn.Linear(mlp1_in_dim, mlp1_hidden_dims[0]), activation]
        for i in range(len(mlp1_hidden_dims) - 1):
            mlp1_layers += [nn.Linear(mlp1_hidden_dims[i], mlp1_hidden_dims[i+1]), activation]
        mlp1_layers += [nn.Linear(mlp1_hidden_dims[-1], mlp1_out_dim)]
        self.mlp1 = nn.Sequential(*mlp1_layers)

        # 第二个MLP：输入为mlp1输出和state拼接
        mlp2_in_dim = mlp1_out_dim + mlp2_state_dim
        mlp2_layers = [nn.Linear(mlp2_in_dim, mlp2_hidden_dims[0]), activation]
        for i in range(len(mlp2_hidden_dims) - 1):
            mlp2_layers += [nn.Linear(mlp2_hidden_dims[i], mlp2_hidden_dims[i+1]), activation]
        mlp2_layers += [nn.Linear(mlp2_hidden_dims[-1], mlp2_out_dim)]
        self.mlp2 = nn.Sequential(*mlp2_layers)

        # 输出动作
        self.actor_out = nn.Linear(mlp2_out_dim, num_actions)

        # Critic部分保持原样
        critic_layers = [nn.Linear(num_critic_obs_dim, critic_hidden_dims[0]), activation]
        for i in range(len(critic_hidden_dims) - 1):
            critic_layers += [nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i+1]), activation]
        critic_layers += [nn.Linear(critic_hidden_dims[-1], 1)]
        self.critic = nn.Sequential(*critic_layers)

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

    def forward_actor(self, lidar_input, mlp1_state_input, mlp2_state_input):
        # 1. lidar输入经过第一个MLP
        feat1 = self.mlp1(torch.cat([lidar_input, mlp1_state_input], dim=-1))
        # 2. feat1和lidar, mlp2_state_input拼接，经过第二个MLP
        feat2_input = torch.cat([feat1, lidar_input, mlp2_state_input], dim=-1)
        feat2 = self.mlp2(feat2_input)
        # 输出动作均值
        return self.actor_out(feat2)

    def update_distribution(self, lidar_input, mlp1_state_input, mlp2_state_input):
        mean = self.forward_actor(lidar_input, mlp1_state_input, mlp2_state_input)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, lidar_input, mlp1_state_input, mlp2_state_input, **kwargs):
        self.update_distribution(lidar_input, mlp1_state_input, mlp2_state_input)
        return self.distribution.sample()

    def act_inference(self, lidar_input, mlp1_input, mlp2_state_input):
        return self.forward_actor(lidar_input, mlp1_input, mlp2_state_input)

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)