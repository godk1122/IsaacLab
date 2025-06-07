import torch
import torch.nn as nn

class CascadeMLP(nn.Module):
    def __init__(
        self,
        lidar_input_dim,           # 激光雷达输入维度
        state1_dim,                # 第一次拼接的状态信息维度
        state2_dim,                # 第二次拼接的状态信息维度
        mlp1_hidden_dims=[128, 128],
        mlp2_hidden_dims=[128, 128],
        mlp3_hidden_dims=[128, 128],
        mlp1_out_dim=64,
        mlp2_out_dim=64,
        mlp3_out_dim=64,
        activation=nn.ELU,
    ):
        super().__init__()
        act = activation()

        # 第一个MLP：只接收lidar输入
        mlp1_layers = [nn.Linear(lidar_input_dim, mlp1_hidden_dims[0]), act]
        for i in range(len(mlp1_hidden_dims) - 1):
            mlp1_layers += [nn.Linear(mlp1_hidden_dims[i], mlp1_hidden_dims[i+1]), act]
        mlp1_layers += [nn.Linear(mlp1_hidden_dims[-1], mlp1_out_dim)]
        self.mlp1 = nn.Sequential(*mlp1_layers)

        # 第二个MLP：输入为mlp1输出和state1拼接
        mlp2_in_dim = mlp1_out_dim + state1_dim
        mlp2_layers = [nn.Linear(mlp2_in_dim, mlp2_hidden_dims[0]), act]
        for i in range(len(mlp2_hidden_dims) - 1):
            mlp2_layers += [nn.Linear(mlp2_hidden_dims[i], mlp2_hidden_dims[i+1]), act]
        mlp2_layers += [nn.Linear(mlp2_hidden_dims[-1], mlp2_out_dim)]
        self.mlp2 = nn.Sequential(*mlp2_layers)

        # 第三个MLP：输入为mlp1输出、mlp2输出和state2拼接
        mlp3_in_dim = mlp1_out_dim + mlp2_out_dim + state2_dim
        mlp3_layers = [nn.Linear(mlp3_in_dim, mlp3_hidden_dims[0]), act]
        for i in range(len(mlp3_hidden_dims) - 1):
            mlp3_layers += [nn.Linear(mlp3_hidden_dims[i], mlp3_hidden_dims[i+1]), act]
        mlp3_layers += [nn.Linear(mlp3_hidden_dims[-1], mlp3_out_dim)]
        self.mlp3 = nn.Sequential(*mlp3_layers)

    def forward(self, lidar_input, state1, state2):
        """
        lidar_input: [batch, lidar_input_dim]
        state1: [batch, state1_dim]
        state2: [batch, state2_dim]
        """
        # 1. lidar输入经过第一个MLP
        feat1 = self.mlp1(lidar_input)  # [batch, mlp1_out_dim]

        # 2. feat1和state1拼接，经过第二个MLP
        feat2_input = torch.cat([feat1, state1], dim=-1)
        feat2 = self.mlp2(feat2_input)  # [batch, mlp2_out_dim]

        # 3. feat1、feat2和state2拼接，经过第三个MLP
        feat3_input = torch.cat([feat1, feat2, state2], dim=-1)
        feat3 = self.mlp3(feat3_input)  # [batch, mlp3_out_dim]

        return feat3