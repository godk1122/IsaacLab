# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, \
    RslRlPpoActorCriticCascadeCfg, RslRlPpoCaAlgorithmCfg, RslRlPpoActorCriticRecurrentCfg

@configclass
class QuadcopterPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 200000
    save_interval = 500
    experiment_name = "quadcopter_rate"
    empirical_normalization = False
    
    # resume = True
    # load_run = "2024-11-15_08-24-56"
    # load_checkpoint = "model_300"

    # lidar     
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.2,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[512, 256, 128, 64],
        activation="elu",
    )
    
    
    # hover     
    # policy = RslRlPpoActorCriticCfg(
    #     init_noise_std=1.0,
    #     actor_hidden_dims=[64, 64],
    #     critic_hidden_dims=[64, 64],
    #     activation="elu",
    # )
    
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.15,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

@configclass
class QuadcopterPPORNNRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 200000
    save_interval = 500
    experiment_name = "quadcopter_rate"
    empirical_normalization = False
    
    # resume = True
    # load_run = "2024-11-15_08-24-56"
    # load_checkpoint = "model_300"

    # lidar     
    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=0.2,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[512, 256, 128, 64],
        activation="elu",
        rnn_type="lstm",         # 或 "gru"
        rnn_hidden_dim=128,      # RNN隐藏层维度
        rnn_num_layers=1,        # RNN层数
    )
    
    
    # hover     
    # policy = RslRlPpoActorCriticCfg(
    #     init_noise_std=1.0,
    #     actor_hidden_dims=[64, 64],
    #     critic_hidden_dims=[64, 64],
    #     activation="elu",
    # )
    
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.15,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

@configclass
class QuadcopterPPOCascadeRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 200000
    save_interval = 500
    experiment_name = "quadcopter_rate"
    empirical_normalization = False

    # lidar cascade policy
    policy = RslRlPpoActorCriticCascadeCfg(
        init_noise_std=1.0,
        noise_std_type="scalar",
        lidar_input_dim=72*5,         # 请根据实际输入修改
        mlp1_state_dim=6*3,               # 对应 mlp1_state_dim
        mlp2_state_dim=17*3,               # 对应 mlp2_state_dim
        mlp1_hidden_dims=[128, 64],
        mlp2_hidden_dims=[256, 128, 64],
        mlp1_out_dim=3,
        mlp2_out_dim=64,
        critic_hidden_dims=[512, 256, 128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoCaAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.15,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )