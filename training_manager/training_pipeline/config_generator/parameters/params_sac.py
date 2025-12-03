PARAMS_SAC = {
    # Hyperparameters
    "learning_rate":         [3e-4], # default: 3e-4; range: 1e-5 - 1e-3
    "learning_rate_schedule": ["linear"], # default: "linear"
    "batch_size":           [64], # default: 64–256 / should always be multiple times smaller than buffer_size / 3D Ball is not complex so smaller batches is fine / typical range: 128 - 1024
    "buffer_size":          [], # default: 50000; range: 50000 - 1000000
    "buffer_init_steps":    [0, 10000], # default: 0; range: 1000 - 10000
    "tau":                  [0.005, 0.01], # default: 0.005; range: 0.005 - 0.01
    "steps_per_update":     [1, 20], # default: 1; range: 1 - 20
    "save_replay_buffer":   [False], # true would save all collected experiences; large and not needed
    "init_entcoef":         [0.5, 1], # default: 1; range: 0.5 - 1.0
    "reward_signal_per_step": [1, 20], # default: 1; range: 1 - 20

    # Network settings
    "normalize":             ["false"], # default: false
    "hidden_units":          [32, 512], # default: 128; range: 32 - 512
    "num_layers":            [1, 3], # default: 2; range: 1 - 3
    "vis_encode_type":       ["simple"], # keep for 3D Ball

    # Reward
    "gamma":                 [0.8, 0.995], # default: 0.99; range: 0.8 - 0.995
    "strength":              [1], # default: 1

    # General
    "keep_checkpoints":      [5], # default: 5
    "max_steps":             [1e7], # default: 500000; range: 5e5 - 1e7
    "time_horizon":          [32, 2048], # default: 64; range: 32 - 2048
    "summary_freq":          [50000], # default: 50000
}
