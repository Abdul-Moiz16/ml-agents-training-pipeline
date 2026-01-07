PARAMS_PPO = {
    # Hyperparameters
    "batch_size":            [32], # range: 32–256 / should always be multiple times smaller than buffer_size / 3D Ball is not complex so smaller batches is fine / typical range: 512 - 5120
    "buffer_size":           [10240], # default: 10240; range: 2048 - 409600
    "learning_rate":         [3e-4], # default: 3e-4; range: 1e-5 - 1e-3
    "beta":                  [5.0e-3], # default: 5.0e-3; range: 1e-4 - 1e-2
    "epsilon":               [0.2], # default: 0,2; range: 0.1 - 0.3
    "lambd":                 [0.95], # default: 0.95; range: 0.9 - 0.95
    "num_epoch":             [3], # default: 3; range: 3 - 10
    "learning_rate_schedule": ["linear"],  # default: "linear"

    # Network settings
    "normalize":             [False], # default: False
    "hidden_units":          [64, 128], # default: 128; range: 32 - 512
    "num_layers":            [1, 3], # default: 2; range: 1 - 3
    "vis_encode_type":       ["simple"], # keep for 3D Ball

    # Reward
    "gamma":                 [0.99], # default: 0.99; range: 0.8 - 0.995
    "strength":              [1], # default: 1

    # General
    "keep_checkpoints":      [5], # default: 5
    "max_steps":             [5000000], # default: 500000; range: 5e5 - 1e7
    "time_horizon":          [64], # default: 64; range: 32 - 2048
    "summary_freq":          [10000], # default: 50000
}
