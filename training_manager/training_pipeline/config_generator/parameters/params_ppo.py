PARAMS_PPO = {
    # Hyperparameters
    "batch_size":            [32, 64, 128, 256], # range: 32–256 / should always be multiple times smaller than buffer_size / 3D Ball is not complex so smaller batches is fine / typical range: 512 - 5120
    "buffer_size":           [10240], # default: 10240; range: 2048 - 409600 (here the value does not matter since it is calculated later in generator_base
    "learning_rate":         [1e-5, 1e-3], # default: 3e-4; range: 1e-5 - 1e-3
    "beta":                  [1e-4, 1e-2], # default: 5.0e-3; range: 1e-4 - 1e-2
    "epsilon":               [0.1, 0.3], # default: 0,2; range: 0.1 - 0.3
    "lambd":                 [0.90, 0.95], # default: 0.95; range: 0.9 - 0.95
    "num_epoch":             [3, 5], # default: 3; range: 3 - 10 (we will keep the range 3, 5 since it heavily influences the duration of the run since with a higher value it might not converge, due to overfitting, and take a longer time training)
    "learning_rate_schedule": ["linear"],  # default: "linear"

    # Network settings
    "normalize":             [False], # default: False
    "hidden_units":          [64, 128, 256], # default: 128; range: 32 - 512
    "num_layers":            [1, 2, 3], # default: 2; range: 1 - 3
    "vis_encode_type":       ["simple"], # keep for 3D Ball

    # Reward
    "gamma":                 [0.8, 0.995], # default: 0.99; range: 0.8 - 0.995
    "strength":              [1], # default: 1

    # General
    "keep_checkpoints":      [5], # default: 5
    "max_steps":             [5000000], # default: 500000; range: 5e5 - 1e7
    "time_horizon":          [64, 128, 256], # default: 64; range: 32 - 2048 (limited to 64, 128 and 256 since 3DBall is a short game which does not take many step to complete so no need to look too far ahead)
    "summary_freq":          [10000], # default: 50000
}
