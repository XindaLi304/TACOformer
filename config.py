# config.py
from itertools import product

class Config:
    # ====== Paths ======
    EEG_PATH = "/root/autodl-tmp/datalist_real.npy"
    EOG_PATH = "/root/autodl-tmp/datalist_eog_deap.npy"
    EMG_PATH = "/root/autodl-tmp/datalist_emg_deap.npy"
    Y_PATH   = "/root/autodl-tmp/dataset/arousal.npy"

    # where to save the held-out test split (.npy)
    SAVE_DIR = "./saved_splits"             # will be created if missing
    X_TEST_NPY = f"{SAVE_DIR}/X_test.npy"
    Y_TEST_NPY = f"{SAVE_DIR}/y_test.npy"

    # ====== Data constants (from your original shapes) ======
    T = 60               # time steps per trial
    EEG_C = 81           # EEG channels
    EOG_C = 4            # EOG channels
    EMG_C = 4            # EMG channels
    SIDE_C = EOG_C + EMG_C
    GRID_H, GRID_W = 9, 9      # 81 -> 9x9
    SEG_LEN = 128
    TEST_SIZE = 0.1

    # ====== Training basic ======
    NUM_FOLDS = 9
    SEED = 778539
    N_EPOCHS = 100
    TRAIN_BS = 64
    TEST_BS = 64
    LOG_INTERVAL = 50
    DEVICE_LIST_STYLE = True  # keep your original device print style ['gpu']/['cpu']

    # ====== Hyperparameter search space (grid) ======
    # Keep them modest first; you can expand later.
    GRID = {
        "learning_rate": [1e-3, 5e-4],
        "weight_decay": [0.0, 1e-4],
        "embedding_dim": [128],    # keep as in your script
        "depth": [3, 4],
        "side_depth": [3, 4],
        "heads": [4],
        "side_heads": [4],
        "cross_heads": [4],
        "dim_head": [32],
        "side_dim_head": [32],
        "cross_dim_head": [32],
        "mlp_dim": [128],
        "side_mlp_dim": [128],
        "dropout": [0.2],
        "emb_dropout": [0.2],
        "pool": ["cls"]
    }

def grid_iter(grid: dict):
    keys = list(grid.keys())
    for values in product(*[grid[k] for k in keys]):
        yield dict(zip(keys, values))
