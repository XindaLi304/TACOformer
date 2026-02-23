# main.py
import os
import numpy as np
import torch
from config import Config, grid_iter
from utils import set_seed, device_print_like_original
from data import load_raw_arrays, build_merged_data, split_and_save_test, make_sequences
from search import pick_best_hparams
from model import ViT
from train import train_one_epoch, evaluate

def main():
    # 0) seed
    set_seed(Config.SEED)

    # 1) load raw & merge
    data_eeg, data_eog, data_emg, y = load_raw_arrays()
    print("raw shapes:", data_eeg.shape, data_eog.shape, data_emg.shape)
    data = build_merged_data(data_eeg, data_eog, data_emg)
    print("merged data:", data.shape)

    # 2) train/test split, save test as .npy
    X_train, y_train, X_test, y_test = split_and_save_test(
        data, y, seed=Config.SEED, test_size=Config.TEST_SIZE
    )
    print("X_test saved to:", Config.X_TEST_NPY, "shape:", X_test.shape)

    # 3) flatten sequences for CV (train only)
    X_seq, y_seq = make_sequences(X_train, y_train)

    # 4) hyperparameter grid search with k-fold CV
    grid = list(grid_iter(Config.GRID))
    best_hp, best_score = pick_best_hparams(
        X_seq, y_seq, grid,
        num_folds=Config.NUM_FOLDS,
        n_epochs=Config.N_EPOCHS,
        bs_train=Config.TRAIN_BS,
        bs_test=Config.TEST_BS,
        log_interval=Config.LOG_INTERVAL
    )

    # 5) Train final model on the full training split with best hyperparameters

    device_style = device_print_like_original()
    print(device_style)

    # We need a loader for full training set
    from torch.utils.data import TensorDataset, DataLoader
    train_loader = DataLoader(TensorDataset(X_seq, y_seq), batch_size=Config.TRAIN_BS, shuffle=True)

    net = ViT(
        num_classes=2, height=Config.GRID_H, width=Config.GRID_W,
        embedding_dim=best_hp["embedding_dim"], side_dim=Config.SIDE_C,
        side_mid_dim=64, side_embedding_dim=128,
        depth=best_hp["depth"], side_depth=best_hp["side_depth"],
        heads=best_hp["heads"], side_heads=best_hp["side_heads"], cross_heads=best_hp["cross_heads"],
        dim_head=best_hp["dim_head"], side_dim_head=best_hp["side_dim_head"], cross_dim_head=best_hp["cross_dim_head"],
        mlp_dim=best_hp["mlp_dim"], side_mlp_dim=best_hp["side_mlp_dim"],
        pool=best_hp["pool"], dropout=best_hp["dropout"], emb_dropout=best_hp["emb_dropout"]
    )

    optimizer = torch.optim.Adam(net.parameters(), lr=best_hp["learning_rate"], weight_decay=best_hp["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=-1)  # keep your original behavior
    use_warmup = True  # keep consistency with your code

    print("Start training final model with best hyperparams...")
    for epoch in range(1, Config.N_EPOCHS + 1):
        train_one_epoch(net, train_loader, optimizer, scheduler, use_warmup, log_interval=Config.LOG_INTERVAL)

    # 6) Evaluate on held-out test set saved on disk
    #    reshape test into (N*T, C, S) and evaluate (with confusion matrix)
    X_test = np.load(Config.X_TEST_NPY)
    Y_test = np.load(Config.Y_TEST_NPY)
    X_te, y_te = make_sequences(X_test, Y_test)

    test_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=Config.TEST_BS, shuffle=False)

    print("Evaluating on held-out test set with the best model...")
    acc = evaluate(net, test_loader, real_test=True, collect_cm=True)
    print(f"Final held-out test accuracy: {acc:.2f}%")

if __name__ == "__main__":
    from torch.utils.data import DataLoader, TensorDataset  # local import to avoid unused warnings
    main()
