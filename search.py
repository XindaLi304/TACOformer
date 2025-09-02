# search.py
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from data import make_sequences, get_loaders, make_kfold
from model import ViT
from train import train_one_epoch, evaluate

def run_cv_for_params(X_seq, y_seq, hp, num_folds, n_epochs, bs_train, bs_test, log_interval):
    """
    Run k-fold CV for a given hyperparameter dict `hp`.
    Returns the mean validation accuracy across folds.
    """
    kf = make_kfold(num_folds, seed=hp.get("seed", 778539))
    fold_accs = []

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X_seq)):
        train_loader, valid_loader = get_loaders(X_seq, y_seq, train_idx, valid_idx, bs_train, bs_test)

        net = ViT(
            num_classes=2, height=9, width=9,
            embedding_dim=hp["embedding_dim"], side_dim=8,
            side_mid_dim=64, side_embedding_dim=128,
            depth=hp["depth"], side_depth=hp["side_depth"],
            heads=hp["heads"], side_heads=hp["side_heads"], cross_heads=hp["cross_heads"],
            dim_head=hp["dim_head"], side_dim_head=hp["side_dim_head"], cross_dim_head=hp["cross_dim_head"],
            mlp_dim=hp["mlp_dim"], side_mlp_dim=hp["side_mlp_dim"],
            pool=hp["pool"], dropout=hp["dropout"], emb_dropout=hp["emb_dropout"]
        )

        optimizer = torch.optim.Adam(net.parameters(), lr=hp["learning_rate"], weight_decay=hp["weight_decay"])
        # Keep your original scheduler style (per-batch step). T_max=-1 "disables" cosine; we mimic your original.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=-1)
        use_warmup = True  # keep your warmup usage

        for epoch in range(1, n_epochs + 1):
            train_one_epoch(net, train_loader, optimizer, scheduler, use_warmup, log_interval)
            _ = evaluate(net, valid_loader, fold=fold, real_test=False)

        acc = evaluate(net, valid_loader, fold=fold, real_test=False)
        fold_accs.append(acc)

    return float(sum(fold_accs) / len(fold_accs))

def pick_best_hparams(X_seq, y_seq, grid, num_folds, n_epochs, bs_train, bs_test, log_interval):
    """
    Try all grid combinations; return (best_hp: dict, best_score: float).
    """
    best_hp = None
    best_score = -1.0
    tried = 0

    for hp in grid:
        tried += 1
        print(f"\n=== Trying hyperparameters set #{tried}: {hp}")
        score = run_cv_for_params(X_seq, y_seq, hp, num_folds, n_epochs, bs_train, bs_test, log_interval)
        print(f"Mean CV accuracy: {score:.2f}%")
        if score > best_score:
            best_score = score
            best_hp = hp

    print(f"\n>>> Best hyperparameters: {best_hp}")
    print(f">>> Best mean CV accuracy: {best_score:.2f}%")
    return best_hp, best_score
