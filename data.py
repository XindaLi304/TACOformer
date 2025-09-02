# data.py
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
from config import Config

def load_raw_arrays():
    """Load raw arrays from disk and return (data_eeg, data_eog, data_emg, y)."""
    data_eeg = np.load(Config.EEG_PATH)
    data_eog = np.load(Config.EOG_PATH)
    data_emg = np.load(Config.EMG_PATH)
    y = np.load(Config.Y_PATH)
    return data_eeg, data_eog, data_emg, y

def build_merged_data(data_eeg, data_eog, data_emg):
    """
    Shape transforms based on your original script:
    EEG: (1280, 60, 81, 128)
    EOG: (1280, 60, 4, 128)
    EMG: (1280, 60, 4, 128)
    Concatenate along channel dim -> (1280, 60, 81+4+4=89, 128), then squeeze if needed.
    """
    data_eeg = np.reshape(data_eeg, (-1, Config.T, Config.EEG_C, Config.SEG_LEN))
    data_eog = np.reshape(data_eog, (-1, Config.T,  Config.EOG_C, Config.SEG_LEN))
    data_emg = np.reshape(data_emg, (-1, Config.T,  Config.EMG_C, Config.SEG_LEN))
    data = np.concatenate((data_eeg, data_eog), axis=2)
    data = np.concatenate((data, data_emg), axis=2)
    data = np.squeeze(data)
    return data

def split_and_save_test(data, y, seed: int, test_size: float):
    """
    Split train/test, save test arrays to .npy, and return (X_train, y_train, X_test, y_test).
    y is reshaped to (N, T) as in your code.
    """
    y = np.reshape(y, (-1, Config.T))
    # cast types same as your pipeline
    y = y.astype(np.float64)
    data = data.astype(np.float64)

    X_train, X_test, y_train, y_test = train_test_split(
        data, y, test_size=test_size, random_state=seed
    )

    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    np.save(Config.X_TEST_NPY, X_test)
    np.save(Config.Y_TEST_NPY, y_test)

    return X_train, y_train, X_test, y_test

def make_sequences(X_trials, y_trials):
    """
    Flatten trial dimension: (N_trials, T, C, S) -> (N_trials*T, C, S)
    Labels: (N_trials, T) -> (N_trials*T,)
    """
    n, t, c, s = X_trials.shape
    X = X_trials.reshape(n*t, c, s)
    y = y_trials.reshape(n*t)
    # types aligned with your original tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return X, y

def get_loaders(X, y, train_idx, valid_idx, batch_train, batch_test):
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_va, y_va = X[valid_idx], y[valid_idx]

    # as in your code: reshape to (samples, 89, 128)
    # we already made_sequences before calling this function
    train_ds = TensorDataset(X_tr, y_tr)
    valid_ds = TensorDataset(X_va, y_va)

    train_loader = DataLoader(train_ds, batch_size=batch_train, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_test, shuffle=False)
    return train_loader, valid_loader

def make_kfold(n_splits, seed):
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)
