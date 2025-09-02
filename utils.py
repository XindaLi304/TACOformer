# utils.py
import os
import random
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def device_print_like_original():
    return ['gpu' if torch.cuda.is_available() else 'cpu']
