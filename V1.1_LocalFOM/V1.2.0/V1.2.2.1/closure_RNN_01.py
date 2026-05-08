__all__ = ['device', 'dtype']

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path

current_dir = Path(__file__).parent
fig_dir = Path(__file__).parent / "Fig"
# fig_dir.mkdir(exist_ok=True)


outdata_dir = current_dir / "Out_data"
# outdata_dir.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
torch.set_default_dtype(dtype)


# ============================================================
# 0. 配置
# ============================================================

config = {
    "seed": 42,

    # 数据参数
    "n_samples": 1200,
    "train_ratio": 0.8,

    # RNN 参数
    "input_dim": 1,
    "output_dim": 1,
    "hidden_dim": 64,
    "num_layers": 1,
    "seq_len": 20,

    # 训练参数
    "epochs": 1000,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "weight_decay": 0.0,

    "print_every": 100,
}


# ============================================================
# 1. 数据集预处理
# ============================================================
train_A = np.load(outdata_dir / "Subdomain_A_snapshots.npz")
nt_a, nx_a , ny_a = train_A["snapshots"].shape

train_B = np.load(outdata_dir / "Subdomain_B_snapshots.npz")
nt_b, nx_b , ny_b = train_B["snapshots"].shape

target = np.load(outdata_dir / "coupled_snapshots.npz")
