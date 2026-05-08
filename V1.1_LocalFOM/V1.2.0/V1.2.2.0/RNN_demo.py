import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path

fig_dir = Path(__file__).parent / "Fig"
fig_dir.mkdir(exist_ok=True)

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


torch.manual_seed(config["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")


# ============================================================
# 1. 构造时间序列数据
# ============================================================

def true_function(t):
    return (1 + 0.5 * torch.sin(0.2 * torch.pi * t)) * torch.sin(2.0 * torch.pi * t)


t = torch.linspace(0, 4, config["n_samples"]).reshape(-1, 1)
y = true_function(t)

noise_level = 0.05
y = y + noise_level * torch.randn_like(y)


# ============================================================
# 2. 构造序列样本
# ============================================================

def create_sequences(data, seq_len):
    """
    data: shape = [N, input_dim]

    return:
        X_seq: shape = [N - seq_len, seq_len, input_dim]
        Y_seq: shape = [N - seq_len, input_dim]
    """
    X_seq = []
    Y_seq = []

    for i in range(len(data) - seq_len):
        X_seq.append(data[i:i + seq_len])
        Y_seq.append(data[i + seq_len])

    X_seq = torch.stack(X_seq, dim=0)
    Y_seq = torch.stack(Y_seq, dim=0)

    return X_seq, Y_seq


X_seq, Y_seq = create_sequences(y, config["seq_len"])

print("X_seq shape:", X_seq.shape)
print("Y_seq shape:", Y_seq.shape)


# ============================================================
# 3. 划分训练集和验证集
# ============================================================

n_total = X_seq.shape[0]
n_train = int(config["train_ratio"] * n_total)

X_train = X_seq[:n_train]
Y_train = Y_seq[:n_train]

X_val = X_seq[n_train:]
Y_val = Y_seq[n_train:]


# ============================================================
# 4. 归一化
# ============================================================

x_mean = X_train.mean()
x_std = X_train.std()

y_mean = Y_train.mean()
y_std = Y_train.std()

X_train_norm = (X_train - x_mean) / x_std
Y_train_norm = (Y_train - y_mean) / y_std

X_val_norm = (X_val - x_mean) / x_std
Y_val_norm = (Y_val - y_mean) / y_std

X_train_norm = X_train_norm.to(device)
Y_train_norm = Y_train_norm.to(device)

X_val_norm = X_val_norm.to(device)
Y_val_norm = Y_val_norm.to(device)


# ============================================================
# 5. 定义 RNN 模型
# ============================================================

class RNNRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()

        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="tanh",
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        x shape:
            [batch_size, seq_len, input_dim]
        """

        # rnn_out shape:
        # [batch_size, seq_len, hidden_dim]
        rnn_out, h_last = self.rnn(x)

        # 取最后一个时间步的 hidden state
        last_hidden = rnn_out[:, -1, :]

        # 输出预测值
        y_pred = self.fc(last_hidden)

        return y_pred


model = RNNRegressor(
    input_dim=config["input_dim"],
    hidden_dim=config["hidden_dim"],
    output_dim=config["output_dim"],
    num_layers=config["num_layers"],
).to(device)

print(model)


# ============================================================
# 6. 定义 loss 和 optimizer
# ============================================================

loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config["learning_rate"],
    weight_decay=config["weight_decay"],
)


# ============================================================
# 7. 训练循环
# ============================================================

train_losses = []
val_losses = []

n_train = X_train_norm.shape[0]

for epoch in range(1, config["epochs"] + 1):

    model.train()

    perm = torch.randperm(n_train, device=device)

    epoch_train_loss = 0.0

    for i in range(0, n_train, config["batch_size"]):
        batch_idx = perm[i:i + config["batch_size"]]

        xb = X_train_norm[batch_idx]
        yb = Y_train_norm[batch_idx]

        pred = model(xb)

        loss = loss_fn(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item() * xb.shape[0]

    epoch_train_loss /= n_train

    model.eval()

    with torch.no_grad():
        val_pred = model(X_val_norm)
        val_loss = loss_fn(val_pred, Y_val_norm).item()

    train_losses.append(epoch_train_loss)
    val_losses.append(val_loss)

    if epoch % config["print_every"] == 0:
        print(
            f"Epoch {epoch:5d} | "
            f"train loss = {epoch_train_loss:.6e} | "
            f"val loss = {val_loss:.6e}"
        )


# ============================================================
# 8. 验证集预测并反归一化
# ============================================================

model.eval()

with torch.no_grad():
    pred_val_norm = model(X_val_norm).cpu()

pred_val = pred_val_norm * y_std + y_mean


# ============================================================
# 9. 画图
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(Y_val.numpy(), label="True")
plt.plot(pred_val.numpy(), "--", label="RNN prediction")
plt.xlabel("Validation sample index")
plt.ylabel("y")
plt.title(f"RNN Layer:{config['num_layers']}, Hidden dim:{config['hidden_dim']}, Seq len:{config['seq_len']}")
plt.legend()
plt.tight_layout()
plt.savefig(f"{fig_dir}/rnn_prediction_l{config['num_layers']}_dim{config['hidden_dim']}_S{config['seq_len']}.png", dpi=200)
plt.show()


plt.figure(figsize=(8, 5))
plt.semilogy(train_losses, label="Train loss")
plt.semilogy(val_losses, label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.title(f"RNN Layer:{config['num_layers']}, Hidden dim:{config['hidden_dim']}, Seq len:{config['seq_len']}")
plt.legend()
plt.tight_layout()
plt.savefig(f"{fig_dir}/rnn_loss_curve_l{config['num_layers']}_dim{config['hidden_dim']}_S{config['seq_len']}.png", dpi=200)
plt.show()