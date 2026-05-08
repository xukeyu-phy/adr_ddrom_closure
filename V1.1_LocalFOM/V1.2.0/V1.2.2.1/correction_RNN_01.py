import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# -----------------------------
# 1. Load data
# -----------------------------
current_dir = Path(__file__).parent
fig_dir = Path(__file__).parent / "Fig"
# fig_dir.mkdir(exist_ok=True)


outdata_dir = current_dir / "Out_data"

train_A = np.load(outdata_dir / "Subdomain_A_snapshots.npz")
train_B = np.load(outdata_dir / "Subdomain_B_snapshots.npz")
target = np.load(outdata_dir / "coupled_snapshots.npz")
mesh_A = np.load(outdata_dir / "Subdomain_A_Mesh.pkl", allow_pickle=True)
mesh_B = np.load(outdata_dir / "Subdomain_B_Mesh.pkl", allow_pickle=True)

# grid_A = mesh_A["grid"]
grid_A_x = mesh_A["grid_x"]  # (61, 101)
grid_A_y = mesh_A["grid_y"]  # (61, 101)
# grid_B = mesh_B["grid"]  # (41, 101)
grid_B_x = mesh_B["grid_x"]  # (41, 101)
grid_B_y = mesh_B["grid_y"]  # (41, 101)


uA_single = train_A["snapshots"]   # (201, 61, 101)
uB_single = train_B["snapshots"]   # (201, 41, 101)
u_couple = target["snapshots"]     # (201, 101, 101)

uA_couple = u_couple[:, :61, :]
uB_couple = u_couple[:, 60:, :]

nt = uA_single.shape[0]

# -----------------------------
# 2. Train/val/test split
# -----------------------------
train_end = 160
val_end = 180

# -----------------------------
# 3. POD utility
# -----------------------------
def build_pod_basis(data_list, r, train_end):
    """
    data_list: list of arrays, each shape (nt, nx, ny)
    build POD basis using only training time interval.
    """
    X_list = []
    for data in data_list:
        X = data[:train_end].reshape(train_end, -1)
        X_list.append(X)

    X_all = np.concatenate(X_list, axis=0)  # (num_snapshots, space_dim)

    mean = X_all.mean(axis=0, keepdims=True)
    X_centered = X_all - mean

    # SVD: X_centered = U S Vh
    _, _, Vh = np.linalg.svd(X_centered, full_matrices=False)

    basis = Vh[:r].T   # (space_dim, r)

    return mean.reshape(-1), basis


def project_to_pod(data, mean, basis):
    """
    data: (nt, nx, ny)
    return coeffs: (nt, r)
    """
    X = data.reshape(data.shape[0], -1)
    coeff = (X - mean) @ basis
    return coeff


def reconstruct_from_pod(coeff, mean, basis, shape_2d):
    """
    coeff: (nt, r)
    return: (nt, nx, ny)
    """
    X_rec = coeff @ basis.T + mean
    return X_rec.reshape(coeff.shape[0], *shape_2d)


# -----------------------------
# 4. Build POD bases
# -----------------------------
rA = 16
rB = 16

mean_A, VA = build_pod_basis([uA_single, uA_couple], rA, train_end)
mean_B, VB = build_pod_basis([uB_single, uB_couple], rB, train_end)

aA_single = project_to_pod(uA_single, mean_A, VA)
aA_couple = project_to_pod(uA_couple, mean_A, VA)

aB_single = project_to_pod(uB_single, mean_B, VB)
aB_couple = project_to_pod(uB_couple, mean_B, VB)

# -----------------------------
# 5. Build sequence dataset
# -----------------------------
q = 5  # history length

X_seq = []
Y_seq = []
time_ids = []

for n in range(q - 1, nt):
    hist_A = aA_single[n - q + 1:n + 1]
    hist_B = aB_single[n - q + 1:n + 1]

    x = np.concatenate([hist_A, hist_B], axis=1)  # (q, rA + rB)

    yA = aA_couple[n] - aA_single[n]
    yB = aB_couple[n] - aB_single[n]

    y = np.concatenate([yA, yB], axis=0)          # (rA + rB,)

    X_seq.append(x)
    Y_seq.append(y)
    time_ids.append(n)

X_seq = np.array(X_seq)
Y_seq = np.array(Y_seq)
time_ids = np.array(time_ids)

# -----------------------------
# 6. Temporal split
# -----------------------------
train_mask = time_ids < train_end
val_mask = (time_ids >= train_end) & (time_ids < val_end)
test_mask = time_ids >= val_end

X_train, Y_train = X_seq[train_mask], Y_seq[train_mask]
X_val, Y_val = X_seq[val_mask], Y_seq[val_mask]
X_test, Y_test = X_seq[test_mask], Y_seq[test_mask]

# -----------------------------
# 7. Normalize using train only
# -----------------------------
x_mean = X_train.mean(axis=(0, 1), keepdims=True)
x_std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8

y_mean = Y_train.mean(axis=0, keepdims=True)
y_std = Y_train.std(axis=0, keepdims=True) + 1e-8

X_train_n = (X_train - x_mean) / x_std
X_val_n = (X_val - x_mean) / x_std
X_test_n = (X_test - x_mean) / x_std

Y_train_n = (Y_train - y_mean) / y_std
Y_val_n = (Y_val - y_mean) / y_std
Y_test_n = (Y_test - y_mean) / y_std

# -----------------------------
# 8. Torch dataset
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train_t = torch.tensor(X_train_n, dtype=torch.float32).to(device)
Y_train_t = torch.tensor(Y_train_n, dtype=torch.float32).to(device)

X_val_t = torch.tensor(X_val_n, dtype=torch.float32).to(device)
Y_val_t = torch.tensor(Y_val_n, dtype=torch.float32).to(device)

# -----------------------------
# 9. Simple RNN model
# -----------------------------
class SimpleRNNClosure(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            nonlinearity="tanh"
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, h = self.rnn(x)
        last = out[:, -1, :]
        y = self.fc(last)
        return y


input_dim = rA + rB
output_dim = rA + rB
hidden_dim = 32

model = SimpleRNNClosure(input_dim, hidden_dim, output_dim).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# -----------------------------
# 10. Training loop
# -----------------------------
n_epochs = 2000
best_val = np.inf
best_state = None

for epoch in range(1, n_epochs + 1):
    model.train()
    optimizer.zero_grad()

    pred = model(X_train_t)
    loss = loss_fn(pred, Y_train_t)

    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t)
        val_loss = loss_fn(val_pred, Y_val_t)

    if val_loss.item() < best_val:
        best_val = val_loss.item()
        best_state = model.state_dict()

    if epoch % 100 == 0:
        print(
            f"Epoch {epoch:5d} | "
            f"Train Loss = {loss.item():.6e} | "
            f"Val Loss = {val_loss.item():.6e}"
        )

model.load_state_dict(best_state)
print("Best val loss:", best_val)