import copy
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


# ============================================================
# 0. Settings
# ============================================================

MODE = "state"
# MODE = "rhs"

q = 5                    # history length
rA = 16                  # POD modes for subdomain A
rB = 16                  # POD modes for subdomain B
hidden_dim = 32
n_epochs = 2000
lr = 5e-4

dt = 0.00025             # 改成你真实的时间步长

train_end = 160
val_end = 180

plot_time_id = 200       # 用于画场图
seed = 42


# ============================================================
# 1. Reproducibility
# ============================================================

np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 2. Load data
# ============================================================

current_dir = Path(__file__).parent
fig_dir = current_dir / "Fig"
fig_dir.mkdir(exist_ok=True)

outdata_dir = current_dir / "Out_data"

train_A = np.load(outdata_dir / "Subdomain_A_snapshots.npz")
train_B = np.load(outdata_dir / "Subdomain_B_snapshots.npz")
target = np.load(outdata_dir / "coupled_snapshots.npz")

mesh_A = np.load(outdata_dir / "Subdomain_A_Mesh.pkl", allow_pickle=True)
mesh_B = np.load(outdata_dir / "Subdomain_B_Mesh.pkl", allow_pickle=True)

grid_A_x = mesh_A["grid_x"]  # (61, 101)
grid_A_y = mesh_A["grid_y"]  # (61, 101)
grid_B_x = mesh_B["grid_x"]  # (41, 101)
grid_B_y = mesh_B["grid_y"]  # (41, 101)

uA_single = train_A["snapshots"]   # (201, 61, 101)
uB_single = train_B["snapshots"]   # (201, 41, 101)
u_couple = target["snapshots"]     # (201, 101, 101)

uA_couple = u_couple[:, :61, :]
uB_couple = u_couple[:, 60:, :]

nt = uA_single.shape[0]

print("uA_single:", uA_single.shape)
print("uB_single:", uB_single.shape)
print("u_couple :", u_couple.shape)
print("uA_couple:", uA_couple.shape)
print("uB_couple:", uB_couple.shape)


# ============================================================
# 3. POD utilities
# ============================================================

def build_pod_basis(data_list, r, train_end):
    """
    Build POD basis from a list of datasets.
    Each dataset shape: (nt, nx, ny).

    Return:
        mean:  (space_dim,)
        basis: (space_dim, r)
        s: singular values
        energy: cumulative energy ratio
    """
    X_list = []

    for data in data_list:
        X = data[:train_end].reshape(train_end, -1)
        X_list.append(X)

    X_all = np.concatenate(X_list, axis=0)

    mean = X_all.mean(axis=0, keepdims=True)
    X_centered = X_all - mean

    _, s, Vh = np.linalg.svd(X_centered, full_matrices=False)

    basis = Vh[:r].T
    energy = np.cumsum(s ** 2) / np.sum(s ** 2)

    return mean.reshape(-1), basis, s, energy


def project_to_pod(data, mean, basis):
    """
    data: (nt, nx, ny)
    return coeff: (nt, r)
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


# ============================================================
# 4. Plot POD singular values
# ============================================================

def plot_pod_singular_values(sA, energyA, sB, energyB, fig_dir):
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.semilogy(np.arange(1, len(sA) + 1), sA / sA[0], "o-", markersize=3, label="Subdomain A")
    ax.semilogy(np.arange(1, len(sB) + 1), sB / sB[0], "s-", markersize=3, label="Subdomain B")

    ax.set_xlabel("Mode index")
    ax.set_ylabel("Normalized singular value")
    ax.set_title("POD singular value decay")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()

    plt.tight_layout()
    plt.savefig(fig_dir / "pod_singular_values.png", dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(np.arange(1, len(energyA) + 1), energyA, "o-", markersize=3, label="Subdomain A")
    ax.plot(np.arange(1, len(energyB) + 1), energyB, "s-", markersize=3, label="Subdomain B")

    ax.axhline(0.999, linestyle="--", linewidth=1.0, label="99.9%")
    ax.set_xlabel("Number of modes")
    ax.set_ylabel("Cumulative energy ratio")
    ax.set_title("POD cumulative energy")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    plt.tight_layout()
    plt.savefig(fig_dir / "pod_cumulative_energy.png", dpi=300)
    plt.close()


# ============================================================
# 5. Build POD bases
# ============================================================

# 这里为了 demo，让 POD basis 同时看到 single 和 coupled 的训练段。
# 后续正式 localized ROM 时，可以只用 local offline snapshots 来构造 basis。
mean_A, VA, sA, energyA = build_pod_basis(
    [uA_single, uA_couple],
    r=rA,
    train_end=train_end
)

mean_B, VB, sB, energyB = build_pod_basis(
    [uB_single, uB_couple],
    r=rB,
    train_end=train_end
)

# mean, V, s, energyA = build_pod_basis(
#     [uA_single, uA_couple],
#     r=rA,
#     train_end=train_end
# )

plot_pod_singular_values(sA, energyA, sB, energyB, fig_dir)

print(f"A: first {rA} modes energy = {energyA[rA - 1]:.8f}")
print(f"B: first {rB} modes energy = {energyB[rB - 1]:.8f}")


# ============================================================
# 6. Project snapshots to POD coefficients
# ============================================================

aA_single = project_to_pod(uA_single, mean_A, VA)
aA_couple = project_to_pod(uA_couple, mean_A, VA)

aB_single = project_to_pod(uB_single, mean_B, VB)
aB_couple = project_to_pod(uB_couple, mean_B, VB)

print("aA_single:", aA_single.shape)
print("aB_single:", aB_single.shape)


# ============================================================
# 7. Build sequence dataset
# ============================================================

def build_sequence_dataset_state(aA_single, aA_couple, aB_single, aB_couple, q):
    """
    State correction:
        input  = [aA_single history, aB_single history]
        output = [aA_couple - aA_single, aB_couple - aB_single]
    """
    nt = aA_single.shape[0]

    X_seq = []
    Y_seq = []
    time_ids = []

    for n in range(q - 1, nt):
        hist_A = aA_single[n - q + 1:n + 1]
        hist_B = aB_single[n - q + 1:n + 1]

        x = np.concatenate([hist_A, hist_B], axis=1)

        yA = aA_couple[n] - aA_single[n]
        yB = aB_couple[n] - aB_single[n]

        y = np.concatenate([yA, yB], axis=0)

        X_seq.append(x)
        Y_seq.append(y)
        time_ids.append(n)

    return np.array(X_seq), np.array(Y_seq), np.array(time_ids)


def build_sequence_dataset_rhs(aA_single, aA_couple, aB_single, aB_couple, q, dt):
    """
    RHS closure:
        input  = [aA_single history, aB_single history]
        output = [(daA_couple/dt - daA_single/dt),
                  (daB_couple/dt - daB_single/dt)]
    """
    nt = aA_single.shape[0]

    X_seq = []
    Y_seq = []
    time_ids = []

    for n in range(q - 1, nt - 1):
        hist_A = aA_single[n - q + 1:n + 1]
        hist_B = aB_single[n - q + 1:n + 1]

        x = np.concatenate([hist_A, hist_B], axis=1)

        dA_couple = (aA_couple[n + 1] - aA_couple[n]) / dt
        dA_single = (aA_single[n + 1] - aA_single[n]) / dt

        dB_couple = (aB_couple[n + 1] - aB_couple[n]) / dt
        dB_single = (aB_single[n + 1] - aB_single[n]) / dt

        mA = dA_couple - dA_single
        mB = dB_couple - dB_single

        y = np.concatenate([mA, mB], axis=0)

        X_seq.append(x)
        Y_seq.append(y)
        time_ids.append(n)

    return np.array(X_seq), np.array(Y_seq), np.array(time_ids)


if MODE == "state":
    X_seq, Y_seq, time_ids = build_sequence_dataset_state(
        aA_single, aA_couple, aB_single, aB_couple, q
    )
elif MODE == "rhs":
    X_seq, Y_seq, time_ids = build_sequence_dataset_rhs(
        aA_single, aA_couple, aB_single, aB_couple, q, dt
    )
else:
    raise ValueError(f"Unsupported MODE: {MODE}")

print("MODE:", MODE)
print("X_seq:", X_seq.shape)
print("Y_seq:", Y_seq.shape)
print("time_ids:", time_ids.shape)


# ============================================================
# 8. Temporal split
# ============================================================

train_mask = time_ids < train_end
val_mask = (time_ids >= train_end) & (time_ids < val_end)
test_mask = time_ids >= val_end

X_train, Y_train = X_seq[train_mask], Y_seq[train_mask]
X_val, Y_val = X_seq[val_mask], Y_seq[val_mask]
X_test, Y_test = X_seq[test_mask], Y_seq[test_mask]

print("X_train:", X_train.shape, "Y_train:", Y_train.shape)
print("X_val  :", X_val.shape, "Y_val  :", Y_val.shape)
print("X_test :", X_test.shape, "Y_test :", Y_test.shape)


# ============================================================
# 9. Normalization
# ============================================================

x_mean = X_train.mean(axis=(0, 1), keepdims=True)
x_std = X_train.std(axis=(0, 1), keepdims=True) + 1e-10

y_mean = Y_train.mean(axis=0, keepdims=True)
y_std = Y_train.std(axis=0, keepdims=True) + 1e-10

X_train_n = (X_train - x_mean) / x_std
X_val_n = (X_val - x_mean) / x_std
X_test_n = (X_test - x_mean) / x_std
X_all_n = (X_seq - x_mean) / x_std

Y_train_n = (Y_train - y_mean) / y_std
Y_val_n = (Y_val - y_mean) / y_std
Y_test_n = (Y_test - y_mean) / y_std


# ============================================================
# 10. Torch tensors
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

X_train_t = torch.tensor(X_train_n, dtype=torch.float32).to(device)
Y_train_t = torch.tensor(Y_train_n, dtype=torch.float32).to(device)

X_val_t = torch.tensor(X_val_n, dtype=torch.float32).to(device)
Y_val_t = torch.tensor(Y_val_n, dtype=torch.float32).to(device)

X_all_t = torch.tensor(X_all_n, dtype=torch.float32).to(device)


# ============================================================
# 11. Simple RNN closure model
# ============================================================

class SimpleRNNClosure(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            nonlinearity="tanh",
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, h = self.rnn(x)
        last = out[:, -1, :]
        y = self.fc(last)
        return y


input_dim = rA + rB
output_dim = rA + rB

model = SimpleRNNClosure(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()


# ============================================================
# 12. Training
# ============================================================

train_losses = []
val_losses = []

best_val = np.inf
best_state = None

for epoch in range(1, n_epochs + 1):
    model.train()

    optimizer.zero_grad()
    pred_train = model(X_train_t)
    train_loss = loss_fn(pred_train, Y_train_t)

    train_loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_val = model(X_val_t)
        val_loss = loss_fn(pred_val, Y_val_t)

    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())

    if val_loss.item() < best_val:
        best_val = val_loss.item()
        best_state = copy.deepcopy(model.state_dict())

    if epoch % 100 == 0:
        print(
            f"Epoch {epoch:5d} | "
            f"Train Loss = {train_loss.item():.6e} | "
            f"Val Loss = {val_loss.item():.6e}"
        )

model.load_state_dict(best_state)

print("Best val loss:", best_val)


# ============================================================
# 13. Plot training loss
# ============================================================

def plot_training_loss(train_losses, val_losses, fig_dir, mode):
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.semilogy(train_losses, label="Train loss")
    ax.semilogy(val_losses, label="Validation loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title(f"Training loss ({mode})")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()

    plt.tight_layout()
    plt.savefig(fig_dir / f"training_loss_{mode}.png", dpi=300)
    plt.close()


plot_training_loss(train_losses, val_losses, fig_dir, MODE)


# ============================================================
# 14. Predict closure/correction for all available sequence times
# ============================================================

model.eval()
with torch.no_grad():
    Y_pred_all_n = model(X_all_t).cpu().numpy()

Y_pred_all = Y_pred_all_n * y_std + y_mean

print("Y_pred_all:", Y_pred_all.shape)


# ============================================================
# 15. Build corrected POD coefficients
# ============================================================

def apply_state_correction(
    aA_single, aB_single, Y_pred_all, time_ids, rA, rB
):
    """
    a_pred[n] = a_single[n] + NN_delta[n]
    """
    aA_pred = aA_single.copy()
    aB_pred = aB_single.copy()

    for k, n in enumerate(time_ids):
        y = Y_pred_all[k]

        delta_A = y[:rA]
        delta_B = y[rA:rA + rB]

        aA_pred[n] = aA_single[n] + delta_A
        aB_pred[n] = aB_single[n] + delta_B

    return aA_pred, aB_pred


def apply_rhs_closure(
    aA_single, aB_single, Y_pred_all, time_ids, rA, rB, dt, q
):
    """
    Use learned RHS closure:

        a_pred[n+1]
        =
        a_pred[n]
        +
        (a_single[n+1] - a_single[n])
        +
        dt * m_NN[n]

    Here the closure input is still built from single trajectories.
    This is a first teacher-forced RHS-closure test.
    """
    nt = aA_single.shape[0]

    aA_pred = aA_single.copy()
    aB_pred = aB_single.copy()

    pred_map = {}
    for k, n in enumerate(time_ids):
        pred_map[int(n)] = Y_pred_all[k]

    # before q-1, keep single trajectory
    for n in range(q - 1, nt - 1):
        if n not in pred_map:
            continue

        y = pred_map[n]

        mA = y[:rA]
        mB = y[rA:rA + rB]

        base_step_A = aA_single[n + 1] - aA_single[n]
        base_step_B = aB_single[n + 1] - aB_single[n]

        aA_pred[n + 1] = aA_pred[n] + base_step_A + dt * mA
        aB_pred[n + 1] = aB_pred[n] + base_step_B + dt * mB

    return aA_pred, aB_pred


if MODE == "state":
    aA_pred, aB_pred = apply_state_correction(
        aA_single, aB_single, Y_pred_all, time_ids, rA, rB
    )
elif MODE == "rhs":
    aA_pred, aB_pred = apply_rhs_closure(
        aA_single, aB_single, Y_pred_all, time_ids, rA, rB, dt, q
    )
else:
    raise ValueError(f"Unsupported MODE: {MODE}")


# ============================================================
# 16. Reconstruct full fields
# ============================================================

uA_pred = reconstruct_from_pod(
    aA_pred, mean_A, VA, shape_2d=uA_single.shape[1:]
)

uB_pred = reconstruct_from_pod(
    aB_pred, mean_B, VB, shape_2d=uB_single.shape[1:]
)


def merge_subdomains(uA, uB, average_interface=True):
    """
    uA: (nt, 61, 101)
    uB: (nt, 41, 101)

    Full domain:
        A uses indices 0:61
        B uses indices 60:101
    """
    nt, nxA, ny = uA.shape
    nxB = uB.shape[1]

    nx_full = nxA + nxB - 1
    u_full = np.zeros((nt, nx_full, ny), dtype=uA.dtype)

    u_full[:, :nxA, :] = uA
    u_full[:, nxA:, :] = uB[:, 1:, :]

    if average_interface:
        u_full[:, nxA - 1, :] = 0.5 * (uA[:, -1, :] + uB[:, 0, :])

    return u_full


u_pred_full = merge_subdomains(uA_pred, uB_pred, average_interface=True)
u_single_full = merge_subdomains(uA_single, uB_single, average_interface=True)

grid_x = np.concatenate([grid_A_x, grid_B_x[1:, :]], axis=0)
grid_y = np.concatenate([grid_A_y, grid_B_y[1:, :]], axis=0)

print("u_pred_full:", u_pred_full.shape)
print("grid_x:", grid_x.shape)


# ============================================================
# 17. Error computation
# ============================================================

def compute_global_errors(u_pred, u_ref):
    diff = u_pred - u_ref

    abs_l2 = np.linalg.norm(diff.reshape(-1), ord=2)
    ref_l2 = np.linalg.norm(u_ref.reshape(-1), ord=2)
    rel_l2 = abs_l2 / (ref_l2 + 1e-14)

    abs_linf = np.max(np.abs(diff))
    ref_linf = np.max(np.abs(u_ref))
    rel_linf = abs_linf / (ref_linf + 1e-14)

    return {
        "abs_l2": abs_l2,
        "rel_l2": rel_l2,
        "abs_linf": abs_linf,
        "rel_linf": rel_linf,
    }


def compute_time_errors(u_pred, u_ref):
    nt = u_pred.shape[0]

    rel_l2_time = np.zeros(nt)
    abs_l2_time = np.zeros(nt)
    abs_linf_time = np.zeros(nt)
    rel_linf_time = np.zeros(nt)

    for n in range(nt):
        diff = u_pred[n] - u_ref[n]

        abs_l2 = np.linalg.norm(diff.reshape(-1), ord=2)
        ref_l2 = np.linalg.norm(u_ref[n].reshape(-1), ord=2)

        abs_linf = np.max(np.abs(diff))
        ref_linf = np.max(np.abs(u_ref[n]))

        abs_l2_time[n] = abs_l2
        rel_l2_time[n] = abs_l2 / (ref_l2 + 1e-14)

        abs_linf_time[n] = abs_linf
        rel_linf_time[n] = abs_linf / (ref_linf + 1e-14)

    return abs_l2_time, rel_l2_time, abs_linf_time, rel_linf_time


errors_pred = compute_global_errors(u_pred_full, u_couple)
errors_single = compute_global_errors(u_single_full, u_couple)

print("\n================ Global Errors ================")
print("Single vs coupled:")
for k, v in errors_single.items():
    print(f"  {k:10s}: {v:.6e}")

print(f"\nNN corrected ({MODE}) vs coupled:")
for k, v in errors_pred.items():
    print(f"  {k:10s}: {v:.6e}")

abs_l2_t, rel_l2_t, abs_linf_t, rel_linf_t = compute_time_errors(
    u_pred_full, u_couple
)

abs_l2_single_t, rel_l2_single_t, abs_linf_single_t, rel_linf_single_t = compute_time_errors(
    u_single_full, u_couple
)


# ============================================================
# 18. Plot error curves
# ============================================================

def plot_error_curves(
    rel_l2_single_t,
    rel_l2_t,
    rel_linf_single_t,
    rel_linf_t,
    fig_dir,
    mode
):
    t_id = np.arange(len(rel_l2_t))

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.semilogy(t_id, rel_l2_single_t, label="Single vs coupled")
    ax.semilogy(t_id, rel_l2_t, label="NN corrected vs coupled")

    ax.axvline(train_end, linestyle="--", linewidth=1.0, label="train/val split")
    ax.axvline(val_end, linestyle=":", linewidth=1.0, label="val/test split")

    ax.set_xlabel("Time index")
    ax.set_ylabel("Relative L2 error")
    ax.set_title(f"Relative L2 error ({mode})")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()

    plt.tight_layout()
    plt.savefig(fig_dir / f"relative_l2_error_{mode}.png", dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.semilogy(t_id, rel_linf_single_t, label="Single vs coupled")
    ax.semilogy(t_id, rel_linf_t, label="NN corrected vs coupled")

    ax.axvline(train_end, linestyle="--", linewidth=1.0, label="train/val split")
    ax.axvline(val_end, linestyle=":", linewidth=1.0, label="val/test split")

    ax.set_xlabel("Time index")
    ax.set_ylabel("Relative Linf error")
    ax.set_title(f"Relative Linf error ({mode})")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()

    plt.tight_layout()
    plt.savefig(fig_dir / f"relative_linf_error_{mode}.png", dpi=300)
    plt.close()


plot_error_curves(
    rel_l2_single_t,
    rel_l2_t,
    rel_linf_single_t,
    rel_linf_t,
    fig_dir,
    MODE
)


# ============================================================
# 19. Plot field comparison
# ============================================================

def plot_field_comparison(
    grid_x,
    grid_y,
    u_single,
    u_pred,
    u_couple,
    time_id,
    fig_dir,
    mode
):
    vmin = min(
        np.min(u_single[time_id]),
        np.min(u_pred[time_id]),
        np.min(u_couple[time_id])
    )
    vmax = max(
        np.max(u_single[time_id]),
        np.max(u_pred[time_id]),
        np.max(u_couple[time_id])
    )

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    fields = [
        u_single[time_id],
        u_pred[time_id],
        u_couple[time_id],
        u_pred[time_id] - u_couple[time_id],
    ]

    titles = [
        "Decoupled single",
        "NN corrected",
        "Coupled reference",
        "Error: corrected - coupled",
    ]

    for i, ax in enumerate(axes):
        if i < 3:
            cs = ax.contourf(grid_x, grid_y, fields[i], levels=50, vmin=vmin, vmax=vmax)
        else:
            cs = ax.contourf(grid_x, grid_y, fields[i], levels=50)

        ax.set_title(titles[i])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        fig.colorbar(cs, ax=ax)

    plt.suptitle(f"Field comparison at time index {time_id} ({mode})")
    plt.tight_layout()
    plt.savefig(fig_dir / f"field_comparison_t{time_id}_{mode}.png", dpi=300)
    plt.close()


plot_field_comparison(
    grid_x,
    grid_y,
    u_single_full,
    u_pred_full,
    u_couple,
    plot_time_id,
    fig_dir,
    MODE
)


# ============================================================
# 20. Save outputs
# ============================================================

np.savez(
    outdata_dir / f"NN_corrected_results_{MODE}.npz",
    u_pred_full=u_pred_full,
    u_single_full=u_single_full,
    u_couple=u_couple,
    aA_pred=aA_pred,
    aB_pred=aB_pred,
    aA_single=aA_single,
    aB_single=aB_single,
    aA_couple=aA_couple,
    aB_couple=aB_couple,
    rel_l2_time=rel_l2_t,
    rel_linf_time=rel_linf_t,
    rel_l2_single_time=rel_l2_single_t,
    rel_linf_single_time=rel_linf_single_t,
)

print(f"\nFigures saved to: {fig_dir}")
print(f"Results saved to: {outdata_dir / f'NN_corrected_results_{MODE}.npz'}")