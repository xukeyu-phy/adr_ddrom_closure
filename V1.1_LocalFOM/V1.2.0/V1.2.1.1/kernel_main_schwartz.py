__all__ = ['device', 'dtype']

import torch
import numpy as np
from kernel_sovle import ADRSolver
from pathlib import Path



current_dir = Path(__file__).parent
outdata_dir = current_dir / "Out_data"
outdata_dir.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
torch.set_default_dtype(dtype)




def Schwartz_Iterator(Asolver, u_A, u_IF_A, Bsolver, u_B, u_IF_B, max_iter, tolerance):
    u_A_old = u_A.copy()
    u_B_old = u_B.copy()
    for it in range(max_iter):
        u_A = Asolver.implicit_euler_step(u_A_old, interface_data=u_IF_B)
        u_IF_A['u_b'] = u_A[-1, :]
        u_IF_A['u_b1'] = u_A[-2, :]
        u_IF_A['u_b2'] = u_A[-3, :]

        u_B = Bsolver.implicit_euler_step(u_B_old, interface_data=u_IF_A)
        u_IF_B['u_b'] = u_B[0, :]
        u_IF_B['u_b1'] = u_B[1, :]
        u_IF_B['u_b2'] = u_B[2, :]
        # u_B = Asolver._setup_boundary_conditions(u_B)


        # compute the error between A_u and B_u at the interface
        error = np.linalg.norm(u_IF_A['u_b'] - u_IF_B['u_b'], ord=2)
        # print(f'Schwartz Iteration {it+1}, Error: {error.item()}')
        if error < tolerance:
            print(f'Convergence achieved! -- Schwartz Iteration {it+1}, Error: {error.item()}')
            break

    return u_A, u_B



def main():
    A_file = current_dir / 'subdomain_A.json'
    A_adrsolver = ADRSolver(device, outdata_dir, A_file)
    u_A = np.zeros((A_adrsolver.nx + 1, A_adrsolver.ny + 1), dtype=np.float64)
    u_A = A_adrsolver._init_initial_condition(u_A)
    u_IF_A = {'u_b': u_A[-1, :],
              'u_b1': u_A[-2, :],
              'u_b2': u_A[-3, :],
              'mu_if': A_adrsolver.mu,
              'lambd_if': A_adrsolver.lambd}


    B_file = current_dir / 'subdomain_B.json'
    B_adrsolver = ADRSolver(device, outdata_dir, B_file)
    u_B = np.zeros((B_adrsolver.nx + 1, B_adrsolver.ny + 1), dtype=np.float64)
    u_B = B_adrsolver._init_initial_condition(u_B)
    u_IF_B = {'u_b': u_B[0, :],
              'u_b1': u_B[1, :],
              'u_b2': u_B[2, :],
              'mu_if': B_adrsolver.mu}

    dt = min(A_adrsolver.dt, B_adrsolver.dt) * 1.0
    A_adrsolver.dt = dt
    B_adrsolver.dt = dt
    print(f'dtau per itera: {dt}')
    # u = adrsolver._main_line()

    t = 0.0
    T_final = 0.05
    iter = 0
    u = np.concatenate((u_A[:, :], u_B[1:, :]), axis=0)
    grid_x = np.concatenate((A_adrsolver.grid_x[:, :], B_adrsolver.grid_x[1:, :]), axis=0)
    grid_y = np.concatenate((A_adrsolver.grid_y[:, :], B_adrsolver.grid_y[1:, :]), axis=0)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    extent_vals = [grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()]
    im = ax.contourf(u.T, levels=50, extent=extent_vals, origin='lower', cmap='jet')
    plt.colorbar(im, label='U')
    fig_dir = current_dir / 'Fig-coupled'
    fig_dir.mkdir(exist_ok=True)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.savefig(fig_dir / f'FULL_u_{iter}.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    while t < T_final:
        if t + dt >= T_final:
            dt = T_final - t
        t += dt
        u_A, u_B = Schwartz_Iterator(A_adrsolver, u_A, u_IF_A, B_adrsolver, u_B, u_IF_B, max_iter=100, tolerance=1e-6)
        iter += 1
        u = np.concatenate((u_A[:, :], u_B[1:, :]), axis=0)
        grid_x = np.concatenate((A_adrsolver.grid_x[:, :], B_adrsolver.grid_x[1:, :]), axis=0)
        grid_y = np.concatenate((A_adrsolver.grid_y[:, :], B_adrsolver.grid_y[1:, :]), axis=0)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        extent_vals = [grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()]
        im = ax.contourf(u.T, levels=50, extent=extent_vals, origin='lower', cmap='jet')
        plt.colorbar(im, label='U')
        # fig_dir = current_dir / 'Fig'
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.savefig(fig_dir / f'FULL_u_{iter}.pdf', dpi=300, bbox_inches='tight')
        plt.close()

    fig, ax = plt.subplots()
    extent_vals = [A_adrsolver.grid_x.min(), A_adrsolver.grid_x.max(), A_adrsolver.grid_y.min(), A_adrsolver.grid_y.max()]
    im = ax.contourf(u_A.T, levels=50, extent=extent_vals, origin='lower', cmap='jet')
    plt.colorbar(im, label='U')
    # fig_dir = current_dir / 'Fig'
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.savefig(fig_dir / f'FULL_uA.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots()
    extent_vals = [B_adrsolver.grid_x.min(), B_adrsolver.grid_x.max(), B_adrsolver.grid_y.min(), B_adrsolver.grid_y.max()]
    im = ax.contourf(u_B.T, levels=50, extent=extent_vals, origin='lower', cmap='jet')
    plt.colorbar(im, label='U')
    # fig_dir = current_dir / 'Fig'
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.savefig(fig_dir / f'FULL_uB.pdf', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
