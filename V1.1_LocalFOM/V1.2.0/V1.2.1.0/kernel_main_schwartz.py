__all__ = ['device', 'dtype']

import torch
import numpy as np
from kernel_sovle_implicit_interface_sch import ADRSolver
from pathlib import Path



current_dir = Path(__file__).parent
outdata_dir = current_dir / "Out_data"
outdata_dir.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
torch.set_default_dtype(dtype)




def Schwartz_Iterator(Asolver, u_A, Bsolver, u_B, dt, max_iter, tolerance):
    for it in range(max_iter):
        u_IF_B = u_B[0, :]
        u_A = Asolver.implicit_euler_step(u_A, interface_data=u_IF_B)
        # u_A = Asolver._setup_boundary_conditions(u_A)

        u_IF_A = u_A[-1, :]
        u_B = Bsolver.implicit_euler_step(u_B, interface_data=u_IF_A)
        # u_B = Asolver._setup_boundary_conditions(u_B)


        # compute the error between A_u and B_u at the interface
        u_IF_A = u_A[-1, :]
        u_IF_B = u_B[0, :]
        error = np.linalg.norm(u_IF_A - u_IF_B, ord=2)
        print(f'Iteration {it+1}, Error: {error.item()}')
        if error < tolerance:
            print('Convergence achieved!')
            break

        return u_A, u_B



def main():
    A_file = current_dir / 'subdomain_A.json'
    A_adrsolver = ADRSolver(device, outdata_dir, A_file)
    u_A = np.zeros((A_adrsolver.nx + 1, A_adrsolver.ny + 1), dtype=np.float64)
    u_A = A_adrsolver._init_initial_condition(u_A)

    B_file = current_dir / 'subdomain_B.json'
    B_adrsolver = ADRSolver(device, outdata_dir, B_file)
    u_B = np.zeros((B_adrsolver.nx + 1, B_adrsolver.ny + 1), dtype=np.float64)
    u_B = B_adrsolver._init_initial_condition(u_B)

    dt = min(A_adrsolver.dt, B_adrsolver.dt) * 2.0
    print(f'dtau per itera: {dt}')
    # u = adrsolver._main_line()

    t = 0.0
    T_final = 0.05
    iter = 0

    while t < T_final:
        if t + dt >= T_final:
            dt = T_final - t
        t += dt
        u_A, u_B = Schwartz_Iterator(A_adrsolver, u_A, B_adrsolver, u_B, dt, max_iter=100, tolerance=1e-6)




if __name__ == "__main__":
    main()
