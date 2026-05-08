import json
import pickle
import time
from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import factorized


class ADRSolver:
    def __init__(self, device, data_dir, para_file):
        self.device = device
        current_dir = Path(__file__).parent
        self.fig_dir = current_dir / 'Fig'
        self.fig_dir.mkdir(exist_ok=True)

        self._init_parameter(para_file)
        self._init_mesh(data_dir)
        self._init_coefficient_2D()
        self._init_time_step()

        self._solver_dt = None
        self._solver = None
        self._A_csc = None
        self.dtau = 0.1 * self.dt
        self.max_pseudo_iter = 5000

    def _main_line(self):
        start_time = time.time()
        u_init = np.zeros((self.nx + 1, self.ny + 1), dtype=np.float64)
        u_init = self._init_initial_condition(u_init)

        t = 0.0
        iter = 0
        u_n = u_init.copy()

        while t < self.T_final:
            if t + self.dt >= self.T_final:
                self.dt = self.T_final - t
            t += self.dt
            u_n = self.implicit_euler_step(u_n, interface_data=None)
            iter += 1
        runtime = time.time() - start_time

        fig, ax = plt.subplots()
        im = ax.imshow(
            u_n.T,
            origin='lower',
            aspect='auto',
            extent=[self.grid_x.min(), self.grid_x.max(), self.grid_y.min(), self.grid_y.max()],
        )
        plt.colorbar(im, label='U')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.savefig(self.fig_dir / 'u_final.pdf', dpi=300, bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots()
        extent_vals = [self.grid_x.min(), self.grid_x.max(), self.grid_y.min(), self.grid_y.max()]
        im = ax.contourf(u_n.T, levels=50, extent=extent_vals, origin='lower', cmap='jet')
        plt.colorbar(im, label='U')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.savefig(self.fig_dir / f'{self.domain}_u_final_contourf.pdf', dpi=300, bbox_inches='tight')
        plt.close()

        print(f'Runtime: {runtime:.3f}s, time step iteration: {iter}')

    def _setup_boundary_conditions(self, u):
        if self.bc_type == 'neumann':
            u[0, :] = u[1, :]
            u[:, 0] = u[:, 1]
            u[-1, :] = u[-2, :]
            u[:, -1] = u[:, -2]
        elif self.bc_type == 'IF':
            u[:, 0] = u[:, 1]
            u[:, -1] = u[:, -2]
            if self.IfacePos == 'Right':
                u[0, :] = u[1, :]
            elif self.IfacePos == 'Left':
                u[-1, :] = u[-2, :]
        return u

    def _utils_idx(self, i, j, ny_i=None):
        if ny_i is None:
            ny_i = self.ny - 1
        return (i - 1) * ny_i + (j - 1)

    def _build_interior_operator(self):
        nx_i = self.nx - 1
        ny_i = self.ny - 1
        n_unknowns = nx_i * ny_i
        aP = 1.0 / self.dt + 1.0 / self.dtau + self.aP_base

        A = lil_matrix((n_unknowns, n_unknowns), dtype=np.float64)

        for i in range(1, nx_i + 1):
            for j in range(1, ny_i + 1):
                p = self._utils_idx(i, j, ny_i)
                A[p, p] = aP

                if i > 1:
                    A[p, self._utils_idx(i - 1, j, ny_i)] = self.aW
                if i < nx_i:
                    A[p, self._utils_idx(i + 1, j, ny_i)] = self.aE
                if j > 1:
                    A[p, self._utils_idx(i, j - 1, ny_i)] = self.aS
                if j < ny_i:
                    A[p, self._utils_idx(i, j + 1, ny_i)] = self.aN

        return A

    def _apply_boundary_contribution(self, A):
        nx_i = self.nx - 1
        ny_i = self.ny - 1

        if self.bc_type == 'neumann':
            for i in range(1, nx_i + 1):
                for j in range(1, ny_i + 1):
                    p = self._utils_idx(i, j, ny_i)
                    if i == 1:
                        A[p, p] += self.aW
                    if i == nx_i:
                        A[p, p] += self.aE
                    if j == 1:
                        A[p, p] += self.aS
                    if j == ny_i:
                        A[p, p] += self.aN
            return A

        if self.bc_type == 'IF':
            for i in range(1, nx_i + 1):
                for j in range(1, ny_i + 1):
                    p = self._utils_idx(i, j, ny_i)
                    if j == 1:
                        A[p, p] += self.aS
                    if j == ny_i:
                        A[p, p] += self.aN

            if self.IfacePos == 'Right':
                for j in range(1, ny_i + 1):
                    p = self._utils_idx(1, j, ny_i)
                    A[p, p] += self.aW
            elif self.IfacePos == 'Left':
                for j in range(1, ny_i + 1):
                    p = self._utils_idx(nx_i, j, ny_i)
                    A[p, p] += self.aE
            else:
                raise ValueError(f'Unsupported IfacePos: {self.IfacePos}')

            A = self._apply_interface_contribution(A)
            return A

        raise ValueError(f'Unsupported bc_type: {self.bc_type}')
    
    def _apply_interface_contribution(self, A):
        ny_i = self.ny - 1
        row_data = self._utils_get_interface_row_data()
        i = row_data['i_row']

        for j in range(1, ny_i + 1):
            p = self._utils_idx(i, j, ny_i)
            A[p, :] = 0.0
            A[p, p] = row_data['A_P']

            if row_data['horiz_label'] == 'W':
                if i <= 1:
                    raise ValueError('Right-interface row requires an interior west neighbor.')
                q_h = self._utils_idx(i - 1, j, ny_i)
                A[p, q_h] = row_data['A_horiz']
            else:
                if i >= self.nx - 1:
                    raise ValueError('Left-interface row requires an interior east neighbor.')
                q_h = self._utils_idx(i + 1, j, ny_i)
                A[p, q_h] = row_data['A_horiz']

            if j > 1:
                A[p, self._utils_idx(i, j - 1, ny_i)] = self.aS
            else:
                A[p, p] += self.aS

            if j < ny_i:
                A[p, self._utils_idx(i, j + 1, ny_i)] = self.aN
            else:
                A[p, p] += self.aN

        return A

    def _utils_get_interface_row_data(self):
        base_diag = (
            1.0 / self.dt
            + 2.0 * self.mu / (self.hx ** 2)
            + 2.0 * self.lambd / self.hx
            + 2.0 * self.mu / (self.hy ** 2)
            + abs(self.beta_y) / self.hy
            + self.sigma
        )

        if self.IfacePos == 'Right':
            if self.beta_x >= 0.0:
                return {
                    'side': 'Right',
                    'i_row': self.nx - 1,
                    'horiz_label': 'W',
                    'A_horiz': -2.0 * self.mu / (self.hx ** 2) - self.beta_x / self.hx,
                    'A_P': base_diag + self.beta_x / self.hx,
                    'R': 2.0 / self.hx,
                }

            return {
                'side': 'Right',
                'i_row': self.nx - 1,
                'horiz_label': 'W',
                'A_horiz': -2.0 * self.mu / (self.hx ** 2) + self.beta_x / self.hx,
                'A_P': base_diag - self.beta_x * (1.0 / self.hx + 2.0 * self.lambd / self.mu),
                'R': 2.0 / self.hx - 2.0 * self.beta_x / self.mu,
            }

        if self.IfacePos == 'Left':
            if self.beta_x >= 0.0:
                return {
                    'side': 'Left',
                    'i_row': 1,
                    'horiz_label': 'E',
                    'A_horiz': -2.0 * self.mu / (self.hx ** 2) - self.beta_x / self.hx,
                    'A_P': base_diag + self.beta_x * (1.0 / self.hx + 2.0 * self.lambd / self.mu),
                    'R': 2.0 / self.hx + 2.0 * self.beta_x / self.mu,
                }

            return {
                'side': 'Left',
                'i_row': 1,
                'horiz_label': 'E',
                'A_horiz': -2.0 * self.mu / (self.hx ** 2) + self.beta_x / self.hx,
                'A_P': base_diag - self.beta_x / self.hx,
                'R': 2.0 / self.hx,
            }

        raise ValueError(f'Unsupported IfacePos: {self.IfacePos}')


    def _utils_normalize_interface_data(self, interface_data):
        ny_i = self.ny - 1

        if interface_data is None:
            return np.zeros(ny_i, dtype=np.float64)

        if np.isscalar(interface_data):
            return np.full(ny_i, float(interface_data), dtype=np.float64)

        g = np.asarray(interface_data, dtype=np.float64).reshape(-1)
        if g.size == ny_i:
            return g.copy()
        if g.size == self.ny + 1:
            return g[1:-1].copy()

        raise ValueError(
            'interface_data must be None, a scalar, length (ny-1), or length (ny+1).'
        )

    def _build_rhs(self, u_old, interface_data=None):
        rhs = (u_old[1:-1, 1:-1] / self.dt).reshape(-1)

        if self.bc_type == 'IF':
            row_data = self._utils_get_interface_row_data()
            g_vals = self._utils_normalize_interface_data(interface_data)
            ny_i = self.ny - 1
            i = row_data['i_row']

            for j in range(1, ny_i + 1):
                p = self._utils_idx(i, j, ny_i)
                rhs[p] += row_data['R'] * g_vals[j - 1]

        return rhs
    
    def _build_rhs_dual_time(self, u_pseudo_old, u_physical_old, interface_data=None):
        rhs = (u_physical_old[1:-1, 1:-1] / self.dt).reshape(-1) + (u_pseudo_old[1:-1, 1:-1] / self.dtau).reshape(-1)

        if self.bc_type == 'IF':
            row_data = self._utils_get_interface_row_data()
            g_vals = self._utils_normalize_interface_data(interface_data)
            ny_i = self.ny - 1
            i = row_data['i_row']

            for j in range(1, ny_i + 1):
                p = self._utils_idx(i, j, ny_i)
                rhs[p] += row_data['R'] * g_vals[j - 1]

        return rhs


    def _build_sparse_solver(self):
        self._solver_dt = self.dt
        A = self._build_interior_operator()
        A = self._apply_boundary_contribution(A)
        self._A_csc = A.tocsc()
        self._solver = factorized(self._A_csc)

    def _ensure_sparse_solver(self):
        if self._solver is None or self._solver_dt is None or abs(self.dt - self._solver_dt) > 1.0e-15:
            self._build_sparse_solver()

    def implicit_euler_step(self, u_n, interface_data=None):
        self._ensure_sparse_solver()
        u_old = self._setup_boundary_conditions(u_n.copy())
        # for it in range(5000):

        rhs = self._build_rhs(u_old, interface_data=interface_data)
        sol = self._solver(rhs)

        u_new = u_old.copy()
        nx_i = self.nx - 1
        ny_i = self.ny - 1
        u_new[1:-1, 1:-1] = sol.reshape(nx_i, ny_i)
        u_new = self._setup_boundary_conditions(u_new)

            # error = np.linalg.norm(u_old - u_new, ord=2)

            # u_old = self._setup_boundary_conditions(u_new.copy())
            # print(f'Iteration {it+1}, Error: {error.item()}')
            # if error < 1e-6:
            #     print(f'Convergence achieved! Interation : {it+1}, error = {error:.3e}')
            #     break
        return u_new
    
    def implicit_euler_step(self, u_n, interface_data=None):
        self._ensure_sparse_solver()

        u_physical_old = self._setup_boundary_conditions(u_n.copy())
        u_pseudo_old = u_physical_old.copy()

        for it in range(self.max_pseudo_iter):
            rhs = self._build_rhs_dual_time(
                u_pseudo_old=u_pseudo_old,
                u_physical_old=u_physical_old,
                interface_data=interface_data,
            )

            sol = self._solver(rhs)

            u_new = u_pseudo_old.copy()
            nx_i = self.nx - 1
            ny_i = self.ny - 1
            u_new[1:-1, 1:-1] = sol.reshape(nx_i, ny_i)

            u_new = self._setup_boundary_conditions(u_new)

            error = np.linalg.norm(u_new - u_pseudo_old, ord=2)
            u_pseudo_old = u_new.copy()

            if error < 1e-6:
                break

        return u_pseudo_old

    def _init_initial_condition(self, u):
        u = np.cos(2.0 * np.pi * self.grid_x) * np.cos(2.0 * np.pi * self.grid_y)

        fig, ax = plt.subplots()
        im = ax.imshow(
            u.T,
            origin='lower',
            aspect='auto',
            extent=[self.grid_x.min(), self.grid_x.max(), self.grid_y.min(), self.grid_y.max()],
        )
        plt.colorbar(im, label='U0')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.savefig(self.fig_dir / f'{self.domain}_u_initial.pdf', dpi=300, bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots()
        extent_vals = [self.grid_x.min(), self.grid_x.max(), self.grid_y.min(), self.grid_y.max()]
        im = ax.contourf(u.T, levels=50, extent=extent_vals, origin='lower', cmap='jet')
        plt.colorbar(im, label='U')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.savefig(self.fig_dir / f'{self.domain}_u_initial_contourf.pdf', dpi=300, bbox_inches='tight')
        plt.close()

        return u

    def _init_time_step(self):
        min_dx = np.min(self.dx)
        min_dy = np.min(self.dy)
        dd_min = min(min_dx, min_dy)

        dt_explicit_diffusion = dd_min ** 2 / (4.0 * self.mu + 1.0e-30)
        dt_explicit_advection = 1.0 / (abs(self.beta_x) / min_dx + abs(self.beta_y) / min_dy + 1.0e-30)
        dt_explicit_reaction = 1.0 / (self.sigma + 1.0e-30)
        dt_explicit = 1.0 / (
            1.0 / dt_explicit_advection + 1.0 / dt_explicit_diffusion + 1.0 / dt_explicit_reaction
        )
        self.dt = 5.0 * self.cfl * dt_explicit

        print(f'dtau per itera: {self.dt}')

    def _init_coefficient_2D(self):
        self.hx = self.dx[0, 0]
        self.hy = self.dy[0, 0]

        self.beta_x_plus = max(self.beta_x, 0.0)
        self.beta_x_minus = min(self.beta_x, 0.0)
        self.beta_y_plus = max(self.beta_y, 0.0)
        self.beta_y_minus = min(self.beta_y, 0.0)

        self.aW = -self.mu / (self.hx ** 2) - self.beta_x_plus / self.hx
        self.aE = -self.mu / (self.hx ** 2) + self.beta_x_minus / self.hx
        self.aS = -self.mu / (self.hy ** 2) - self.beta_y_plus / self.hy
        self.aN = -self.mu / (self.hy ** 2) + self.beta_y_minus / self.hy
        self.aP_base = (
            2.0 * self.mu / (self.hx ** 2)
            + 2.0 * self.mu / (self.hy ** 2)
            + abs(self.beta_x) / self.hx
            + abs(self.beta_y) / self.hy
            + self.sigma
        )

    def _init_mesh(self, data_dir):
        x_coords = np.linspace(self.x_min, self.x_max, self.nx + 1, dtype=np.float64)
        y_coords = np.linspace(self.y_min, self.y_max, self.ny + 1, dtype=np.float64)

        self.grid_x, self.grid_y = np.meshgrid(x_coords, y_coords, indexing='ij')
        self.grid = np.stack([self.grid_x, self.grid_y], axis=-1)

        self.dx = self.grid_x[1:, :] - self.grid_x[:-1, :]
        self.dy = self.grid_y[:, 1:] - self.grid_y[:, :-1]

        mesh = {
            'grid': self.grid,
            'grid_x': self.grid_x,
            'grid_y': self.grid_y,
            'dx': self.dx,
            'dy': self.dy,
        }
        with open(data_dir / f'{self.domain}_Mesh.pkl', 'wb') as f:
            pickle.dump(mesh, f)

    def _init_parameter(self, param_filename):
        def convert_fraction_strings(obj):
            if isinstance(obj, dict):
                return {k: convert_fraction_strings(v) for k, v in obj.items()}
            if isinstance(obj, str) and '/' in obj and obj.replace('/', '').replace('-', '').isdigit():
                return float(Fraction(obj))
            return obj

        with open(param_filename, 'r') as f:
            param_dict = json.load(f)
            param_dict = convert_fraction_strings(param_dict)

        self.domain = param_dict['domain']
        self.mu = float(param_dict['mu'])
        self.beta_x = float(param_dict['beta_x'])
        self.beta_y = float(param_dict['beta_y'])
        self.sigma = float(param_dict['sigma'])
        self.lambd = float(param_dict['lambda'])

        self.nx = int(param_dict['nx'])
        self.ny = int(param_dict['ny'])
        self.x_min = float(param_dict['x_min'])
        self.x_max = float(param_dict['x_max'])
        self.y_min = float(param_dict['y_min'])
        self.y_max = float(param_dict['y_max'])
        self.gc = int(param_dict['ghostcell'])
        self.cfl = float(param_dict['cfl'])
        self.T_final = float(param_dict['T_final'])
        self.bc_type = param_dict['bc_type']
        self.IfacePos = param_dict['IfacePos']
