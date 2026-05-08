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
        self.fig_dir = current_dir / "Fig"
        self.fig_dir.mkdir(exist_ok=True)

        self._init_parameter(para_file)
        self._init_mesh(data_dir)
        self._init_coefficient_2D()
        self._init_time_step()

        self._solver_dt = None
        self._solver = None
        self._A_csc = None

    def _main_line(self):
        start_time = time.time()
        u_init = np.zeros((self.nx_gc + 1, self.ny_gc + 1), dtype=np.float64)
        u_init = self._init_initial_condition(u_init)

        t = 0.0
        u_n = u_init.copy()

        while t < self.T_final:
            if t + self.dt >= self.T_final:
                self.dt = self.T_final - t
            t += self.dt
            u_n = self.implicit_euler_step(u_n)
            u_n = self._setup_boundary_conditions(u_n)

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

        print(f'Runtime: {runtime:.3f}s')

    def _setup_boundary_conditions(self, u):
        if self.bc_type == 'neumann':
            u[0, :] = u[1, :]
            u[:, 0] = u[:, 1]
            u[-1, :] = u[-2, :]
            u[:, -1] = u[:, -2]
        elif self.bc_type == 'IF':
            pass
        return u

    def _utils_idx(self, i, j, ny_grid):
        # ny_i = self.ny_gc - 1
        return (i - 1) * ny_grid + j - 1

    def _build_interior_operator(self):
        nx_grid = self.nx_gc - 1
        ny_grid = self.ny_gc - 1
        n_unknowns = nx_grid * ny_grid
        aP = 1.0 / self.dt + self.aP_base

        A = lil_matrix((n_unknowns, n_unknowns), dtype=np.float64)

        for i in range(self.gc + 1, nx_grid + 1):
            for j in range(self.gc + 1, ny_grid + 1):
                p = self._utils_idx(i, j, ny_grid)     # return the index in the flattened of the discrete eq. for each cell(including the gc)
                A[p, p] = aP

                if i > self.gc + 1:
                    A[p, self._utils_idx(i - 1, j, ny_grid)] = self.aW
                if i < nx_grid:
                    A[p, self._utils_idx(i + 1, j, ny_grid)] = self.aE
                if j > self.gc + 1:
                    A[p, self._utils_idx(i, j - 1, ny_grid)] = self.aS
                if j < ny_grid:
                    A[p, self._utils_idx(i, j + 1, ny_grid)] = self.aN

        return A

    def _apply_boundary_contribution(self, A):
        nx_grid = self.nx_gc - 1
        ny_grid = self.ny_gc - 1

        if self.bc_type == 'neumann':
            for i in range(self.gc + 1, nx_grid + 1):
                for j in range(self.gc + 1, ny_grid + 1):
                    p = self._utils_idx(i, j, ny_grid)
                    if i == self.gc + 1:
                        A[p, p] += self.aW
                    if i == nx_grid:
                        A[p, p] += self.aE
                    if j == self.gc + 1:
                        A[p, p] += self.aS
                    if j == ny_grid:
                        A[p, p] += self.aN
            return A

        if self.bc_type == 'IF':
            return self._apply_interface_contribution(A)

        raise ValueError(f'Unsupported bc_type: {self.bc_type}')

    def _apply_interface_contribution(self, A):
        raise NotImplementedError('Schwarz interface contribution is not included yet.')

    def _build_sparse_solver(self):
        self._solver_dt = self.dt
        A = self._build_interior_operator()
        A = self._apply_boundary_contribution(A)
        self._A_csc = A.tocsc()
        self._solver = factorized(self._A_csc)


    def implicit_euler_step(self, u_n):
        if self._solver is None or self._solver_dt is None or abs(self.dt - self._solver_dt) > 1.0e-15:
            self._build_sparse_solver()
        u_old = self._setup_boundary_conditions(u_n.copy())

        rhs = (u_old[1:-1, 1:-1] / self.dt).reshape(-1)
        sol = self._solver(rhs)

        u_new = u_old.copy()
        nx_i = self.nx_gc - 1
        ny_i = self.ny_gc - 1
        u_new[1:-1, 1:-1] = sol.reshape(nx_i, ny_i)
        u_new = self._setup_boundary_conditions(u_new)
        return u_new

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
        # min_dx = np.min(self.dx)
        # min_dy = np.min(self.dy)
        min_dx = self.hx
        min_dy = self.hy
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
        self.hx = (self.x_max - self.x_min)/self.nx
        self.hy = (self.y_max - self.y_min)/self.ny
        self.nx_gc = self.nx + 2 * self.gc
        self.ny_gc = self.ny + 2 * self.gc


        x_coords = np.linspace(self.x_min - self.hx * self.gc, self.x_max + self.hx * self.gc, self.nx_gc + 1, dtype=np.float64)
        y_coords = np.linspace(self.y_min - self.hy * self.gc, self.y_max + self.hy * self.gc, self.ny_gc + 1, dtype=np.float64)

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
