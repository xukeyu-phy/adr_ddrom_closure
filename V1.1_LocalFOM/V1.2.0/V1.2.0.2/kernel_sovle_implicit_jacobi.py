import torch
import time
import json
from pathlib import Path
from fractions import Fraction


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

    def _main_line(self):
        start_time = time.time()
        u_init = torch.zeros((int(self.nx.item()) + 1, int(self.ny.item()) + 1),
                             device=self.device, dtype=torch.float64)
        u_init = self._init_initial_condition(u_init)

        t = 0.0
        t_iter = 0
        u_n = u_init.clone()

        while t < float(self.T_final):
            if (t + float(self.dt)) >= float(self.T_final):
                self.dt = torch.tensor(float(self.T_final) - t, device=self.device, dtype=torch.float64)
            t += float(self.dt)
            t_iter += 1
            u_n = self.runge_kutta_1_step(u_n)
            u_n = self._setup_boundary_conditions(u_n)

        end_time = time.time()
        runtime = end_time - start_time

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        im = ax.imshow(u_n.T.cpu().numpy(), origin='lower', aspect='auto',
                       extent=[self.grid_x.cpu().numpy().min(), self.grid_x.cpu().numpy().max(),
                               self.grid_y.cpu().numpy().min(), self.grid_y.cpu().numpy().max()])
        plt.colorbar(im, label='U')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.savefig(self.fig_dir / "u_final.pdf", dpi=300, bbox_inches="tight")
        plt.close()

        fig, ax = plt.subplots()
        extent_vals = [
            self.grid_x.cpu().numpy().min(), self.grid_x.cpu().numpy().max(),
            self.grid_y.cpu().numpy().min(), self.grid_y.cpu().numpy().max()
        ]
        im = ax.contourf(u_n.T.cpu().numpy(), levels=50, extent=extent_vals, origin='lower', cmap='jet')
        plt.colorbar(im, label='U')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.savefig(self.fig_dir / f"{self.domain}_u_final_contourf.pdf", dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Runtime: {runtime:.3f}s")

    def _setup_boundary_conditions(self, u):
        if self.bc_type == 'neumann':
            u[0, :] = u[1, :]
            u[:, 0] = u[:, 1]
            u[-1, :] = u[-2, :]
            u[:, -1] = u[:, -2]
        elif self.bc_type == 'IF':
            pass
        return u

    def rhs(self, u):
        diffusion = self.rhs_diffusion(u)
        convection = self.rhs_convection(u)
        source = self.rhs_source(u)

        spatial_rhs = torch.zeros_like(u)
        interior = slice(1, -1)
        spatial_rhs[interior, interior] = (
            diffusion[interior, interior]
            - convection[interior, interior]
            - source[interior, interior]
        )
        return spatial_rhs

    def rhs_source(self, u):
        return self.sigma * u

    def rhs_convection(self, u):
        # retained only for compatibility / reference to original explicit structure
        u_c = u[1:-1, 1:-1]
        u_w = u[:-2, 1:-1]
        u_e = u[2:, 1:-1]
        u_n = u[1:-1, 2:]
        u_s = u[1:-1, :-2]

        convection = torch.zeros_like(u, device=self.device)
        convection[1:-1, 1:-1] = (
            self.beta_x_plus * (u_c - u_w) / self.hx
            + self.beta_x_minus * (u_e - u_c) / self.hx
            + self.beta_y_plus * (u_c - u_s) / self.hy
            + self.beta_y_minus * (u_n - u_c) / self.hy
        )
        return convection

    def rhs_diffusion(self, u):
        u_c = u[1:-1, 1:-1]
        u_w = u[:-2, 1:-1]
        u_e = u[2:, 1:-1]
        u_n = u[1:-1, 2:]
        u_s = u[1:-1, :-2]

        nabla2 = torch.zeros_like(u)
        nabla2[1:-1, 1:-1] = self.mu * (
            (u_e - 2.0 * u_c + u_w) / (self.hx ** 2)
            + (u_n - 2.0 * u_c + u_s) / (self.hy ** 2)
        )
        return nabla2

    def runge_kutta_1_step(self, u_n):
        # Name kept unchanged. Internally this is Backward Euler solved by weighted Jacobi.
        self._refresh_diag_for_current_dt()
        u_old = self._setup_boundary_conditions(u_n.clone())
        u_new = u_old.clone()

        rhs = u_old[1:-1, 1:-1] / self.dt
        diag = self._diag_interior.clone()

        max_iter = 5000
        tol = 1.0e-10
        omega = 0.4

        for _ in range(max_iter):
            u_prev = u_new.clone()

            west = u_prev[:-2, 1:-1]
            east = u_prev[2:, 1:-1]
            south = u_prev[1:-1, :-2]
            north = u_prev[1:-1, 2:]

            numer = rhs - self.aW * west - self.aE * east - self.aS * south - self.aN * north
            u_jac = numer / diag

            u_new[1:-1, 1:-1] = (1.0 - omega) * u_prev[1:-1, 1:-1] + omega * u_jac
            u_new = self._setup_boundary_conditions(u_new)

            err = torch.max(torch.abs(u_new - u_prev))
            if float(err) < tol:
                print(f'Convergence achieved after {_+1} iterations with error {err.item():.2e}')
                
                break

        return u_new

    def _init_initial_condition(self, u):
        u = torch.cos(2 * torch.pi * self.grid_x) * torch.cos(2 * torch.pi * self.grid_y)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        im = ax.imshow(u.T.cpu().numpy(), origin='lower', aspect='auto',
                       extent=[self.grid_x.cpu().numpy().min(), self.grid_x.cpu().numpy().max(),
                               self.grid_y.cpu().numpy().min(), self.grid_y.cpu().numpy().max()])
        plt.colorbar(im, label='U0')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.savefig(self.fig_dir / f"{self.domain}_u_initial.pdf", dpi=300, bbox_inches="tight")
        plt.close()

        fig, ax = plt.subplots()
        extent_vals = [
            self.grid_x.cpu().numpy().min(), self.grid_x.cpu().numpy().max(),
            self.grid_y.cpu().numpy().min(), self.grid_y.cpu().numpy().max()
        ]
        im = ax.contourf(u.T.cpu().numpy(), levels=50, extent=extent_vals, origin='lower', cmap='jet')
        plt.colorbar(im, label='U')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.savefig(self.fig_dir / f"{self.domain}_u_initial_contourf.pdf", dpi=300, bbox_inches="tight")
        plt.close()

        return u

    def _init_time_step(self):
        min_dx = torch.min(self.dx)
        min_dy = torch.min(self.dy)
        dd_min = min(min_dx, min_dy)

        # For implicit method, dt no longer needs the explicit CFL restriction for stability.
        # We keep a moderate default to avoid changing the user's outer logic too much.
        dt_explicit_diffusion = dd_min ** 2 / (4 * self.mu + 1e-30)
        dt_explicit_advection = 1.0 / (torch.abs(self.beta_x) / min_dx + torch.abs(self.beta_y) / min_dy + 1e-30)
        dt_explicit_reaction = 1.0 / (self.sigma + 1e-30)
        dt_explicit = 1.0 / (1.0 / dt_explicit_advection + 1.0 / dt_explicit_diffusion + 1.0 / dt_explicit_reaction)
        self.dt = 1.0 * self.cfl * dt_explicit

        self._refresh_diag_for_current_dt()
        print(f'dtau per itera: {self.dt}')

    def _init_coefficient_2D(self):
        self.hx = self.dx[0, 0].clone()
        self.hy = self.dy[0, 0].clone()

        self.beta_x_plus = torch.clamp(self.beta_x, min=0.0)
        self.beta_x_minus = torch.clamp(self.beta_x, max=0.0)
        self.beta_y_plus = torch.clamp(self.beta_y, min=0.0)
        self.beta_y_minus = torch.clamp(self.beta_y, max=0.0)

        self.aW = -self.mu / (self.hx ** 2) - self.beta_x_plus / self.hx
        self.aE = -self.mu / (self.hx ** 2) + self.beta_x_minus / self.hx
        self.aS = -self.mu / (self.hy ** 2) - self.beta_y_plus / self.hy
        self.aN = -self.mu / (self.hy ** 2) + self.beta_y_minus / self.hy
        self.aP_base = (
            2.0 * self.mu / (self.hx ** 2)
            + 2.0 * self.mu / (self.hy ** 2)
            + torch.abs(self.beta_x) / self.hx
            + torch.abs(self.beta_y) / self.hy
            + self.sigma
        )

        nx_i = int(self.nx.item()) - 1
        ny_i = int(self.ny.item()) - 1
        self._diag_interior = torch.full((nx_i, ny_i), float(1.0 / self.dt + self.aP_base),
                                         device=self.device, dtype=torch.float64) if hasattr(self, 'dt') else None

    def _refresh_diag_for_current_dt(self):
        nx_i = int(self.nx.item()) - 1
        ny_i = int(self.ny.item()) - 1
        self._diag_interior = torch.full((nx_i, ny_i), float(1.0 / self.dt + self.aP_base),
                                         device=self.device, dtype=torch.float64)
        if self.bc_type == 'neumann':
            self._diag_interior[0, :] += self.aW
            self._diag_interior[-1, :] += self.aE
            self._diag_interior[:, 0] += self.aS
            self._diag_interior[:, -1] += self.aN
        elif self.bc_type == 'IF':
            pass

    def _init_mesh(self, data_dir):
        nx_int = int(self.nx.item())
        ny_int = int(self.ny.item())
        x_coords = torch.linspace(float(self.x_min), float(self.x_max), nx_int + 1, device=self.device, dtype=torch.float64)
        y_coords = torch.linspace(float(self.y_min), float(self.y_max), ny_int + 1, device=self.device, dtype=torch.float64)

        self.grid_x, self.grid_y = torch.meshgrid(x_coords, y_coords, indexing='ij')
        self.grid = torch.stack([self.grid_x, self.grid_y], dim=-1)

        self.dx = self.grid_x[1:, :] - self.grid_x[:-1, :]
        self.dy = self.grid_y[:, 1:] - self.grid_y[:, :-1]

        if len(self.dx.shape) == 3:
            self.dx = self.dx.unsqueeze(-1)
        if len(self.dy.shape) == 3:
            self.dy = self.dy.unsqueeze(-1)

        Mesh = {
            'grid': self.grid.cpu(),
            'grid_x': self.grid_x.cpu(),
            'grid_y': self.grid_y.cpu(),
            'dx': self.dx.cpu(),
            'dy': self.dy.cpu()}
        torch.save(Mesh, data_dir / f"{self.domain}_Mesh.pt")

    def _init_parameter(self, param_filename):
        def convert_fraction_strings(obj):
            if isinstance(obj, dict):
                return {k: convert_fraction_strings(v) for k, v in obj.items()}
            elif isinstance(obj, str) and '/' in obj and obj.replace('/', '').replace('-', '').isdigit():
                return Fraction(obj)
            else:
                return obj

        with open(param_filename, 'r') as f:
            param_dict = json.load(f)
            param_dict = convert_fraction_strings(param_dict)

        self.domain = param_dict['domain']
        self.mu = torch.tensor(param_dict['mu'], device=self.device, dtype=torch.float64)
        self.beta_x = torch.tensor(param_dict['beta_x'], device=self.device, dtype=torch.float64)
        self.beta_y = torch.tensor(param_dict['beta_y'], device=self.device, dtype=torch.float64)
        self.sigma = torch.tensor(param_dict['sigma'], device=self.device, dtype=torch.float64)
        self.lambd = torch.tensor(param_dict['lambda'], device=self.device, dtype=torch.float64)

        self.nx = torch.tensor(param_dict['nx'], device=self.device, dtype=torch.int64)
        self.ny = torch.tensor(param_dict['ny'], device=self.device, dtype=torch.int64)
        self.x_min = torch.tensor(param_dict['x_min'], device=self.device, dtype=torch.float64)
        self.x_max = torch.tensor(param_dict['x_max'], device=self.device, dtype=torch.float64)
        self.y_min = torch.tensor(param_dict['y_min'], device=self.device, dtype=torch.float64)
        self.y_max = torch.tensor(param_dict['y_max'], device=self.device, dtype=torch.float64)
        self.gc = torch.tensor(param_dict['ghostcell'], device=self.device, dtype=torch.int64)
        self.cfl = torch.tensor(param_dict['cfl'], device=self.device, dtype=torch.float64)
        self.T_final = torch.tensor(param_dict['T_final'], device=self.device, dtype=torch.float64)
        self.bc_type = param_dict['bc_type']
        self.IfacePos = param_dict['IfacePos']

        # defer diag init until dt is available
        self._diag_interior = None
