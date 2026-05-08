import torch
import matplotlib.pyplot as plt
from pathlib import Path

class Config:
    def __init__(self, device, dtype):
        self.device = device
        self.dtype = dtype       
        torch.set_default_dtype(dtype)
                
        # Numerical Parameters  
        self.T_final = 0.05       
        self.cfl = 1.0 
        self.convergence_tol = 1e-6  # Convergence tolerance
        self.ghostcell = 0           # Number of ghost cells
        
        # non-uniform grids
        self.n1 = 60          
        self.n2 = 40
        # self.n3 = 20           
        self.grid_segments = [
            (0.00, 0.60),    # First segment
            (0.60, 1.00)]    # Second segment
          
        
        # Equal rational grid cells for HF simulation
        self.N = 100         

        # Boundary Conditions
        self.bc_type = 'neumann'  # Options: 'dirichlet' or 'neumann'
        self.bc_value = 0.125       # Boundary value for Dirichlet BC
        
        # Initial Conditions
        self.initial_condition_type = 'analytical' # Options: 'uniform' or 'analytical'
        self.initial_value = 0.125              # Initial value for uniform condition

    def _create_uniform_grid_2D(self, data_dir):
        data_dir.mkdir(exist_ok=True)
        segments = self.grid_segments
        n1, n2 = self.n1, self.n2
        gc = self.ghostcell

        coords_1 = torch.linspace(segments[0][0], segments[0][1], n1+1, 
                                 device=self.device, dtype=self.dtype)
        coords_2 = torch.linspace(segments[1][0], segments[1][1], n2+1, 
                                 device=self.device, dtype=self.dtype)  
        coords = torch.cat([coords_1, coords_2[1:]])

        if gc > 0:
            lower_spacing = coords[1] - coords[0]
            lower_ghost = torch.linspace(
                coords[0] - gc * lower_spacing, 
                coords[0] - lower_spacing, 
                gc,
                device=self.device, dtype=self.dtype)            
            upper_spacing = coords[-1] - coords[-2]
            upper_ghost = torch.linspace(
                coords[-1] + upper_spacing, 
                coords[-1] + gc * upper_spacing, 
                gc,
                device=self.device, dtype=self.dtype)
            coords = torch.cat([lower_ghost, coords, upper_ghost])

        x_coords = coords
        y_coords = coords

        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='ij')    
        coord_tensor = torch.stack([grid_x, grid_y], dim=-1)
        
        self.dx = grid_x[1:, :] - grid_x[:-1, :]
        self.dy = grid_y[:, 1:] - grid_y[:, :-1] 

        if len(self.dx.shape) == 2:
            self.dx = self.dx.unsqueeze(-1)
        if len(self.dy.shape) == 2:
            self.dy = self.dy.unsqueeze(-1)

        grid = coord_tensor
        self.Nx, self.Ny = grid.shape[0], grid.shape[1]

        Mesh = {
        'grid': coord_tensor.cpu(),
        'grid_x': grid_x.cpu(),
        'grid_y': grid_y.cpu(),
        'dx': self.dx.cpu(),
        'dy': self.dy.cpu()}
        torch.save(Mesh, data_dir/'Mesh_domain.pt')
        
        self.grid_x = grid_x
        self.grid_y = grid_y


        # ------------------- Subdomain -------------------
        x_coords_1 = coords_1
        grid_x_1, grid_y_1 = torch.meshgrid(x_coords_1, y_coords, indexing='ij')
        coord_tensor_1 = torch.stack([grid_x_1, grid_y_1], dim=-1)
        dx_1 = grid_x_1[1:, :] - grid_x_1[:-1, :]
        dy_1 = grid_y_1[:, 1:] - grid_y_1[:, :-1]
        if len(dx_1.shape) == 2: dx_1 = dx_1.unsqueeze(-1)
        if len(dy_1.shape) == 2: dy_1 = dy_1.unsqueeze(-1)
        Mesh_1 = {
            'grid': coord_tensor_1.cpu(),
            'grid_x': grid_x_1.cpu(),
            'grid_y': grid_y_1.cpu(),
            'dx': dx_1.cpu(),
            'dy': dy_1.cpu()
        }
        torch.save(Mesh_1, data_dir / 'Mesh_subdomain_1.pt')

        x_coords_2 = coords_2
        grid_x_2, grid_y_2 = torch.meshgrid(x_coords_2, y_coords, indexing='ij')
        coord_tensor_2 = torch.stack([grid_x_2, grid_y_2], dim=-1)
        dx_2 = grid_x_2[1:, :] - grid_x_2[:-1, :]
        dy_2 = grid_y_2[:, 1:] - grid_y_2[:, :-1]
        if len(dx_2.shape) == 2: dx_2 = dx_2.unsqueeze(-1)
        if len(dy_2.shape) == 2: dy_2 = dy_2.unsqueeze(-1)
        Mesh_2 = {
            'grid': coord_tensor_2.cpu(),
            'grid_x': grid_x_2.cpu(),
            'grid_y': grid_y_2.cpu(),
            'dx': dx_2.cpu(),
            'dy': dy_2.cpu()
        }
        torch.save(Mesh_2, data_dir / 'Mesh_subdomain_2.pt')

        self.Mesh_1 = Mesh_1
        self.Mesh_2 = Mesh_2
        # return coord_tensor, self.grid_x, self.grid_y, self.dx, self.dy


    def _create_uniform_grid_3D(self, data_dir):
        data_dir.mkdir(exist_ok=True)
        segments = self.grid_segments
        n1, n2, n3 = self.n1, self.n2, self.n3
        gc = self.ghostcell

        coords_1 = torch.linspace(segments[0][0], segments[0][1], n1+1, 
                                 device=self.device, dtype=self.dtype)
        coords_2 = torch.linspace(segments[1][0], segments[1][1], n2+1, 
                                 device=self.device, dtype=self.dtype)[1:]  
        coords_3 = torch.linspace(segments[2][0], segments[2][1], n3+1, 
                                 device=self.device, dtype=self.dtype)[1:]  
        coords = torch.cat([coords_1, coords_2, coords_3])

        if gc > 0:
            lower_spacing = coords[1] - coords[0]
            lower_ghost = torch.linspace(
                coords[0] - gc * lower_spacing, 
                coords[0] - lower_spacing, 
                gc,
                device=self.device, dtype=self.dtype)            
            upper_spacing = coords[-1] - coords[-2]
            upper_ghost = torch.linspace(
                coords[-1] + upper_spacing, 
                coords[-1] + gc * upper_spacing, 
                gc,
                device=self.device, dtype=self.dtype)
            coords = torch.cat([lower_ghost, coords, upper_ghost])

        x_coords = coords - 0.5
        y_coords = coords - 0.5
        z_coords = coords.clone()
        
        grid_x, grid_y, grid_z = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')    
        coord_tensor = torch.stack([grid_x, grid_y, grid_z], dim=-1)
        
        self.dx = grid_x[1:, :, :] - grid_x[:-1, :, :]
        self.dy = grid_y[:, 1:, :] - grid_y[:, :-1, :]
        self.dz = grid_z[:, :, 1:] - grid_z[:, :, :-1]    

        if len(self.dx.shape) == 3:
            self.dx = self.dx.unsqueeze(-1)
        if len(self.dy.shape) == 3:
            self.dy = self.dy.unsqueeze(-1)
        if len(self.dz.shape) == 3:
            self.dz = self.dz.unsqueeze(-1)

        grid = coord_tensor
        self.Nx, self.Ny, self.Nz = grid.shape[0], grid.shape[1], grid.shape[2]

        Mesh = {
        'grid': coord_tensor.cpu(),
        'grid_x': grid_x.cpu(),
        'grid_y': grid_y.cpu(),
        'grid_z': grid_z.cpu(), 
        'dx': self.dx.cpu(),
        'dy': self.dy.cpu(),
        'dz': self.dz.cpu()}
        torch.save(Mesh, data_dir/'Mesh.pt')
        
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_z = grid_z

        return coord_tensor, self.grid_x, self.grid_y, self.grid_z,self.dx, self.dy, self.dz
    

    def _setup_phy_ps(self, param):
        normal_param = param
        self.mu_1 = torch.tensor(normal_param['mu_1'], device=self.device, dtype=torch.float64)
        self.beta_1x = torch.tensor(normal_param['beta_1x'], device=self.device, dtype=torch.float64)
        self.beta_1y = torch.tensor(normal_param['beta_1y'], device=self.device, dtype=torch.float64)
        self.sigma_1 = torch.tensor(normal_param['sigma_1'], device=self.device, dtype=torch.float64)
        self.lambda_1 = torch.tensor(normal_param['lambda_1'], device=self.device, dtype=torch.float64)
        self.mu_2 = torch.tensor(normal_param['mu_2'], device=self.device, dtype=torch.float64)
        self.beta_2x = torch.tensor(normal_param['beta_2x'], device=self.device, dtype=torch.float64)
        self.beta_2y = torch.tensor(normal_param['beta_2y'], device=self.device, dtype=torch.float64)
        self.sigma_2 = torch.tensor(normal_param['sigma_2'], device=self.device, dtype=torch.float64)
        self.lambda_2 = torch.tensor(normal_param['lambda_2'], device=self.device, dtype=torch.float64)

    
    def _get_time_step(self, phy_dict):
        self._setup_phy_ps(phy_dict)
        min_dx = torch.min(self.dx)  
        min_dy = torch.min(self.dy)  
        # min_dz = torch.min(self.dz)
        dd_min = min(min_dx, min_dy)

        dt_diffusion_1 = dd_min**2/ (4*self.mu_1)
        dt_advection_1 = 1 / (self.beta_1x/min_dx + self.beta_1y/min_dy)
        dt_reaction_1 = 1 / (self.sigma_1 + 1e-20)

        dt_diffusion_2 = dd_min**2/ (4*self.mu_2)
        dt_advection_2 = 1 / (self.beta_2x/min_dx + self.beta_2y/min_dy)
        dt_reaction_2 = 1 / (self.sigma_2 + 1e-20)       

        dt_1 = self.cfl  / (1/dt_advection_1 + 1/dt_diffusion_1 + 1/dt_reaction_1)
        dt_2 = self.cfl  / (1/dt_advection_2 + 1/dt_diffusion_2 + 1/dt_reaction_2)

        dt = min(dt_1, dt_2)
        print(f'dtau per itera: {dt}')
        return dt.clone().detach().to(device=self.device, dtype=self.dtype)
    
        
    def _setup_coefficient_3D(self):
        # x direction
        dx_im1 = self.dx[:-1, 1:-1, 1:-1, :]  # dx_{i-1/2}
        dx_ip1 = self.dx[1:, 1:-1, 1:-1, :]   # dx_{i+1/2}        
        
        self.alpha_x = 2.0 / (dx_im1 * (dx_im1 + dx_ip1))    # Calculate weight coefficient 
        self.beta_x = -2.0 / (dx_im1 * dx_ip1)
        self.gamma_x = 2.0 / (dx_ip1 * (dx_im1 + dx_ip1))

        # y direction
        dy_jm1 = self.dy[1:-1, :-1, 1:-1, :]  # dy_{j-1/2}
        dy_jp1 = self.dy[1:-1, 1:, 1:-1, :]   # dy_{j+1/2}
        
        self.alpha_y = 2.0 / (dy_jm1 * (dy_jm1 + dy_jp1))
        self.beta_y = -2.0 / (dy_jm1 * dy_jp1)
        self.gamma_y = 2.0 / (dy_jp1 * (dy_jm1 + dy_jp1))

        # z direction
        dz_km1 = self.dz[1:-1, 1:-1, :-1, :]  # dz_{k-1/2}
        dz_kp1 = self.dz[1:-1, 1:-1, 1:, :]   # dz_{k+1/2}
        
        self.alpha_z = 2.0 / (dz_km1 * (dz_km1 + dz_kp1))
        self.beta_z = -2.0 / (dz_km1 * dz_kp1)
        self.gamma_z = 2.0 / (dz_kp1 * (dz_km1 + dz_kp1))   
    
    def _setup_coefficient_2D(self):
        # # x direction
        # dx_im1 = self.dx[:-1, 1:-1,  :]  # dx_{i-1/2}
        # dx_ip1 = self.dx[1:, 1:-1, :]   # dx_{i+1/2}        
        
        # self.alpha_x = 2.0 / (dx_im1 * (dx_im1 + dx_ip1))    # Calculate weight coefficient 
        # self.beta_x = -2.0 / (dx_im1 * dx_ip1)
        # self.gamma_x = 2.0 / (dx_ip1 * (dx_im1 + dx_ip1))

        # # y direction
        # dy_jm1 = self.dy[1:-1, :-1, :]  # dy_{j-1/2}
        # dy_jp1 = self.dy[1:-1, 1:, :]   # dy_{j+1/2}
        
        # self.alpha_y = 2.0 / (dy_jm1 * (dy_jm1 + dy_jp1))
        # self.beta_y = -2.0 / (dy_jm1 * dy_jp1)
        # self.gamma_y = 2.0 / (dy_jp1 * (dy_jm1 + dy_jp1))

        self.dx = self.grid_x[1:, :] - self.grid_x[:-1, :]
        self.dy = self.grid_y[:, 1:] - self.grid_y[:, :-1]
        
        dx_im1 = self.dx[:-1, :]
        dx_ip1 = self.dx[ 1:, :]   
        dy_jm1 = self.dy[:, :-1]  
        dy_jp1 = self.dy[:, 1: ]   

        alpha_x = - dx_ip1 / (dx_im1 * (dx_im1 + dx_ip1))    
        beta_x = (dx_ip1 - dx_im1) / (dx_im1 * dx_ip1)
        gamma_x = dx_im1 / (dx_ip1 * (dx_im1 + dx_ip1))
        
        alpha_y = -dy_jp1 / (dy_jm1 * (dy_jm1 + dy_jp1))
        beta_y = (dy_jp1 - dy_jm1) / (dy_jm1 * dy_jp1)
        gamma_y = dy_jm1 / (dy_jp1 * (dy_jm1 + dy_jp1))
        self.NX = self.grid_x.shape[0]
        self.NY = self.grid_y.shape[1]
        self.d1x_coeff = torch.zeros((self.NX, self.NY, 3), dtype=torch.float64, device=self.device)
        self.d1x_coeff[1:-1, :, :] = torch.stack([alpha_x, beta_x, gamma_x], dim=2)
        self.d1x_coeff[0, :, 1] = 1 / self.dx[0, :] 
        self.d1x_coeff[-1, :, 1] = 1 / self.dx[-1, :]

        self.d1y_coeff = torch.zeros((self.NX, self.NY, 3), dtype=torch.float64, device=self.device)
        self.d1y_coeff[:, 1:-1, :] = torch.stack([alpha_y, beta_y, gamma_y], dim=2)
        self.d1y_coeff[:, 0, 1] = 1 / self.dy[:, 0]
        self.d1y_coeff[:, -1, 1] = 1 / self.dy[:, -1]      

        alpha_x = 2.0 / (dx_im1 * (dx_im1 + dx_ip1))    
        beta_x = -2.0 / (dx_im1 * dx_ip1)
        gamma_x = 2.0 / (dx_ip1 * (dx_im1 + dx_ip1))
        
        alpha_y = 2.0 / (dy_jm1 * (dy_jm1 + dy_jp1))
        beta_y = -2.0 / (dy_jm1 * dy_jp1)
        gamma_y = 2.0 / (dy_jp1 * (dy_jm1 + dy_jp1))

        self.d2x_coeff = torch.zeros((self.NX, self.NY, 3), dtype=torch.float64, device=self.device)
        self.d2x_coeff[1:-1, :, :] = torch.stack([alpha_x, beta_x, gamma_x], dim=2)
        self.d2y_coeff = torch.zeros((self.NX, self.NY, 3), dtype=torch.float64, device=self.device)
        self.d2y_coeff[:, 1:-1, :] = torch.stack([alpha_y, beta_y, gamma_y], dim=2)



    def plot_mesh(self, mesh, ax, title="", point_size=8, line_width=0.8, color='k'):
        grid_x = mesh['grid_x']
        grid_y = mesh['grid_y']

        if isinstance(grid_x, torch.Tensor):
            grid_x = grid_x.numpy()
        if isinstance(grid_y, torch.Tensor):
            grid_y = grid_y.numpy()

        Nx, Ny = grid_x.shape

        # 画纵向网格线（固定 j，沿 i 方向）
        for j in range(Ny):
            ax.plot(grid_x[:, j], grid_y[:, j], color=color, lw=line_width)

        # 画横向网格线（固定 i，沿 j 方向）
        for i in range(Nx):
            ax.plot(grid_x[i, :], grid_y[i, :], color=color, lw=line_width)

        # 画网格点
        ax.scatter(grid_x, grid_y, s=point_size, color='r')

        ax.set_title(title)
        ax.set_aspect('equal')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(False)

        return ax




    def check_spacing(self, mesh, name="Mesh"):
        dx = mesh['dx']
        dy = mesh['dy']

        if isinstance(dx, torch.Tensor):
            dx = dx.squeeze(-1).numpy()
        if isinstance(dy, torch.Tensor):
            dy = dy.squeeze(-1).numpy()

        print(f"----- {name} -----")
        print(f"dx shape = {dx.shape}, dy shape = {dy.shape}")
        print(f"dx min/max = {dx.min():.6f} / {dx.max():.6f}")
        print(f"dy min/max = {dy.min():.6f} / {dy.max():.6f}")
        print(f"dx uniform? {abs(dx.max() - dx.min()) < 1e-12}")
        print(f"dy uniform? {abs(dy.max() - dy.min()) < 1e-12}")
        print()




    def __repr__(self):
        return f"""Numerical Config(
                    grid=(n1:{self.n1}, n2:{self.n2}, n3:{self.n3}, N:{self.N}), segments={self.grid_segments},
                    T_final={self.T_final}, ghostcell={self.ghostcell}
                )"""


