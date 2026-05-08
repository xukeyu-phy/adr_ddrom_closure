import torch
import time
from config import Config
from pathlib import Path

class ADRSolver:
    def __init__(self, device, dtype, phy_ps, data_dir):
        self.device = device
        config = Config(device, dtype)
        self._init_config(config, phy_ps, data_dir)
        torch.set_default_dtype(dtype)


        current_dir = Path(__file__).parent
        self.fig_dir = current_dir / "Fig"
        



    def _main_line(self):
        start_time = time.time()
        u_init = torch.zeros((self.config.Nx, self.config.Ny), device=self.device)
        u_init = self._setup_initial_condition(u_init)
        u_init = self._setup_boundary_conditions(u_init, self.config.ghostcell, self.config.bc_type, self.config.bc_value)

        u_sub_1 = u_init[0:self.nx_sub1, :].clone()
        u_sub_2 = u_init[self.nx_sub1-1:self.Nx, :].clone()
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # im = ax.imshow(u_init.T.cpu().numpy(), origin='lower',aspect='auto', 
        #                extent=[self.config.grid_x.cpu().numpy().min(), self.config.grid_x.cpu().numpy().max(), 
        #                self.config.grid_y.cpu().numpy().min(), self.config.grid_y.cpu().numpy().max()],)
        
        # plt.colorbar(im, label='U0')
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # plt.savefig( self.fig_dir / "u_initialBC_neumann.pdf", dpi=300, bbox_inches="tight")
        # plt.show()
        # plt.close()

        t = 0.0
        t_iter = 0
        u_n = u_init.clone()        

        while t < self.config.T_final:
            if (t + self.dt) >= self.config.T_final:
                self.dt = self.config.T_final - t
            t += self.dt
            t_iter += 1                

            schwartz_tol = 1.0
            if schwartz_tol > self.config.convergence_tol:
                
                # ============== Subdomain ==============
                u_sub_1_prev = u_sub_1.clone()
                u_sub_1 = self.runge_kutta_1_step(u_sub_1_prev, self.dt, self.beta_1x, self.beta_1y, 
                                                  self.d1x_W_sub_1, self.d1x_C_sub_1, self.d1x_E_sub_1, 
                                                  self.d1y_S_sub_1, self.d1y_C_sub_1, self.d1y_N_sub_1, 
                                                  self.mu_1, self.d1x_W_sub_1, self.d1x_C_sub_1, self.d1x_E_sub_1, 
                                                  self.d1y_S_sub_1, self.d1y_C_sub_1, self.d1y_N_sub_1, 
                                                  self.sigma_1)
                u_sub_1 = self._setup_boundary_conditions(u_sub_1, self.config.ghostcell, self.config.bc_type, self.config.bc_value)
                du_sub_1 = torch.abs(u_sub_1 - u_sub_1_prev)
                l2_du_sub_1 = torch.norm(du_sub_1, p=2)

                u_sub_2_prev = u_sub_2.clone()
                u_sub_2 = self.runge_kutta_1_step(u_sub_2_prev, self.dt, self.beta_2x, self.beta_2y, 
                                                  self.d1x_W_sub_2, self.d1x_C_sub_2, self.d1x_E_sub_2, 
                                                  self.d1y_S_sub_2, self.d1y_C_sub_2, self.d1y_N_sub_2, 
                                                  self.mu_2, self.d1x_W_sub_2, self.d1x_C_sub_2, self.d1x_E_sub_2, 
                                                  self.d1y_S_sub_2, self.d1y_C_sub_2, self.d1y_N_sub_2, 
                                                  self.sigma_2)
                u_sub_2 = self._setup_boundary_conditions(u_sub_2, self.config.ghostcell, self.config.bc_type, self.config.bc_value)
                du_sub_2 = torch.abs(u_sub_2 - u_sub_2_prev)
                l2_du_sub_2 = torch.norm(du_sub_2, p=2)


                # ============== Interface ==============

            
        end_time = time.time()
        runtime =  end_time - start_time
        print(f"Runtime: {runtime:.3f}s")  
        print(f"iteration: {t_iter}")  
        return u_n

    def _interface_exchange(self, u_sub_1, u_sub_2):
        pass


    def rhs(self, u, beta_x, beta_y, d1x_W, d1x_C, d1x_E, d1y_S, d1y_C, d1y_N, mu, d2x_W, d2x_C, d2x_E, d2y_S, d2y_C, d2y_N, sigma):
        u = u.to(self.device)
        
        diffusion = self.rhs_diffusion(u, mu, d2x_W, d2x_C, d2x_E, d2y_S, d2y_C, d2y_N)
        convection = self.rhs_convection(u, beta_x, beta_y, d1x_W, d1x_C, d1x_E, d1y_S, d1y_C, d1y_N)
        source = self.rhs_source(u, sigma)
        
        spatial_rhs = torch.zeros_like(u, device=self.device)
        interior = slice(1, -1)
        
        spatial_rhs[interior, interior, interior, :] = (
            diffusion[interior, interior, interior, :]
            - convection[interior, interior, interior, :] 
            - source[interior, interior, interior, :]
            )        
        return spatial_rhs

    
    def _setup_boundary_conditions(self, u, gc, bc_type, bc_value):
        if bc_type == 'dirichlet':
            u[0, :, :, :] = bc_value
            u[-1, :, :, :] = bc_value
            u[:, 0, :, :] = bc_value
            u[:, -1, :, :] = bc_value
            u[:, :, 0, :] = bc_value
            u[:, :, -1, :] = bc_value
            
        elif bc_type == 'periodic':
            u[:gc, :, :, :] = u[-2*gc:-gc, :, :, :]  
            u[-gc:, :, :, :] = u[gc:2*gc, :, :, :]  

            u[:, :gc, :, :] = u[:, -2*gc:-gc, :, :] 
            u[:, -gc:, :, :] = u[:, gc:2*gc, :, :]
            
            u[:, :, :gc, :] = u[:, :, -2*gc:-gc, :]
            u[:, :, -gc:, :] = u[:, :, gc:2*gc, :]

        elif bc_type == 'neumann':
            u[0, :] = u[1, :]
            u[:, 0] = u[:, 1]
            u[-1, :] = u[-2, :]
            u[:, -1] = u[:, -2]
        else:
            raise ValueError(f"Unsupported boundary condition type: {bc_type}")
        
        return u
    

    def _setup_initial_condition(self, u):
        if self.config.initial_condition_type == 'uniform':
            u[:, :, :] = self.config.initial_value
        elif self.config.initial_condition_type == 'analytical':
            u = torch.cos(2 * torch.pi * self.config.grid_x) * torch.cos(2 * torch.pi * self.config.grid_y) 

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        im = ax.imshow(u.T.cpu().numpy(), origin='lower',aspect='auto', 
                       extent=[self.config.grid_x.cpu().numpy().min(), self.config.grid_x.cpu().numpy().max(), 
                       self.config.grid_y.cpu().numpy().min(), self.config.grid_y.cpu().numpy().max()],)
        
        plt.colorbar(im, label='U0')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.savefig( self.fig_dir / "u_initialBC.pdf", dpi=300, bbox_inches="tight")
        # plt.show()
        plt.close()

        return u
    
    def rhs_source(self, u, sigma):
        return sigma * u
    
    def rhs_convection(self, u, beta_x, beta_y, d1x_W, d1x_C, d1x_E, d1y_S, d1y_C, d1y_N):
        u_c = u[1:-1, 1:-1, :]
        u_w = u[:-2, 1:-1, :]
        u_e = u[2:, 1:-1, :]
        u_n = u[1:-1, 2:, :]
        u_s = u[1:-1, :-2, :]

        du_dx = d1x_W * u_w + d1x_C * u_c + d1x_E * u_e
        du_dy = d1y_S * u_s + d1y_C * u_c + d1y_N * u_n
        
        convection = torch.zeros_like(u, device=self.device)
        convection[1:-1, 1:-1, :] = beta_x * du_dx + beta_y * du_dy
        return convection
    
    def rhs_diffusion(self, u, mu, d2x_W, d2x_C, d2x_E, d2y_S, d2y_C, d2y_N):
        u_c = u[1:-1, 1:-1, :]
        u_w = u[:-2, 1:-1, :]
        u_e = u[2:, 1:-1, :]
        u_n = u[1:-1, 2:, :]
        u_s = u[1:-1, :-2, :]
    
        d2u_dx2 = d2x_W * u_w + d2x_C * u_c + d2x_E * u_e
        d2u_dy2 = d2y_S * u_s + d2y_C * u_c + d2y_N * u_n

        nabla2 = u.clone()
        nabla2[1:-1, 1:-1, :] = mu * (d2u_dx2 + d2u_dy2)
        return mu * nabla2
    
    
    def _init_config(self, config, phy_ps, data_dir):
        config._create_uniform_grid_2D(data_dir)
        config._setup_phy_ps(phy_ps)
        config._setup_coefficient_2D()
        self.dt = config._get_time_step(phy_ps)
        
        self.Nx = config.grid_x.shape[0]
        self.Ny = config.grid_y.shape[1]

        self.nx_sub1 = config.Mesh_1['grid_x'].shape[0]
        self.ny_sub1 = config.Mesh_1['grid_y'].shape[1]
        self.nx_sub2 = config.Mesh_2['grid_x'].shape[0]
        self.ny_sub2 = config.Mesh_2['grid_y'].shape[1]

        self.d1x_W = config.d1x_coeff[1:-1, 1:-1, 0]
        self.d1x_C = config.d1x_coeff[1:-1, 1:-1, 1]
        self.d1x_E = config.d1x_coeff[1:-1, 1:-1, 2]
        self.d1y_S = config.d1y_coeff[1:-1, 1:-1, 0]
        self.d1y_C = config.d1y_coeff[1:-1, 1:-1, 1]
        self.d1y_N = config.d1y_coeff[1:-1, 1:-1, 2]

        self.d2x_W = config.d2x_coeff[1:-1, 1:-1, 0]
        self.d2x_C = config.d2x_coeff[1:-1, 1:-1, 1]
        self.d2x_E = config.d2x_coeff[1:-1, 1:-1, 2]
        self.d2y_S = config.d2y_coeff[1:-1, 1:-1, 0]
        self.d2y_C = config.d2y_coeff[1:-1, 1:-1, 1]
        self.d2y_N = config.d2y_coeff[1:-1, 1:-1, 2]
    
        self.dx_center = 0.5 * (config.dx[1:, 1:-1] + config.dx[:-1, 1:-1])
        self.dy_center = 0.5 * (config.dy[1:-1, 1:] + config.dy[1:-1, :-1])

        self.d1x_W_sub_1 = self.d1x_W[0:self.nx_sub1-2, :].clone()
        self.d1x_C_sub_1 = self.d1x_C[0:self.nx_sub1-2, :].clone()
        self.d1x_E_sub_1 = self.d1x_E[0:self.nx_sub1-2, :].clone()
        self.d1y_S_sub_1 = self.d1y_S[0:self.nx_sub1-2, :].clone()
        self.d1y_N_sub_1 = self.d1y_N[0:self.nx_sub1-2, :].clone()
        self.d1y_C_sub_1 = self.d1y_C[0:self.nx_sub1-2, :].clone()        

        self.d1x_W_sub_2 = self.d1x_W[self.nx_sub1-1:, :].clone()
        self.d1x_C_sub_2 = self.d1x_C[self.nx_sub1-1:, :].clone()
        self.d1x_E_sub_2 = self.d1x_E[self.nx_sub1-1:, :].clone()
        self.d1y_S_sub_2 = self.d1y_S[self.nx_sub1-1:, :].clone()
        self.d1y_N_sub_2 = self.d1y_N[self.nx_sub1-1:, :].clone()
        self.d1y_C_sub_2 = self.d1y_C[self.nx_sub1-1:, :].clone()   

        self.d2x_W_sub_1 = self.d2x_W[0:self.nx_sub1-2, :].clone()
        self.d2x_C_sub_1 = self.d2x_C[0:self.nx_sub1-2, :].clone()
        self.d2x_E_sub_1 = self.d2x_E[0:self.nx_sub1-2, :].clone()
        self.d2y_S_sub_1 = self.d2y_S[0:self.nx_sub1-2, :].clone()
        self.d2y_N_sub_1 = self.d2y_N[0:self.nx_sub1-2, :].clone()
        self.d2y_C_sub_1 = self.d2y_C[0:self.nx_sub1-2, :].clone()        

        self.d2x_W_sub_2 = self.d2x_W[self.nx_sub1-1:, :].clone()
        self.d2x_C_sub_2 = self.d2x_C[self.nx_sub1-1:, :].clone()
        self.d2x_E_sub_2 = self.d2x_E[self.nx_sub1-1:, :].clone()
        self.d2y_S_sub_2 = self.d2y_S[self.nx_sub1-1:, :].clone()
        self.d2y_N_sub_2 = self.d2y_N[self.nx_sub1-1:, :].clone()
        self.d2y_C_sub_2 = self.d2y_C[self.nx_sub1-1:, :].clone() 

        self.mu_1 = config.mu_1
        self.beta_1x = config.beta_1x
        self.beta_1y = config.beta_1y
        self.sigma_1 = config.sigma_1
        self.lambda_1 = config.lambda_1
        self.mu_2 = config.mu_2
        self.beta_2x = config.beta_2x
        self.beta_2y = config.beta_2y
        self.sigma_2 = config.sigma_2
        self.lambda_2 = config.lambda_2


        # self.d1x_W_sub_2 = self.d1x_W[self.nx_sub1-1:, :].clone()
        print(f"1")

    def runge_kutta_2_step(self, u_n, dt, ghostcell=None, bc_type=None):

        k1 = self.rhs(u_n)
        u_1 = u_n + 1.0 * dt * k1
        
        k2 = self.rhs(u_1)        
        u_new = u_n + (dt/2.0) * (k1 + k2)
        
        return u_new

    def runge_kutta_1_step(self, u_n, dt, beta_x, beta_y, d1x_W, d1x_C, d1x_E, d1y_S, d1y_C, d1y_N, mu, d2x_W, d2x_C, d2x_E, d2y_S, d2y_C, d2y_N, sigma):

        k1 = self.rhs(u_n, beta_x, beta_y, d1x_W, d1x_C, d1x_E, d1y_S, d1y_C, d1y_N, mu, d2x_W, d2x_C, d2x_E, d2y_S, d2y_C, d2y_N, sigma)

        u_1 = u_n + 1.0 * dt * k1
        
        # k2 = self.rhs(u_1)        
        # u_new = u_n + (dt/2.0) * (k1 + k2)
        
        return u_1