# Code is copied with necessary refactoring from https://github.com/arkadaw9/r3_sampling_icml2023

import torch
import numpy as np

def neural_net(dnn, x, t):
        u = dnn(torch.cat([x, t], dim=1))
        return u

def residual_net(dnn, x, t, pde_nu, pde_beta):
    """ Autograd for calculating the residual for different systems."""
    u = neural_net(dnn, x=x, t=t)

    u_t = torch.autograd.grad(
        u, t,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True
    )[0]
    u_x = torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True
    )[0]

    u_xx = torch.autograd.grad(
        u_x, x,
        grad_outputs=torch.ones_like(u_x),
        retain_graph=True,
        create_graph=True
        )[0]

    return u_t - pde_nu*u_xx + pde_beta*u_x

def loss_res_pinn(dnn, x_f, t_f, pde_nu, pde_beta):
    r_pred = residual_net(dnn, x_f, t_f, pde_nu, pde_beta)
    loss_r = torch.mean(r_pred**2)
    return loss_r

def loss_res(dnn, x_f, t_f, pde_nu, pde_beta):
    return loss_res_pinn(dnn, x_f, t_f, pde_nu, pde_beta)


def net_b_derivatives(u_lb, u_ub, x_bc_lb, x_bc_ub):
    """For taking BC derivatives."""

    u_lb_x = torch.autograd.grad(
        u_lb, x_bc_lb,
        grad_outputs=torch.ones_like(u_lb),
        retain_graph=True,
        create_graph=True
        )[0]

    u_ub_x = torch.autograd.grad(
        u_ub, x_bc_ub,
        grad_outputs=torch.ones_like(u_ub),
        retain_graph=True,
        create_graph=True
        )[0]
    return u_lb_x, u_ub_x
    
def loss_bcs(dnn, X_lb, X_ub, pde_nu):
    t_lb, x_lb = X_lb[:,0:1], X_lb[:,1:2]
    t_ub, x_ub = X_ub[:,0:1], X_ub[:,1:2]
    u_pred_lb = neural_net(dnn, t=t_lb, x=x_lb)
    u_pred_ub = neural_net(dnn, t=t_ub, x=x_ub)
    loss_bc = torch.mean((u_pred_lb - u_pred_ub) ** 2)
    if pde_nu != 0:
        u_pred_lb_x, u_pred_ub_x = net_b_derivatives(u_pred_lb, u_pred_ub, x_lb, x_ub)
        loss_bc += torch.mean((u_pred_lb_x - u_pred_ub_x) ** 2)
    return loss_bc
    
def loss_ics(dnn, X_ic, Y_ic):
    # Evaluate the network over IC
    t = X_ic[:,0:1]
    x = X_ic[:,1:2]
    u_pred = neural_net(dnn, t=t, x=x)
    # Compute the initial loss
    loss_ics = torch.mean((Y_ic.flatten() - u_pred.flatten())**2)
    return loss_ics
    

class UniformSampler(torch.nn.Module):
    def __init__(self, x_lim: tuple, t_lim: tuple, N=1000, device=None):
        super(UniformSampler, self).__init__()
        
        self.N = N
        self.x_lim = x_lim
        self.t_lim = t_lim
        self.device = device if device is not None else torch.device('cpu')
        self.update()
    
    def update(self):
        self.x = torch.zeros(self.N, 1, dtype=torch.float32, device=self.device).uniform_(*self.x_lim)
        self.t = torch.zeros(self.N, 1, dtype=torch.float32, device=self.device).uniform_(*self.t_lim)
        return self.x, self.t

class DatasetGenerator(torch.utils.data.Dataset):
    def __init__(self, x_star, t_star, usol, N=10):
        super(DatasetGenerator, self).__init__()
        self.x_star = x_star
        self.t_star = t_star
        self.usol = usol
        self.N = N
        self.x_lim = (x_star.min(), x_star.max())
    
    def __get_item__(self, idx):
        t_idx = len(self.t_star)//self.N
        if idx == self.N:
            usol = np.copy(self.usol[:,(idx-1)*t_idx:])
            t_star_ = np.copy(self.t_star[(idx-1)*t_idx:])
            t_star_ -= t_star_[0] # timeshift
        else:
            usol = np.copy(self.usol[: , (idx-1)*t_idx : idx*t_idx+1])
            t_star_ = np.copy(self.t_star[(idx-1)*t_idx : idx*t_idx+1])
            t_star_ -= t_star_[0] # timeshift
        
        t0, t1 = 0, (t_star_[-1,0]-t_star_[0,0]) # time limit (shifted)
        state0 = np.copy(usol[:, 0:1]) # first point of the cropeed usol
        t_lim = (t0, t1)
        return state0, t_star_, usol, t_lim, self.x_lim

def function(u0: str):
    """Initial condition, string --> function."""

    if u0 == 'sin(x)':
        u0 = lambda x: np.sin(x)
    elif u0 == 'sin(pix)':
        u0 = lambda x: np.sin(np.pi*x)
    elif u0 == 'sin^2(x)':
        u0 = lambda x: np.sin(x)**2
    elif u0 == 'sin(x)cos(x)':
        u0 = lambda x: np.sin(x)*np.cos(x)
    elif u0 == '0.1sin(x)':
        u0 = lambda x: 0.1*np.sin(x)
    elif u0 == '0.5sin(x)':
        u0 = lambda x: 0.5*np.sin(x)
    elif u0 == '10sin(x)':
        u0 = lambda x: 10*np.sin(x)
    elif u0 == '50sin(x)':
        u0 = lambda x: 50*np.sin(x)
    elif u0 == '1+sin(x)':
        u0 = lambda x: 1 + np.sin(x)
    elif u0 == '2+sin(x)':
        u0 = lambda x: 2 + np.sin(x)
    elif u0 == '6+sin(x)':
        u0 = lambda x: 6 + np.sin(x)
    elif u0 == '10+sin(x)':
        u0 = lambda x: 10 + np.sin(x)
    elif u0 == 'sin(2x)':
        u0 = lambda x: np.sin(2*x)
    elif u0 == 'tanh(x)':
        u0 = lambda x: np.tanh(x)
    elif u0 == '2x':
        u0 = lambda x: 2*x
    elif u0 == 'x^2':
        u0 = lambda x: x**2
    elif u0 == 'gauss':
        x0 = np.pi
        sigma = np.pi/4
        u0 = lambda x: np.exp(-np.power((x - x0)/sigma, 2.)/2.)
    return u0

def convection_diffusion(u0: str, nu, beta, source=0, xgrid=256, nt=100):
    """Calculate the u solution for convection/diffusion, assuming PBCs.
    Args:
        u0: Initial condition
        nu: viscosity coefficient
        beta: wavespeed coefficient
        source: q (forcing term), option to have this be a constant
        xgrid: size of the x grid
    Returns:
        u_vals: solution
    """

    N = xgrid
    h = 2 * np.pi / N
    x = np.arange(0, 2*np.pi, h) # not inclusive of the last point
    t = np.linspace(0, 1, nt).reshape(-1, 1)
    X, T = np.meshgrid(x, t)

    # call u0 this way so array is (n, ), so each row of u should also be (n, )
    u0 = function(u0)
    u0 = u0(x)

    G = (np.copy(u0)*0)+source # G is the same size as u0

    IKX_pos =1j * np.arange(0, N/2+1, 1)
    IKX_neg = 1j * np.arange(-N/2+1, 0, 1)
    IKX = np.concatenate((IKX_pos, IKX_neg))
    IKX2 = IKX * IKX

    uhat0 = np.fft.fft(u0)
    nu_factor = np.exp(nu * IKX2 * T - beta * IKX * T)
    A = uhat0 - np.fft.fft(G)*0 # at t=0, second term goes away
    uhat = A*nu_factor + np.fft.fft(G)*T # for constant, fft(p) dt = fft(p)*T
    u = np.real(np.fft.ifft(uhat))

    u_vals = u.flatten()
    return u_vals

class Loss:
    def __init__(self, device, xgrid, nt, pde_beta, u0_str, N_f):
        x_star = np.linspace(0, 2*np.pi, xgrid, endpoint=False).reshape(-1, 1) # not inclusive
        t_star = np.linspace(0, 1, nt).reshape(-1, 1)
        x_lim = (0, 2*np.pi)
        t_lim = (0, 1)

        pde_nu=0.0
        usol = convection_diffusion(u0_str, pde_nu, pde_beta, xgrid= xgrid, nt=nt)
        usol = usol.reshape(-1, 1) # Exact solution reshaped into (n, 1)
        usol = usol.reshape(len(t_star), len(x_star)) # Exact on the (x,t) grid
        usol = usol.T
        dataset = DatasetGenerator(x_star, t_star, usol, 1)
        state0, t_star, usol, t_lim, x_lim = dataset.__get_item__(1)

        t_ic = torch.zeros((x_star.shape[0], 1), device=device).float()
        x_ic = torch.tensor(x_star.reshape(-1, 1), device=device).float()
        self.X_ic = torch.cat([t_ic, x_ic], dim=1)
        t_lb = torch.tensor(t_star.reshape(-1,1), device=device).float()
        x_lb = torch.ones_like(t_lb, device=device).float() * 0. ## x_lim is zero
        self.X_lb = torch.cat([t_lb, x_lb], dim=1)
        self.X_lb.requires_grad = True
        t_ub = torch.tensor(t_star.reshape(-1,1), device=device).float()
        x_ub = torch.ones_like(t_ub, device=device).float() * (2*np.pi) ## x_lim is zero
        self.X_ub = torch.cat([t_ub, x_ub], dim=1)
        self.X_ub.requires_grad = True
        self.Y_ic = torch.tensor(state0, device=device).float()
        sampler = UniformSampler(x_lim=x_lim,
                                t_lim=t_lim, 
                                N=N_f, 
                                device=device, )
        self.x_f = sampler.x
        self.t_f = sampler.t
        self.x_f.requires_grad = True
        self.t_f.requires_grad = True

        self.pde_nu = pde_nu
        self.pde_beta = pde_beta

        self.device=device
        self.usol = usol
        self.x_star = x_star
        self.t_star = t_star

    def predict(self, dnn, t, x):
        dnn.eval()
        x = torch.tensor(x, requires_grad=True).float().to(
            self.device).unsqueeze(1)
        t = torch.tensor(t, requires_grad=True).float().to(
            self.device).unsqueeze(1)

        u = neural_net(dnn, t=t, x=x)
        u = u.detach().cpu().numpy()
        return u
    
    def evaluate_model(self, dnn):
        # Get trained network parameters
        X, T = np.meshgrid(self.x_star, self.t_star) # all the X grid points T times, all the T grid points X times
        n_x = self.x_star.shape[0]
        n_t = self.t_star.shape[0]

        u_pred = self.predict(dnn, T.flatten(), X.flatten())
        u_pred = u_pred.reshape(n_t, n_x)
        u_pred = u_pred.T

        error = np.linalg.norm(u_pred - self.usol) / np.linalg.norm(self.usol) 
        return torch.tensor(error).to(self.device)
    
    
    def get_loss(self, dnn, loss_type, lamda_ic=1.0, lamba_f=1.0, lamda_bc=1.0):


        L_0 = loss_ics(dnn, self.X_ic, self.Y_ic)
        L_bc = loss_bcs(dnn, self.X_lb, self.X_ub, self.pde_nu)
        L_f = loss_res(dnn, self.x_f, self.t_f, self.pde_nu, self.pde_beta)
        
        # Compute loss
        loss = lamba_f * L_f + lamda_ic * L_0 + lamda_bc * L_bc

        test_loss = self.evaluate_model(dnn)
        
        # if method == "pinn_uniform":
        #     with torch.no_grad():
        #         self.x_f, self.t_f = self.sampler.update()
        #         self.x_f.requires_grad = True
        #         self.t_f.requires_grad = True
                    
        loss_dic = {
            'total': loss,
            'ic': L_0,
            'bc': L_bc,
            'residual': L_f,
            'test_mse': test_loss
        }
        return loss_dic[loss_type]