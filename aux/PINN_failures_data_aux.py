# Code is copied with necessary refactoring from https://github.com/a1k12/characterizing-pinns-failure-modes

import numpy as np
import random
import torch
from collections import OrderedDict

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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def sample_random(X_all, N):
    """Given an array of (x,t) points, sample N points from this."""
    set_seed(0) # this can be fixed for all N_f

    idx = np.random.choice(X_all.shape[0], N, replace=False)
    X_sampled = X_all[idx, :]

    return X_sampled

N_f=100
nt=100
xgrid=256
source=0
u0_str='sin(x)'
nu = 0.0
rho = 0.0
net='DNN'
activation='tanh'
layers=[50,50,50,50,1]
loss_style='mean'
system='convection'

x = np.linspace(0, 2*np.pi, xgrid, endpoint=False).reshape(-1, 1) # not inclusive
t = np.linspace(0, 1, nt).reshape(-1, 1)

# remove initial and boundaty data from X_star
t_noinitial = t[1:]
# remove boundary at x=0
x_noboundary = x[1:]
X_noboundary, T_noinitial = np.meshgrid(x_noboundary, t_noinitial)
X_star_noinitial_noboundary = np.hstack((X_noboundary.flatten()[:, None], T_noinitial.flatten()[:, None]))
X_f_train = sample_random(X_star_noinitial_noboundary, N_f)

G = np.full(X_f_train.shape[0], float(source))

X, T = np.meshgrid(x, t) # all the X grid points T times, all the T grid points X times

bc_lb = np.hstack((X[:,0:1], T[:,0:1])) # boundary condition at x = 0, and t = [0, 1]

# generate the other BC, now at x=2pi
t = np.linspace(0, 1, nt).reshape(-1, 1)
x_bc_ub = np.array([2*np.pi]*t.shape[0]).reshape(-1, 1)
bc_ub = np.hstack((x_bc_ub, t))

xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T)) # initial condition, from x = [-end, +end] and t=0
X_u_train = xx1 # (x,t) for initial condition

layers.insert(0, X_u_train.shape[-1])

    
class DNN(torch.nn.Module):
    def __init__(self, layers, activation, use_batch_norm=False, use_instance_norm=False):
        super(DNN, self).__init__()

        # parameters
        self.depth = len(layers) - 1

        if activation == 'identity':
            self.activation = torch.nn.Identity
        elif activation == 'tanh':
            self.activation = torch.nn.Tanh
        elif activation == 'relu':
            self.activation = torch.nn.ReLU
        elif activation == 'gelu':
            self.activation = torch.nn.GELU
        elif activation == 'sin':
            self.activation = Sine
        self.use_batch_norm = use_batch_norm
        self.use_instance_norm = use_instance_norm

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )

            if self.use_batch_norm:
                layer_list.append(('batchnorm_%d' % i, torch.nn.BatchNorm1d(num_features=layers[i+1])))
            if self.use_instance_norm:
                layer_list.append(('instancenorm_%d' % i, torch.nn.InstanceNorm1d(num_features=layers[i+1])))

            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out

class PhysicsInformedNN_pbc():
    """PINNs (convection/diffusion/reaction) for periodic boundary conditions."""
    def __init__(self, beta, L=1, device='cpu'):

        self.system = system

        self.epoch=0

        self.x_u = torch.tensor(X_u_train[:, 0:1], requires_grad=True).float().to(device)
        self.t_u = torch.tensor(X_u_train[:, 1:2], requires_grad=True).float().to(device)
        self.x_f = torch.tensor(X_f_train[:, 0:1], requires_grad=True).float().to(device)
        self.t_f = torch.tensor(X_f_train[:, 1:2], requires_grad=True).float().to(device)
        self.x_bc_lb = torch.tensor(bc_lb[:, 0:1], requires_grad=True).float().to(device)
        self.t_bc_lb = torch.tensor(bc_lb[:, 1:2], requires_grad=True).float().to(device)
        self.x_bc_ub = torch.tensor(bc_ub[:, 0:1], requires_grad=True).float().to(device)
        self.t_bc_ub = torch.tensor(bc_ub[:, 1:2], requires_grad=True).float().to(device)
        self.net = net

        if self.net == 'DNN':
            self.dnn = DNN(layers, activation).to(device)
        else: # "get_errorsetrained" can be included in model path
            # the dnn is within the PINNs class
            self.dnn = torch.load(net).dnn

        self.layers = layers
        self.nu = nu
        self.beta = beta
        self.rho = rho

        self.G = torch.tensor(G, requires_grad=True).float().to(device)
        self.G = self.G.reshape(-1, 1)

        self.L = L

        self.loss_style = loss_style
        
        u_vals = convection_diffusion(u0_str, nu, self.beta, source, xgrid, nt)
        u_star = u_vals.reshape(-1, 1) # Exact solution reshaped into (n, 1)
        Exact = u_star.reshape(len(t), len(x)) # Exact on the (x,t) grid
        uu1 = Exact[0:1,:].T # u(x, t) at t=0
        u_train = uu1 # just the initial condition
        
        self.u = torch.tensor(u_train, requires_grad=True).float().to(device)
        
        

    def net_u(self, x, t):
        """The standard DNN that takes (x,t) --> u."""
        u = self.dnn(torch.cat([x, t], dim=1))
        return u

    def net_f(self, x, t):
        """ Autograd for calculating the residual for different systems."""
        u = self.net_u(x, t)

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

        if 'convection' in self.system or 'diffusion' in self.system:
            f = u_t - self.nu*u_xx + self.beta*u_x - self.G
        elif 'rd' in self.system:
            f = u_t - self.nu*u_xx - self.rho*u + self.rho*u**2
        elif 'reaction' in self.system:
            f = u_t - self.rho*u + self.rho*u**2
        return f

    def net_b_derivatives(self, u_lb, u_ub, x_bc_lb, x_bc_ub):
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

    def loss_pinn(self, type_="u"):

        sub_loss=0
        if type_=='total' or type_=='u':
            u_pred = self.net_u(self.x_u, self.t_u)
            loss_u = torch.mean((self.u - u_pred) ** 2)
            sub_loss = loss_u
        if type_=='total' or type_=='b':
            u_pred_lb = self.net_u(self.x_bc_lb, self.t_bc_lb)
            u_pred_ub = self.net_u(self.x_bc_ub, self.t_bc_ub)
            if self.nu != 0:
                u_pred_lb_x, u_pred_ub_x = self.net_b_derivatives(u_pred_lb, u_pred_ub, self.x_bc_lb, self.x_bc_ub)
            loss_b = torch.mean((u_pred_lb - u_pred_ub) ** 2)
            sub_loss= loss_b
        if type_=='total' or type_=='f':
            f_pred = self.net_f(self.x_f, self.t_f)
            loss_f = torch.mean(f_pred ** 2)
            sub_loss = loss_f
            
        if type_=='total':
            loss = loss_u + loss_b + self.L*loss_f
            return loss
        else:
            return sub_loss

    def train(self):
        self.dnn.train()
        self.optimizer.step(self.loss_pinn)

    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)

        self.dnn.eval()
        u = self.net_u(x, t)
        u = u.detach().cpu().numpy()

        return u


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


class PhysicsInformedNN_pbc_helper():
    
    def __init__(self, beta, L, device):
        self.model = PhysicsInformedNN_pbc(beta, L, device=device)

    def get_PINN(self, device):
        return self.model.dnn.to(device)


    def get_errors(self, dnn, type_="u"):
        self.model.dnn = dnn
        return self.model.loss_pinn(type_=type_)
