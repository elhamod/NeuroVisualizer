# Code is copied with necessary refactoring from https://github.com/PredictiveIntelligenceLab/PINNsNTK 

import numpy as np
import torch

# Define PINN model
a = 0.5
c = 2
ics_coords = np.array([[0.0, 0.0],
                        [0.0, 1.0]])
bc1_coords = np.array([[0.0, 0.0],
                        [1.0, 0.0]])
bc2_coords = np.array([[0.0, 1.0],
                        [1.0, 1.0]])
dom_coords = np.array([[0.0, 0.0],
                        [1.0, 1.0]])


def u_tt(x, a, c):
    t = x[:,0:1]
    x = x[:,1:2]
    u_tt = -(c * np.pi)**2 * np.sin( np.pi * x) * np.cos(c * np.pi * t) - \
            a * (4 * c * np.pi)**2 *  np.sin(2 * c * np.pi* x) * np.cos(4 * c * np.pi * t)
    return u_tt

def u_xx(x, a, c):
    t = x[:,0:1]
    x = x[:,1:2]
    u_xx = - np.pi**2 * np.sin( np.pi * x) * np.cos(c * np.pi * t) - \
              a * (2 * c * np.pi)** 2 * np.sin(2 * c * np.pi* x) * np.cos(4 * c * np.pi * t)
    return  u_xx


def u(x, a, c):
    """
    :param x: x = (t, x)
    """
    t = x[:,0:1]
    x = x[:,1:2]
    return np.sin(np.pi * x) * np.cos(c * np.pi * t) + \
            a * np.sin(2 * c * np.pi* x) * np.cos(4 * c  * np.pi * t)


def r(x, a, c):
    return u_tt(x, a, c) - c**2 * u_xx(x, a, c)


class Sampler:
    # Initialize the class
    def __init__(self, dim, coords, func, name = None):
        self.dim = dim
        self.coords = coords
        self.func = func
        self.name = name
    def sample(self, N):
        x = self.coords[0:1,:] + (self.coords[1:2,:]-self.coords[0:1,:])*np.random.rand(N, self.dim)
        y = self.func(x)
        return x, y



res_sampler = Sampler(2, dom_coords, lambda x: r(x, a, c), name='Forcing')
X, _ = res_sampler.sample(np.int32(1e5))
mu_X, sigma_X = X.mean(0), X.std(0)
mu_t, sigma_t = mu_X[0], sigma_X[0]
mu_x, sigma_x = mu_X[1], sigma_X[1]

def net_u_t(model, t, x):
    # Ensure t is a tensor with gradient tracking
    if not torch.is_tensor(t):
        t = torch.from_numpy(t)
        x = torch.from_numpy(x)
    else:
        t = t.clone()
        x = x.clone()

    t = t.requires_grad_(True)

    u = net_u(model, t, x)

    # Compute du/dt
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0] / sigma_t

    return u_t

# Forward pass for u
def net_u(model, t_u, x_u):
    # Combine the sliced data
    # Note: This step may vary depending on how your model uses these inputs
    if not torch.is_tensor(t_u):
        t_u = torch.from_numpy(t_u)
        x_u = torch.from_numpy(x_u)

    t_u = t_u.to(model[0].weight.device).requires_grad_(True)
    x_u = x_u.to(model[0].weight.device).requires_grad_(True)

    input_tensor = torch.cat((t_u, x_u), dim=1)

    u = model(input_tensor.float())
    return u



def operator(u, t, x, c, sigma_t=1.0, sigma_x=1.0):
    # u = u.clone().requires_grad_(True)
    # t = t.clone().requires_grad_(True)
    # x = x.clone().requires_grad_(True)      
    # u = u.requires_grad_(True)
    # t = t.requires_grad_(True)
    # x = x.requires_grad_(True)


    # Gradients w.r.t t and x
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0] / sigma_t
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0] / sigma_x

    # Second order gradients
    u_tt = torch.autograd.grad(u_t, t, grad_outputs=torch.ones_like(u_t), create_graph=True)[0] / sigma_t
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0] / sigma_x

    # Residual
    residual = u_tt - c**2 * u_xx

    return residual


# Forward pass for the residual
def net_r(model, t, x):
    if not torch.is_tensor(t):
        t = torch.from_numpy(t).to(model[0].weight.device)
        x = torch.from_numpy(x).to(model[0].weight.device)
    t = t.requires_grad_(True)
    x = x.requires_grad_(True)

    u = net_u(model, t, x)
    residual = operator(u, t, x,
                                c,
                                sigma_t,
                                sigma_x)
    return residual
    

# Evaluates predictions at test points
def predict_u(X_star, model):
    X_star = (X_star - mu_X) / sigma_X
    # tf_dict = {t_u_tf: X_star[:, 0:1], self.x_u_tf: X_star[:, 1:2]}
    # u_star = self.sess.run(self.u_pred, tf_dict)

    if not torch.is_tensor(X_star):
        X_star = torch.from_numpy(X_star).to(model[0].weight.device)

    t_u = X_star[:, 0:1]
    x_u = X_star[:, 1:2]

    # Pass the data through the model
    u_star = net_u(model, t_u, x_u)

    # Detach the output from the computation graph and convert to numpy array
    u_star = u_star.detach().cpu().numpy()

    return u_star

    # Evaluates predictions at test points

def predict_r(X_star, model):
    X_star = (X_star - mu_X) / sigma_X
    # tf_dict = {self.t_r_tf: X_star[:, 0:1], self.x_r_tf: X_star[:, 1:2]}
    # r_star = self.sess.run(self.r_pred, tf_dict)

    if not torch.is_tensor(X_star):
        X_star = torch.from_numpy(X_star).to(model[0].weight.device)

    # Slice the input data
    t_r = X_star[:, 0:1]
    x_r = X_star[:, 1:2]

    # Pass the data through the model
    r_star = net_r(model, t_r, x_r)

    # Detach the output from the computation graph and convert to numpy array
    r_star = r_star.detach().cpu().numpy()

    return r_star


def fetch_minibatch(self, sampler, N):
    X, Y = sampler.sample(N)
    X = (X - self.mu_X) / self.sigma_X
    return X, Y





nn = 200
t = np.linspace(dom_coords[0, 0], dom_coords[1, 0], nn)[:, None]
x = np.linspace(dom_coords[0, 1], dom_coords[1, 1], nn)[:, None]
t, x = np.meshgrid(t, x)
X_star = np.hstack((t.flatten()[:, None], x.flatten()[:, None]))

u_star = u(X_star, a, c)
R_star = r(X_star, a, c)


def get_PINN(layer_sizes, device):
    layers = []
    for i, j in zip(layer_sizes[:-1], layer_sizes[1:]):
        layer = torch.nn.Linear(i, j)
        layers.append(layer)
        layers.append(torch.nn.Tanh())
    layers = layers[:-1]
    return torch.nn.Sequential(*layers).to(device)


def get_errors(model, type="u"):
    if type=="ic":
        error_ics_u_t = torch.mean(net_u_t(model, ics_coords[:, 0:1], ics_coords[:, 1:2])**2)
        return error_ics_u_t
    if type=="bc":
        loss_ics_u = torch.mean(( torch.from_numpy(u(ics_coords, a, c)).to(model[0].weight.device) - net_u(model, ics_coords[:, 0:1], ics_coords[:, 0:1]))**2)
        loss_bc1 = torch.mean(net_u(model, bc1_coords[:, 0:1], bc1_coords[:, 1:2])**2)
        loss_bc2 = torch.mean(net_u(model, bc2_coords[:, 0:1], bc2_coords[:, 1:2])**2)
        error_bcs =loss_ics_u + loss_bc1 + loss_bc2
        return error_bcs
    elif type=="r":
        r_pred = predict_r(X_star, model)
        error_r = np.linalg.norm(R_star - r_pred, 2) / (np.linalg.norm(R_star, 2) + np.finfo(np.float32).eps)
        return torch.from_numpy(np.array([error_r])).to(model[0].weight.device)
    else: ##u
        u_pred = predict_u(X_star, model)
        error_u = np.linalg.norm(u_star - u_pred, 2) / (np.linalg.norm(u_star, 2) + np.finfo(np.float32).eps)
        return torch.from_numpy(np.array([error_u])).to(model[0].weight.device)