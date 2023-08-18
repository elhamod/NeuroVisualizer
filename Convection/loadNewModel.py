# Code is copied with necessary refactoring from https://github.com/arkadaw9/r3_sampling_icml2023

import torch
from torch import nn
from collections import OrderedDict

class DNN(torch.nn.Module):
    def __init__(self, layers, activation, use_batch_norm=False, use_instance_norm=False, init="default"):
        super(DNN, self).__init__()
        
        # parameters
        self.depth = len(layers) - 1
        
        self.activation = self.get_activation(activation)
        self.use_batch_norm = use_batch_norm
        self.use_instance_norm = use_instance_norm

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, nn.Linear(layers[i], layers[i+1]))
            )

            if self.use_batch_norm:
                layer_list.append(('batchnorm_%d' % i, torch.nn.BatchNorm1d(num_features=layers[i+1])))
            if self.use_instance_norm:
                layer_list.append(('instancenorm_%d' % i, torch.nn.InstanceNorm1d(num_features=layers[i+1])))

            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)
        
        
        self.init_weights(init=init)
        
            
    def get_activation(self, activation):
        if activation == 'identity':
            return torch.nn.Identity
        elif activation == 'tanh':
            return torch.nn.Tanh
        elif activation == 'relu':
            return torch.nn.ReLU
        elif activation == 'gelu':
            return torch.nn.GELU
#         elif activation == 'sin':
#             return Sine
    
    def forward(self, inp):
        out = self.layers(inp)
        return out
    
    def init_weights(self, init):
        if init in ['xavier_uniform','kaiming_uniform', 'xavier_normal', 'kaiming_normal']:
            with torch.no_grad():
                print(f"Initializing Network with {init} Initialization..")
                for m in self.layers:
                    if hasattr(m, 'weight'):
                        if init == "xavier_uniform":
                            nn.init.xavier_uniform_(m.weight)
                        elif init == "kaiming_uniform":
                            nn.init.kaiming_uniform_(m.weight)
                        elif init == "xavier_normal":
                            nn.init.xavier_normal_(m.weight)
                        elif init == "kaiming_normal":
                            nn.init.kaiming_normal_(m.weight)
                        m.bias.data.fill_(0.0)


def get_PINN(layers, activation, device):
    return DNN(layers=layers, 
                        activation=activation
                        ).to(device)