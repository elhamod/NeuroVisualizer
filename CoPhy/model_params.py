# Code is copied with necessary refactoring from https://github.com/marcellodebernardi/loss-landscapes 

from abc import ABC, abstractmethod
import abc
import itertools
import math
import torch
import numpy as np





class ModelParameters:
    """
    A ModelParameters object is an abstract view of a model's optimizable parameters as a tensor. This class
    enables the parameters of models of the same 'shape' (architecture) to be operated on as if they were 'real'
    tensors. A ModelParameters object cannot be converted to a true tensor as it is potentially irregularly
    shaped.
    """

    def __init__(self, parameters: list):
        if not isinstance(parameters, list) and all(isinstance(p, torch.Tensor) for p in parameters):
            raise AttributeError('Argument to ModelParameter is not a list of torch.Tensor objects.')

        self.parameters = parameters

    def __len__(self) -> int:
        """
        Returns the number of model layers within the parameter tensor.
        :return: number of layer tensors
        """
        return len(self.parameters)

    def numel(self) -> int:
        """
        Returns the number of elements (i.e. individual parameters) within the tensor.
        Note that this refers to individual parameters, not layers.
        :return: number of elements in tensor
        """
        return sum(p.numel() for p in self.parameters)

    def __getitem__(self, index) -> torch.nn.Parameter:
        """
        Returns the tensor of the layer at the given index.
        :param index: layer index
        :return: tensor of layer
        """
        return self.parameters[index]

    def __eq__(self, other: 'ModelParameters') -> bool:
        """
        Compares this parameter tensor for equality with the argument tensor, using the == operator.
        :param other: the object to compare to
        :return: true if equal
        """
        if not isinstance(other, ModelParameters) or len(self) != len(other):
            return False
        else:
            return all(torch.equal(p_self, p_other) for p_self, p_other in zip(self.parameters, other.parameters))

    def __add__(self, other: 'ModelParameters') -> 'ModelParameters':
        """
        Constructively returns the result of addition between this tensor and another.
        :param other: other to add
        :return: self + other
        """
        return ModelParameters([self[idx] + other[idx] for idx in range(len(self))])

    def __radd__(self, other: 'ModelParameters') -> 'ModelParameters':
        """
        Constructively returns the result of addition between this tensor and another.
        :param other: model parameters to add
        :return: other + self
        """
        return self.__add__(other)

    def add_(self, other: 'ModelParameters'):
        """
        In-place addition between this tensor and another.
        :param other: model parameters to add
        :return: none
        """
        for idx in range(len(self)):
            self.parameters[idx] += other[idx]#.cuda()

    def __sub__(self, other: 'ModelParameters') -> 'ModelParameters':
        """
        Constructively returns the result of subtracting another tensor from this one.
        :param other: model parameters to subtract
        :return: self - other
        """
        return ModelParameters([self[idx] - other[idx] for idx in range(len(self))])

    def __rsub__(self, other: 'ModelParameters') -> 'ModelParameters':
        """
        Constructively returns the result of subtracting this tensor from another one.
        :param other: other to subtract from
        :return: other - self
        """
        return self.__sub__(other)

    def sub_(self, vector: 'ModelParameters'):
        """
        In-place subtraction of another tensor from this one.
        :param vector: other to subtract
        :return: none
        """
        for idx in range(len(self)):
            self.parameters[idx] -= vector[idx]#.cuda()

    def __mul__(self, scalar) -> 'ModelParameters':
        """
        Constructively returns the result of multiplying this tensor by a scalar.
        :param scalar: scalar to multiply by
        :return: self * scalar
        """
        return ModelParameters([self[idx] * scalar for idx in range(len(self))])

    def __rmul__(self, scalar) -> 'ModelParameters':
        """
        Constructively returns the result of multiplying this tensor by a scalar.
        :param scalar: scalar to multiply by
        :return: scalar * self
        """
        return self.__mul__(scalar)

    def mul_(self, scalar):
        """
        In-place multiplication of this tensor by a scalar.
        :param scalar: scalar to multiply by
        :return: none
        """
        for idx in range(len(self)):
            self.parameters[idx] *= scalar

    def __truediv__(self, scalar) -> 'ModelParameters':
        """
        Constructively returns the result of true-dividing this tensor by a scalar.
        :param scalar: scalar to divide by
        :return: scalar / self
        """
        return ModelParameters([self[idx] / scalar for idx in range(len(self))])

    def truediv_(self, scalar):
        """
        In-place true-division of this tensor by a scalar.
        :param scalar: scalar to divide by
        :return: none
        """
        for idx in range(len(self)):
            self.parameters[idx] /= scalar

    def __floordiv__(self, scalar) -> 'ModelParameters':
        """
        Constructively returns the result of floor-dividing this tensor by a scalar.
        :param scalar: scalar to divide by
        :return: scalar // self
        """
        return ModelParameters([self[idx] // scalar for idx in range(len(self))])

    def floordiv_(self, scalar):
        """
        In-place floor-division of this tensor by a scalar.
        :param scalar: scalar to divide by
        :return: none
        """
        for idx in range(len(self)):
            self.parameters[idx] //= scalar

    def __matmul__(self, other: 'ModelParameters') -> 'ModelParameters':
        """
        Constructively returns the result of tensor-multiplication of this tensor by another tensor.
        :param other: other tensor
        :return: self @ tensor
        """
        raise NotImplementedError()

    def dot(self, other: 'ModelParameters') -> float:
        """
        Returns the vector dot product of this ModelParameters vector and the given other vector.
        :param other: other ModelParameters vector
        :return: dot product of self and other
        """
        param_products = []
        for idx in range(len(self.parameters)):
            param_products.append((self.parameters[idx] * other.parameters[idx]).sum().item())
        return sum(param_products)

    def model_normalize_(self, ref_point: 'ModelParameters', order=2):
        """
        In-place model-wise normalization of the tensor.
        :param ref_point: use this model's norm, if given
        :param order: norm order, e.g. 2 for L2 norm
        :return: none
        """
        for parameter in self.parameters:
            parameter *= (ref_point.model_norm(order) / self.model_norm())

    def layer_normalize_(self, ref_point: 'ModelParameters', order=2):
        """
        In-place layer-wise normalization of the tensor.
        :param ref_point: use this model's layer norms, if given
        :param order: norm order, e.g. 2 for L2 norm
        :return: none
        """
        # in-place normalize each parameter
        for layer_idx, parameter in enumerate(self.parameters, 0):
            parameter *= (ref_point.layer_norm(layer_idx, order) / self.layer_norm(layer_idx, order))

    def filter_normalize_(self, ref_point: 'ModelParameters', order=2):
        """
        In-place filter-wise normalization of the tensor.
        :param ref_point: use this model's filter norms, if given
        :param order: norm order, e.g. 2 for L2 norm
        :return: none
        """
        for l in range(len(self.parameters)):
            # normalize one-dimensional bias vectors
            if len(self.parameters[l].size()) == 1:
                self.parameters[l] *= (ref_point.parameters[l].norm(order) / self.parameters[l].norm(order))
            # normalize two-dimensional weight vectors
            for f in range(len(self.parameters[l])):
                self.parameters[l][f] *= ref_point.filter_norm((l, f), order) / (self.filter_norm((l, f), order))

    def model_norm(self, order=2) -> float:
        """
        Returns the model-wise L-norm of the tensor.
        :param order: norm order, e.g. 2 for L2 norm
        :return: L-norm of tensor
        """
        # L-n norm of model where we treat the model as a flat other
        return math.pow(sum([
            torch.pow(layer, order).sum().item()
            for layer in self.parameters
        ]), 1.0 / order)

    def layer_norm(self, index, order=2) -> float:
        """
        Returns a list of layer-wise L-norms of the tensor.
        :param order: norm order, e.g. 2 for L2 norm
        :param index: layer index
        :return: list of L-norms of layers
        """
        # L-n norms of layer where we treat each layer as a flat other
        return math.pow(torch.pow(self.parameters[index], order).sum().item(), 1.0 / order)

    def filter_norm(self, index, order=2) -> float:
        """
        Returns a 2D list of filter-wise L-norms of the tensor.
        :param order: norm order, e.g. 2 for L2 norm
        :param index: tuple with layer index and filter index
        :return: list of L-norms of filters
        """
        # L-n norm of each filter where we treat each layer as a flat other
        return math.pow(torch.pow(self.parameters[index[0]][index[1]], order).sum().item(), 1.0 / order)

    def as_numpy(self) -> np.ndarray:
        """
        Returns the tensor as a flat numpy array.
        :return: a numpy array
        """
        return np.concatenate([p.numpy().flatten() for p in self.parameters])

    def _get_parameters(self) -> list:
        """
        Returns a reference to the internal parameter data in whatever format used by the source model.
        :return: reference to internal parameter data
        """
        return self.parameters
class ModelWrapper(ABC):
    def __init__(self, modules: list):
        self.modules = modules

    def get_modules(self) -> list:
        return self.modules

    def get_module_parameters(self) -> ModelParameters:
        return ModelParameters([p for module in self.modules for p in module.parameters()])

    def train(self, mode=True) -> 'ModelWrapper':
        for module in self.modules:
            module.train(mode)
        return self

    def eval(self) -> 'ModelWrapper':
        return self.train(False)

    def requires_grad_(self, requires_grad=True) -> 'ModelWrapper':
        for module in self.modules:
            for p in module.parameters():
                p.requires_grad = requires_grad
        return self

    def zero_grad(self) -> 'ModelWrapper':
        for module in self.modules:
            for p in module.parameters():
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()
        return self

    def parameters(self):
        return itertools.chain([module.parameters() for module in self.modules])

    def named_parameters(self):
        return itertools.chain([module.named_parameters() for module in self.modules])

    @abc.abstractmethod
    def forward(self, x):
        pass


class Metric(ABC):
    """ A quantity that can be computed given a model or an agent. """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def __call__(self, model_wrapper: ModelWrapper):
        pass
    
class Loss(Metric):
    """ Computes a specified loss function over specified input-output pairs. """
    def __init__(self, loss_fn, inputs: torch.Tensor, target: torch.Tensor):
        super().__init__()
        self.loss_fn = loss_fn
        self.inputs = inputs
        self.target = target

    def __call__(self, model_wrapper: ModelWrapper) -> float:
        return self.loss_fn(model_wrapper.forward(self.inputs), self.target)
    
class SimpleModelWrapper(ModelWrapper):
    def __init__(self, model: torch.nn.Module):
        super().__init__([model])

    def forward(self, x):
        return self.modules[0](x)

def wrap_model(model):
    if isinstance(model, ModelWrapper):
        return model.requires_grad_(False)
    elif isinstance(model, torch.nn.Module):
        return SimpleModelWrapper(model).requires_grad_(False)
    else:
        raise ValueError('Only models of type torch.nn.modules.module.Module can be passed without a wrapper.')

    

