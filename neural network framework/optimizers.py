from abc import abstractmethod
import numpy
from tensor import Tensor

class ABC():
    pass

class Optimizer(ABC):
    def __init__(self, params):
        self.params = params
    
    @abstractmethod
    def step(self):
        pass

    def reset_grad(self):
        for p in self.params:
            p.grad = None

    def zero_grad(self):
        for p in self.params:
            p.grad = None

class SGD(Optimizer):
    def __init__(self, params, lr = 0.01):
        super().__init__(params)
        self.lr = lr

    def step(self):
        for param in self.params:
            if param.grad is not None:
                if isinstance(param.grad, Tensor):
                    grad_data = param.grad.realize_cached_data()
                else:
                    grad_data = param.grad

                param_data = param.realize_cached_data()
                new_data = param_data - self.lr * grad_data

                param.cached_data = new_data

class Adam(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params)
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.t = 0

        self.m = []
        self.v = []
        for param in self.params:
            param_data = param.realize_cached_data()
            self.m.append(numpy.zeros_like(param_data))
            self.v.append(numpy.zeros_like(param_data))

    def step(self):
        self.t += 1

        for i, param in enumerate(self.params):
            if param.grad is not None:
                if isinstance(param.grad, Tensor):
                    grad_data = param.grad.realize_cached_data()
                else:
                    grad_data = param.grad

                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad_data
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad_data ** 2)

                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)

                param_data = param.realize_cached_data()
                new_data = param_data - self.lr * m_hat / (numpy.sqrt(v_hat) + self.eps)

                param.cached_data = new_data