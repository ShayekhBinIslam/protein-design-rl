import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal


def get_acq_fn(args):
    if args.acq_fn.lower() == "ucb":
        return UCB
    elif args.acq_fn.lower() == "ei":
        return EI
    else:
        return NoAF


class AcquisitionFunctionWrapper():
    def __init__(self, model):
        self.model = model

    def __call__(self, x):
        raise NotImplementedError()

    def update(self, data):
        self.fit(data)

    def fit(self, data):
        self.model.fit(data)

class NoAF(AcquisitionFunctionWrapper):
    def __call__(self, x):
        return self.l2r(self.model(x))

class UCB(AcquisitionFunctionWrapper):
    def __init__(self, model, sequences):
        super().__init__(model)
        self.kappa = 0.1

    def __call__(self, mean, std):
        return mean + self.kappa * std

class EI(AcquisitionFunctionWrapper):
    def __init__(self, model, sequences):
        super().__init__(model)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sigmoid = nn.Sigmoid()
        self.best_f = None
        self.sequences = sequences

    def __call__(self, mean, std):
        if self.best_f is None:
            self.best_f = torch.tensor(self.model.get_fitness(self.sequences).max())

        mean, std = torch.tensor(mean), torch.tensor(std)
        self.best_f = self.best_f.to(mean)
        # deal with batch evaluation and broadcasting
        #view_shape = mean.shape[:-2] #if mean.dim() >= x.dim() else x.shape[:-2]
        #mean = mean.view(view_shape)
        #std = std.view(view_shape)

        u = (mean - self.best_f.expand_as(mean)) / std

        normal = Normal(torch.zeros_like(u), torch.ones_like(u))
        ucdf = normal.cdf(u)
        updf = torch.exp(normal.log_prob(u))
        ei = std * (updf + u * ucdf)
        return ei.cpu().numpy()

    def update(self, data):
        self.best_f = self._get_best_f(data)
        self.fit(data)
