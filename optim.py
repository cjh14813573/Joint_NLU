import math
import torch

'''
class Adam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam,self).__init__(params,defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
'''

class NoamOpt:
    def __init__(self, embeddings_size, factor, warmup, optimizer):
        self.embeddings_size = embeddings_size
        self.factor = factor
        self.warmup = warmup
        self.optimizer = optimizer
        self._step = 1

    def stat_dict(self):
        return {'step': self._step,
                'optimizer': self.optimizer.state_dict()}

    def load_state_dict(self, state_dict):
        self._step = state_dict['step']
        self.optimizer.load_state_dict(state_dict['optimizer'])

    def zero_grade(self):
        return self.optimizer.zero_grade()

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self.optimizer.step()

    def curr_step(self):
        return self._step

    def rate(self, step=None):
        if step is None:
            step = self._step

        return self.factor * (self.embeddings_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

