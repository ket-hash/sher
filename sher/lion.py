#partially adapted from https://github.com/google/automl/tree/master/lion
import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from .types import Params, LossClosure, OptLossClosure, Betas2, Truncated, State, OptFloat
from .constants import PI

class Lion(Optimizer):
    def __init__(
        self,
        params: Params,
        lr: float = 1e-4,
        betas: Betas2 = (0.9, 0.99),
        reduction: str = 'p-norm',
        theta: int = 1,
        truncated: Truncated = (False, None),
        weight_decay: float = 0.0,
        a: float = None
    ):
        self._reductions = ['p-norm', 'entropy', 'huber', 'relativistic', 
                            'log-smoothed', 'sorting-norm-exp-decay', 'sorting-norm-sigmoid-decay',
                           'abs-max', 'sorting-norm-softmax-decay', 'sorting-norm-cosine-decay',
                           ]
        if reduction not in self._reductions:
            raise ValueError("reduction not implemented: {}".format(reduction))
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(betas[1])
            )
        if weight_decay < 0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay)
            )
        if reduction == 'entropy' and (a == None or a <= 0):
            raise ValueError(
                "Invalid entropy regularizer: {}".format(a)
            )
        if reduction == 'huber' and (a == None or a < 0):
            raise ValueError(
                "Invalid huber anchor: {}".format(a)
            )
        if reduction == 'log-smoothed' and (a == None or a < 0):
            raise ValueError(
                "Invalid scaling param: {}".format(a)
            )
        if reduction == 'sorting-norm-exp-decay' and (a == None or a <= 0):
            raise ValueError(
                "Invalid scaling param: {}".format(a)
            )
        if reduction == 'sorting-norm-sigmoid-decay' and (a == None):
            raise ValueError(
                "Scaling param cannot be: {}".format(a)
            )
        if reduction == 'abs-max' and (a == None or a <= 0):
            raise ValueError(
                "Scaling param cannot be: {}".format(a)
            )
        if reduction == 'sorting-norm-cosine-decay' and (a == None or a <= 0):
            raise ValueError(
                "Scaling param cannot be: {}".format(a)
            )
            
        defaults = dict(lr = lr, betas = betas, reduction = reduction, theta = theta,
                        truncated = truncated, weight_decay = weight_decay, a = a)
        super().__init__(params, defaults)
        
    def _lp(self, update, theta, truncated, e):
        decay = torch.sign(update).mul_(torch.abs(update).pow(theta-1)).div_(torch.norm(update, theta).pow(theta-1))
        if not truncated:
            return decay
        else:
            mask = torch.where(torch.norm(update, theta) - update > 0, 1.0, 0.0)
            return mask.mul_(decay)

    def _entropy(self, update, a):
        return torch.tanh(update.mul_(a))

    def _huber(self, update, a):
        return torch.clamp(update, -a, a).div_(a)

    def _relat(self, update, a):
        return update.div(torch.sqrt(update.pow(2).add(a**2)))

    def _lsm(self, update, a):
        return update.div_(torch.abs(update).add_(a))

    def _rank(self, update, a, c):
        abs_x = torch.abs(update)
        ranks = torch.argsort(torch.argsort(abs_x, descending=True)).cuda()
        return c[ranks].mul_(torch.sign(update))
        
    @torch.no_grad()
    def step(self, closure: OptLossClosure = None) -> OptFloat:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data.mul_(1 - group["lr"] * group["weight_decay"])

                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                beta1, beta2 = group["betas"]
                theta = group["theta"]
                truncated, e = group["truncated"]
                reduction = group["reduction"]

                update = exp_avg * beta1 + grad * (1 - beta1)
                
                if reduction == 'entropy':
                    a = group["a"]
                    p.add_(self._entropy(update, a), alpha=-group["lr"])
                if reduction == 'huber':
                    a = group["a"]
                    p.add_(self._huber(update, a), alpha=-group["lr"])
                if reduction == 'relativistic':
                    a = group["a"]
                    p.add_(self._relat(update, a), alpha=-group["lr"])
                if reduction == 'log-smoothed':
                    a = group["a"]
                    p.add_(self._lsm(update, a), alpha=-group["lr"])
                if reduction == 'sorting-norm-exp-decay':
                    a = group["a"]
                    c = torch.exp(-a*torch.arange(torch.numel(update))).cuda()
                    p.add_(self._rank(update, a, c), alpha=-group["lr"])
                if reduction == 'sorting-norm-sigmoid-decay':
                    a = group["a"]
                    c = torch.exp(-torch.arange(torch.numel(update)).add(a)).add_(1.0).pow(-1).cuda()
                    p.add_(self._rank(update, a, c), alpha=-group["lr"])
                if reduction == 'sorting-norm-softmax-decay':
                    a = group["a"]
                    c = F.softmax(torch.arange(torch.numel(update)).float().mul_(-a), dim = 0).cuda()
                    p.add_(self._rank(update, a, c), alpha=-group["lr"])
                if reduction == 'sorting-norm-cosine-decay':
                    a = group["a"]
                    d = torch.numel(update)
                    c = torch.cos(torch.arange(d).mul(PI/2*d)).mul(a).cuda()
                    p.add_(self._rank(update, a, c), alpha=-group["lr"])
                if reduction == 'abs-max':
                    a = group["a"]
                    c = F.one_hot(torch.tensor([0]), num_classes=torch.numel(update)).float().squeeze().mul_(a).cuda()
                    p.add_(self._rank(update, a, c), alpha=-group["lr"])
                
                else:
                    p.add_(self._lp(update, theta, truncated, e), alpha=-group["lr"])
                    
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss

    def reductions(self):
        return self._reductions
    
    