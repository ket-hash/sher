import torch
from torch.utils.tensorboard import SummaryWriter
from utils import add_sher_to_path, tuple_type, truncated_type, update_config_from_args
from dataclasses import dataclass
import argparse

Lion, Betas2, Truncated, PI = add_sher_to_path()

@dataclass
class Config:
    max_iters: int = 1000
    n_dim: int = 2
    lr: float = 0.01
    objective: str = 'rosenbrock'
    reduction: str = 'p-norm'    
    betas: Betas2 = (0.9, 0.99)
    theta: int = 1
    weight_decay: float = 0.0
    truncated: Truncated = (False, None)
    a: float = None
    seed: int = 42
    log_interval: int = 10

def rosenbrock(x, a=1, b=100):
    return torch.sum(b * (x[1:] - x[:-1]**2)**2 + (a - x[:-1])**2)

def rastrigin(x, A=10):
    return 2*A + torch.sum(x.pow(2) - A*torch.cos(2*PI*x))

def sphere(x):
    return torch.sum(x.pow(2))
    
def styblinski_tang(x):
    y = x.pow(4) - 16*x.pow(2) + 5*x
    return torch.sum(y).mul(0.5)

def minimize(conf, objective, init, optimizer, writer):
    for idx in range(conf.max_iters):
        optimizer.zero_grad()
        loss = objective(init)
        loss.backward()
        optimizer.step()
        if idx % conf.log_interval == 0:
            loss = loss.item()
            writer.add_scalar("f(x)", loss, idx)
            print(
                "iteration: {} \tLoss: {:.6f} \tParams: {}".format(
                    idx,
                    loss,
                    init.detach().numpy()
                )
            )
            
def parse_args():
    parser = argparse.ArgumentParser(description='Minimize benchmark functions with Lion-K')
    parser.add_argument('--max_iters', type=int, default=1000, help='maximum minimization steps')
    parser.add_argument('--n_dim', type=int, default=2, help='dimension of objective function')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--objective', type=str, default='rosenbrock', help='Objective function to minimize')
    parser.add_argument('--reduction', type=str, default='p-norm', help='Convex reduction function')
    parser.add_argument('--betas', type=tuple_type, default=(0.9, 0.99), help='Convex reduction function')
    parser.add_argument('--theta', type=int, default=1, help='L-p norm to use for p-norm reduction')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='optimizer weight decay')
    parser.add_argument('--truncated', type=truncated_type, default=(False, None), help='(bool, truncation factor): if True, truncates the norm to truncation factor')
    parser.add_argument('--a', type=float, default=None, help='Scaling factor for optimizer')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--log_interval', type=int, default=10, help='How many batches to wait before logging training status')
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    conf = Config()
    update_config_from_args(conf, args)
    log_dir = "runs/minimization_Lion-K"
    print("Tensorboard: tensorboard --logdir={}".format(log_dir))

    with SummaryWriter(log_dir) as writer:
        torch.manual_seed(conf.seed)
        params = torch.randn(conf.n_dim, requires_grad=True)

        optimizer = Lion([params], lr=conf.lr, betas=conf.betas, reduction=conf.reduction,
                        theta=conf.theta, truncated=conf.truncated, weight_decay=conf.weight_decay, a=conf.a)
        
        if conf.objective == 'rosenbrock':
            minimize(conf, rosenbrock, params, optimizer, writer)
        if conf.objective == 'rastrigin':
            minimize(conf, rastrigin, params, optimizer, writer)
        if conf.objective == 'sphere':
            minimize(conf, sphere, params, optimizer, writer)
        if conf.objective == 'rastrigin':
            minimize(conf, rastrigin, params, optimizer, writer)
        if conf.objective == 'styblinski-tang':
            minimize(conf, styblinski_tang, params, optimizer, writer)

if __name__ == "__main__":
    main()