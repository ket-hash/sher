import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, utils
from utils import add_sher_to_path, tuple_type, truncated_type, update_config_from_args
from dataclasses import dataclass
import argparse
from ResNet import Bottleneck, ResNet, ResNet50
Lion, Betas2, Truncated, _ = add_sher_to_path()

@dataclass
class Config:
    batch_size: int = 64
    test_batch_size: int = 1000
    epochs: int = 15
    lr: float = 0.01
    reduction: str = 'p-norm'    
    betas: Betas2 = (0.9, 0.99)
    theta: int = 1
    weight_decay: float = 0.0
    truncated: Truncated = (False, None)
    a: float = None
    no_cuda: bool = True
    seed: int = 42
    log_interval: int = 10


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(conf, model, device, train_loader, optimizer, epoch, writer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % conf.log_interval == 0:
            loss = loss.item()
            idx = batch_idx + epoch * (len(train_loader))
            writer.add_scalar("Loss/train", loss, idx)
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss,
                )
            )


def test(conf, model, device, test_loader, epoch, scheduler, writer):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    scheduler.step(test_loss)
    fmt = "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n"
    print(
        fmt.format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )

    writer.add_scalar("Accuracy", correct, epoch)
    writer.add_scalar("Loss/test", test_loss, epoch)


def prepare_loaders(conf, use_cuda=False):
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            "../data",
            train=True,
            download=True,
            transform=transform
            ),
        batch_size=conf.batch_size,
        shuffle=True,
        **kwargs,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            "../data",
            train=False,
            transform=transform
            ),
        batch_size=conf.test_batch_size,
        shuffle=True,
        **kwargs,
    )
    return train_loader, test_loader

def parse_args():
    parser = argparse.ArgumentParser(description='Train CIFAR-10 with Lion-K')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=1000, help='Batch size for testing')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--reduction', type=str, default='p-norm', help='Convex reduction function')
    parser.add_argument('--betas', type=tuple_type, default=(0.9, 0.99), help='Convex reduction function')
    parser.add_argument('--theta', type=int, default=1, help='L-p norm to use for p-norm reduction')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='optimizer weight decay')
    parser.add_argument('--truncated', type=truncated_type, default=(False, None), help='(bool, truncation factor): if True, truncates the norm to truncation factor')
    parser.add_argument('--a', type=float, default=None, help='Scaling factor for optimizer')
    parser.add_argument('--no_cuda', action='store_true', default=True, help='Disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--log_interval', type=int, default=10, help='How many batches to wait before logging training status')
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    conf = Config()
    update_config_from_args(conf, args)
    log_dir = "runs/CIFAR10-cos-decay-a=1.2-lr=5e-6"
    print("Tensorboard: tensorboard --logdir={}".format(log_dir))

    with SummaryWriter(log_dir) as writer:
        use_cuda = not conf.no_cuda and torch.cuda.is_available()
        torch.manual_seed(conf.seed)
        #device = torch.device("cuda" if use_cuda else "cpu")
        device = torch.device("cuda")
        print("running on: ", device)
        train_loader, test_loader = prepare_loaders(conf, use_cuda)
        
        #model = Net().to(device)
        model = ResNet50(10).to(device)
        
        images, labels = next(iter(train_loader))
        img_grid = utils.make_grid(images)
        writer.add_image("CIFAR10_images", img_grid)

        optimizer = Lion(model.parameters(), lr=conf.lr, betas=conf.betas, reduction=conf.reduction,
                        theta=conf.theta, truncated=conf.truncated, weight_decay=conf.weight_decay, a=conf.a)
        #optimizer = optim.AdamW(model.parameters(), lr = 1e-3, weight_decay = conf.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=conf.epochs)
        scaler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)


        
        
        for epoch in range(1, conf.epochs + 1):
            train(conf, model, device, train_loader, optimizer, epoch, writer)
            test(conf, model, device, test_loader, epoch, scaler, writer)
            scheduler.step()
            for name, param in model.named_parameters():
                writer.add_histogram(name, param, epoch)
                writer.add_histogram("{}.grad".format(name), param.grad, epoch)


if __name__ == "__main__":
    main()