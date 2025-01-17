import argparse
import copy
import math
import sys

import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
from tensorboardX import SummaryWriter

import datasets
import flows as fnn
import utils
from utils import plot_pca
from MLP import MLP

if sys.version_info < (3, 6):
    print('Sorry, this code might need Python 3.6 or higher')

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Flows')
parser.add_argument(
    '--batch-size',
    type=int,
    default=100,
    help='input batch size for training (default: 100)')
parser.add_argument(
    '--test-batch-size',
    type=int,
    default=1000,
    help='input batch size for testing (default: 1000)')
parser.add_argument(
    '--epochs',
    type=int,
    default=1000,
    help='number of epochs to train (default: 1000)')
parser.add_argument(
    '--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
parser.add_argument(
    '--dataset',
    default='POWER',
    help='POWER | GAS | HEPMASS | MINIBONE | BSDS300 | MOONS')
parser.add_argument(
    '--flow', default='maf', help='flow to use: maf | realnvp | glow')
parser.add_argument(
    '--no-cuda',
    action='store_true',
    default=False,
    help='disables CUDA training')
parser.add_argument(
    '--cond',
    action='store_true',
    default=False,
    help='train class conditional flow (only for MNIST)')
parser.add_argument(
    '--num-blocks',
    type=int,
    default=5,
    help='number of invertible blocks (default: 5)')
parser.add_argument(
    '--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument(
    '--log-interval',
    type=int,
    default=1000,
    help='how many batches to wait before logging training status')
parser.add_argument(
    '--mini',
    action='store_true',
    default=False,
)
parser.add_argument(
    '--flow_num_hidden',
    type=int,
    default=512
)
parser.add_argument(
    '--mlp_num_hidden',
    type=int,
    default=64
)
parser.add_argument(
    '--cond_inputs',
    type=int,
    default=10
)
parser.add_argument(
    '--cond_inputs',
    type=int,
    default=10
)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda:0" if args.cuda else "cpu")

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    
kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

assert args.dataset in [
    'POWER', 'GAS', 'HEPMASS', 'MINIBONE', 'BSDS300', 'MOONS', 'MNIST', 'Bimodal'
]
#####
sigma = True
dataset = getattr(datasets, args.dataset)(sigma=sigma)
print(dataset.trn.x.shape)
print(dataset.trn.y.shape)
print(dataset.val.x.shape)
print(dataset.val.y.shape)
print(dataset.tst.x.shape)
print(dataset.tst.y.shape)

if args.mini:
    trn_nums = 10000
    val_nums = 2000
    test_nums = 1000

    trn_ra = npr.randint(0, 50000, trn_nums)
    val_ra = npr.randint(0, 10000, val_nums)
    tst_ra = npr.randint(0, 10000, test_nums)
else:
    trn_ra = np.arange(0, dataset.trn.N, 1)
    val_ra = np.arange(0, dataset.val.N, 1)
    tst_ra = np.arange(0, dataset.tst.N, 1)


if args.cond:
    assert args.flow in ['maf', 'realnvp'] and args.dataset in ['MNIST', 'Bimodal'], \
        'Conditional flows are implemented only for maf and MNIST'
    
    train_tensor = torch.from_numpy(dataset.trn.x[trn_ra])
    train_labels = torch.from_numpy(dataset.trn.y[trn_ra])
    train_dataset = torch.utils.data.TensorDataset(train_tensor, train_labels)

    valid_tensor = torch.from_numpy(dataset.val.x[val_ra])
    valid_labels = torch.from_numpy(dataset.val.y[val_ra])
    valid_dataset = torch.utils.data.TensorDataset(valid_tensor, valid_labels)

    test_tensor = torch.from_numpy(dataset.tst.x[tst_ra])
    test_labels = torch.from_numpy(dataset.tst.y[tst_ra])
    test_dataset = torch.utils.data.TensorDataset(test_tensor, test_labels)
    num_cond_inputs = dataset.trn.y.shape[1]
else:
    train_tensor = torch.from_numpy(dataset.trn.x)
    train_dataset = torch.utils.data.TensorDataset(train_tensor)

    valid_tensor = torch.from_numpy(dataset.val.x)
    valid_dataset = torch.utils.data.TensorDataset(valid_tensor)

    test_tensor = torch.from_numpy(dataset.tst.x)
    test_dataset = torch.utils.data.TensorDataset(test_tensor)
    num_cond_inputs = None
    
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=args.test_batch_size,
    shuffle=False,
    drop_last=False,
    **kwargs)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.test_batch_size,
    shuffle=False,
    drop_last=False,
    **kwargs)

num_inputs = dataset.n_dims

num_hidden = args.num_hidden

act = 'tanh' if args.dataset == 'GAS' else 'relu'

modules = []

assert args.flow in ['maf', 'maf-split', 'maf-split-glow', 'realnvp', 'glow']
if args.flow == 'glow':
    mask = torch.arange(0, num_inputs) % 2
    mask = mask.to(device).float()

    print("Warning: Results for GLOW are not as good as for MAF yet.")
    for _ in range(args.num_blocks):
        modules += [
            fnn.BatchNormFlow(num_inputs),
            fnn.LUInvertibleMM(num_inputs),
            fnn.CouplingLayer(
                num_inputs, num_hidden, mask, num_cond_inputs,
                s_act='tanh', t_act='relu')
        ]
        mask = 1 - mask
elif args.flow == 'realnvp':
    mask = torch.arange(0, num_inputs) % 2
    mask = mask.to(device).float()

    for _ in range(args.num_blocks):
        modules += [
            fnn.CouplingLayer(
                num_inputs, num_hidden, mask, num_cond_inputs,
                s_act='tanh', t_act='relu'),
            fnn.BatchNormFlow(num_inputs)
        ]
        mask = 1 - mask
elif args.flow == 'maf':
    for _ in range(args.num_blocks):
        modules += [
            fnn.MADE(num_inputs, num_hidden, num_cond_inputs, act=act),
            fnn.BatchNormFlow(num_inputs),
            fnn.Reverse(num_inputs)
        ]
elif args.flow == 'maf-split':
    for _ in range(args.num_blocks):
        modules += [
            fnn.MADESplit(num_inputs, num_hidden, num_cond_inputs,
                         s_act='tanh', t_act='relu'),
            fnn.BatchNormFlow(num_inputs),
            fnn.Reverse(num_inputs)
        ]
elif args.flow == 'maf-split-glow':
    for _ in range(args.num_blocks):
        modules += [
            fnn.MADESplit(num_inputs, num_hidden, num_cond_inputs,
                         s_act='tanh', t_act='relu'),
            fnn.BatchNormFlow(num_inputs),
            fnn.InvertibleMM(num_inputs)
        ]

model = fnn.FlowSequential(*modules)

for module in model.modules():
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight)
        if hasattr(module, 'bias') and module.bias is not None:
            module.bias.data.fill_(0)

mlp = MLP()
mlp.to(device)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

writer = SummaryWriter(comment=f'{args.num_hidden}')
global_step = 0

def train(epoch):
    global global_step, writer
    model().train()
    train_loss = 0

    pbar = tqdm(total=len(train_loader.dataset))
    for batch_idx, data in enumerate(train_loader):
        if isinstance(data, list):
            if len(data) > 1:
                cond_data = data[1].float()
                cond_data = cond_data.to(device)
            else:
                cond_data = None

            data = data[0]
        data = data.to(device)
        optimizer.zero_grad()
        cond_data = mlp(cond_data)
        loss = -model.log_probs(data, cond_data).mean()
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        pbar.update(data.size(0))
        pbar.set_description('Train, Log likelihood in nats: {:.6f}'.format(
            -train_loss / (batch_idx + 1)))
        
        writer.add_scalar('training/loss', loss.item(), global_step)
        global_step += 1
        
    pbar.close()
        
    for module in model.modules():
        if isinstance(module, fnn.BatchNormFlow):
            module.momentum = 0

    if args.cond:
        with torch.no_grad():
            Us = model(train_loader.dataset.tensors[0].to(data.device),
                train_loader.dataset.tensors[1].to(data.device).float())[0].cpu().numpy()
            plot_pca(Us, 2, epoch)
    else:
        with torch.no_grad():
            Us = model(train_loader.dataset.tensors[0].to(data.device))


    for module in model.modules():
        if isinstance(module, fnn.BatchNormFlow):
            module.momentum = 1


def validate(epoch, model, loader, prefix='Validation'):
    global global_step, writer

    model.eval()
    val_loss = 0

    pbar = tqdm(total=len(loader.dataset))
    pbar.set_description('Eval')
    for batch_idx, data in enumerate(loader):
        if isinstance(data, list):
            if len(data) > 1:
                cond_data = data[1].float()
                cond_data = cond_data.to(device)
            else:
                cond_data = None

            data = data[0]
        data = data.to(device)
        with torch.no_grad():
            val_loss += -model.log_probs(data, cond_data).sum().item()  # sum up batch loss
        pbar.update(data.size(0))
        pbar.set_description('Val, Log likelihood in nats: {:.6f}'.format(
            -val_loss / pbar.n))

    writer.add_scalar('validation/LL', val_loss / len(loader.dataset), epoch)

    pbar.close()
    return val_loss / len(loader.dataset)


best_validation_loss = float('inf')
best_validation_epoch = 0
best_model = model

for epoch in range(args.epochs):
    print('\nEpoch: {}'.format(epoch))

    train(epoch)
    validation_loss = validate(epoch, model, valid_loader)

    if epoch - best_validation_epoch >= 50:
        break

    if validation_loss < best_validation_loss:
        best_validation_epoch = epoch
        best_validation_loss = validation_loss
        best_model = copy.deepcopy(model)

    print(
        'Best validation at epoch {}: Average Log Likelihood in nats: {:.4f}'.
        format(best_validation_epoch, -best_validation_loss))

    if args.dataset == 'MOONS' and epoch % 10 == 0:
        utils.save_moons_plot(epoch, model, dataset)
    elif args.dataset == 'MNIST' and epoch % 5 == 0:
        utils.save_images(epoch, model, args.cond)
    elif args.dataset == 'Bimodal' and epoch % 1 == 0:
        utils.sample_dist(epoch, model, sigma=sigma)


validate(best_validation_epoch, best_model, test_loader, prefix='Test')
