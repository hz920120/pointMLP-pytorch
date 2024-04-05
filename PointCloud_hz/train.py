import argparse
import os
import logging
import datetime
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import numpy as np
import yaml
import models as models
from torch.utils.data import DataLoader
from data import TeethPointCloudData
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F


def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('training')
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('-cg', '--config', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)', default='configs/test_config_0331.yaml')
    parser.add_argument('--msg', type=str, help='message after checkpoint')
    parser.add_argument('--epoch', default=300, type=int, help='number of epoch in training')
    parser.add_argument('--num_points', type=int, default=1024, help='Point Number')
    parser.add_argument('--learning_rate', default=0.1, type=float, help='learning rate in training')
    parser.add_argument('--min_lr', default=0.005, type=float, help='min lr')
    parser.add_argument('--weight_decay', type=float, default=2e-4, help='decay rate')
    parser.add_argument('--seed', type=int, help='random seed')

    return parser.parse_args()


def get_yaml(path):
    rf = open(file=path, mode='r', encoding='utf-8')
    crf = rf.read()
    rf.close()
    yaml_data = yaml.load(stream=crf, Loader=yaml.FullLoader)
    return yaml_data


def train(net, trainloader, optimizer, criterion, device):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    train_pred = []
    train_true = []
    time_cost = datetime.datetime.now()
    for batch_idx, ((data, normals), label) in enumerate(trainloader):
        data, normals, label = data.to(device), normals.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)  # so, the input data shape is [batch, 3, 4096]
        normals = normals.permute(0, 2, 1)  # so, the input data shape is [batch, 3, 4096]
        optimizer.zero_grad()
        logits = net(data, normals)
        loss = criterion(logits, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()
        train_loss += loss.item()
        preds = logits.max(dim=1)[1]

        train_true.append(label.cpu().numpy())
        train_pred.append(preds.detach().cpu().numpy())

        total += label.size(0)
        correct += preds.eq(label).sum().item()

    #     progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
    #                  % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    #
    # time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    # train_true = np.concatenate(train_true)
    # train_pred = np.concatenate(train_pred)
    # return {
    #     "loss": float("%.3f" % (train_loss / (batch_idx + 1))),
    #     "acc": float("%.3f" % (100. * metrics.accuracy_score(train_true, train_pred))),
    #     "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(train_true, train_pred))),
    #     "time": time_cost
    # }


def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


def main():
    args = parse_args()
    config = get_yaml(args.config)
    train_loader = DataLoader(TeethPointCloudData(config.get('data_path', None),sample_groups=config.get('sample_groups', None)),
                              num_workers=config.get('num_workers', None),
                              batch_size=config.get('batch_size', None), shuffle=True, drop_last=True)
    net = models.__dict__[config.get('model', None)]()
    device = 'cuda'
    net = net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    # scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=args.min_lr, last_epoch=start_epoch - 1)
    criterion = cal_loss
    train_out = train(net, train_loader, optimizer, criterion, device)
    print(1)


if __name__ == '__main__':
    main()
