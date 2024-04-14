import argparse
import datetime

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from fvcore.common.config import CfgNode
from torch.utils.data import DataLoader
from tqdm import tqdm

import models as models
from data import TeethPointCloudData
from models.loss import cal_total_loss
from utils.utils import LogWriter, save_checkpoint, get_total_params, get_total_trainable_params, cal_cossim


def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('training')
    parser.add_argument('-cg', '--config', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)', default='configs/test.yaml')
    options = parser.parse_args()
    opts = CfgNode(CfgNode.load_yaml_with_base('configs/base.yaml'))
    opts.merge_from_file(options.config)
    return opts


def train(net, trainloader, optimizer, criterion, device, args):
    net.train()
    print('number of model parameters is {}'.format(get_total_params(net)))
    print('number of model trainable parameters is {}'.format(get_total_trainable_params(net)))
    train_loss = 0
    l1_total = 0
    cs_total = 0
    correct = 0
    total = 0
    train_pred = []
    train_true = []
    time_cost = datetime.datetime.now()
    for batch_idx, ((data, normals), label) in enumerate(tqdm(trainloader)):
        data, normals, label = data.to(device), normals.to(device), label.to(device).squeeze()
        # data = data.permute(0, 2, 1)  # so, the input data shape is [batch, 3, 4096]
        # normals = normals.permute(0, 2, 1)  # so, the input data shape is [batch, 3, 4096]
        optimizer.zero_grad()
        logits = net(data, normals)
        total_loss, l1_loss, cs_loss = criterion(logits, label, args.get('use_L1', None), args.get('loss_weights_list', None))
        total_loss.backward()
        # torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()
        train_loss += total_loss.item()
        l1_total += l1_loss.item()
        cs_total += cs_loss.item()

        # if batch_idx % 10 == 0:
        #     print(
        #         'batch: {}/{}, total_loss: {:.4f}, l1_loss: {:.4f}, cs_loss: {:.4f}'.format(batch_idx, len(trainloader),
        #                                                                                     total_loss.item(),
        #                                                                                     l1_loss.item(),
        #                                                                                     cs_loss.item()))
        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return {
        "total_loss": float("%.3f" % (train_loss / (batch_idx + 1))),
        "l1_total": float("%.3f" % (l1_total / (batch_idx + 1))),
        "cs_total": float("%.3f" % (cs_total / (batch_idx + 1)))
    }
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


def validate(net, testloader, device):
    net.eval()
    avg_similarity = 0
    with torch.no_grad():
        for batch_idx, ((data, normals), label) in enumerate(testloader):
            data, normals, label = data.to(device), normals.to(device), label.to(device).squeeze()
            # data = data.permute(0, 2, 1)  # so, the input data shape is [batch, 3, 4096]
            # normals = normals.permute(0, 2, 1)
            logits = net(data, normals)
            similarity = cal_cossim(logits, label)
            avg_similarity += torch.mean(similarity).item()

    print('*' * 50)
    print("Average similarity is: {:.4f}".format(avg_similarity))
    print('*' * 50)


def main():
    args = parse_args()
    train_loader = DataLoader(TeethPointCloudData(args, partition='train'), num_workers=args.num_workers,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(TeethPointCloudData(args, partition='test'), num_workers=args.num_workers,
                             batch_size=args.batch_size, shuffle=False, drop_last=False)
    net = models.__dict__[args.get('model', 'pointMLPElite_hz')](args.sample_groups)
    device = 'cuda'
    net = net.to(device)
    # optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9)
    start_epoch = 0
    # scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=args.min_lr, last_epoch=start_epoch - 1)
    criterion = cal_total_loss
    writer = LogWriter(args.get('log_lath', './checkpoints'), args.model_name)
    for epoch in range(start_epoch, args.epoch):
        print('Epoch(%d/%s) Learning Rate %s:' % (epoch + 1, args.epoch, optimizer.param_groups[0]['lr']))
        loss_dict = train(net, train_loader, optimizer, criterion, device, args)
        if (epoch + 1) % 10 == 0:
            validate(net, test_loader, device)
        print(
            'Epoch {}/{}, train_total_loss: {:.4f}, l1_total_loss: {:.4f}, '
            'cs_total_loss: {:.4f}'.format(
                epoch + 1,
                args.epoch, loss_dict['total_loss'], loss_dict['l1_total'],
                loss_dict['cs_total'])
        )

        writer.scalar_summary(loss_dict, epoch)
        if (epoch + 1) % args.log_interval == 0:
            save_checkpoint(net, args.log_path, args.model_name, epoch, loss_dict['total_loss'])
        # scheduler.step()
    writer.close()


if __name__ == '__main__':
    main()
