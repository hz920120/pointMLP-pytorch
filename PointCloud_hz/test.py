import argparse
import os
import datetime
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
from fvcore.common.config import CfgNode
import models as models
from data import TeethPointCloudData
from utils.utils import cal_cossim


def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('training')
    parser.add_argument('-cg', '--config', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)', default='configs/test_config_0331.yaml')
    options = parser.parse_args()
    opts = CfgNode(CfgNode.load_yaml_with_base('configs/base.yaml'))
    opts.merge_from_file(options.config)
    return opts


def validate(net, testloader, device):
    net.eval()
    avg_similarity = 0
    with torch.no_grad():
        for batch_idx, ((data, normals), label) in enumerate(testloader):
            data, normals, label = data.to(device), normals.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)  # so, the input data shape is [batch, 3, 4096]
            normals = normals.permute(0, 2, 1)
            logits = net(data, normals)
            similarity = cal_cossim(logits, label)
            avg_similarity += torch.mean(similarity).item()

    return avg_similarity


def main():
    args = parse_args()
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"==> Using device: {device}")

    print('==> Preparing data..')

    test_loader = DataLoader(
        TeethPointCloudData(args, partition='test'),
        num_workers=args.num_workers,
        batch_size=args.batch_size, shuffle=False, drop_last=False)
    # Model
    print('==> Building model..')
    net = models.__dict__[args.model]()
    net = net.to(device)
    # checkpoint_path = os.path.join(args.checkpoint, 'best_checkpoint.pth')
    checkpoint_path = "/hz/code/pointmlp/PointCloud_hz/checkpoints/20240407/checkpoint_499_0.55.pth"
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    # criterion = criterion.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    net.load_state_dict(checkpoint, strict=False)

    test_out = validate(net, test_loader, device)


if __name__ == '__main__':
    main()
