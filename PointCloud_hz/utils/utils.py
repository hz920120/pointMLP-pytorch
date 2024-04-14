import torch
import os
import yaml
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


class LogWriter:
    def __init__(self, log_dir, model_name):
        dir = os.path.join(log_dir, model_name, 'logs')
        os.makedirs(dir, exist_ok=True)
        self.writer = SummaryWriter(dir)

    def scalar_summary(self, loss_dict: dict, step):
        for k in loss_dict.keys():
            self.writer.add_scalar(k, loss_dict[k], step)

    def close(self):
        self.writer.close()


def save_checkpoint(model, optimizer, args, epoch, total_loss):
    pth = {
        'epoch': epoch,
        'net': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'total_loss': total_loss,
        'args': args
    }
    save_path = os.path.join(args.log_path, args.model_name)
    os.makedirs(save_path, exist_ok=True)
    torch.save(pth, os.path.join(save_path, 'checkpoint' + '_{}_{:.2f}.pth').format(epoch, total_loss))


def get_yaml(path):
    rf = open(file=path, mode='r', encoding='utf-8')
    crf = rf.read()
    rf.close()
    yaml_data = yaml.load(stream=crf, Loader=yaml.FullLoader)
    return yaml_data


def get_total_params(model):
    return sum(p.numel() for p in model.parameters())


def get_total_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def cal_cossim(pred, gt):
    ''' Calculate cosine similarity '''
    norm = torch.norm(pred, dim=1)
    score = F.cosine_similarity(pred / norm[:, None], gt, dim=1)
    return score
