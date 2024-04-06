import torch
import os
import yaml
from torch.utils.tensorboard import SummaryWriter


class LogWriter:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, loss_dict: dict, step):
        for k in loss_dict.keys():
            self.writer.add_scalar(k, loss_dict[k], step)

    def close(self):
        self.writer.close()

def save_checkpoint(state, path, epoch, total_loss):
    os.makedirs(path, exist_ok=True)
    torch.save(state, os.path.join(path, 'checkpoint'+'_{}_{:.2f}.pth').format(epoch, total_loss))


def get_yaml(path):
    rf = open(file=path, mode='r', encoding='utf-8')
    crf = rf.read()
    rf.close()
    yaml_data = yaml.load(stream=crf, Loader=yaml.FullLoader)
    return yaml_data