import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()
        self.loss_f = nn.CosineEmbeddingLoss()

    def forward(self, pred, gt):
        y = torch.ones(pred.size()[0]).to(pred.device)
        loss = self.loss_f(pred, gt, y)
        return loss


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.loss_f = nn.CrossEntropyLoss()

    def forward(self, pred, gt):
        loss = self.loss_f(pred, gt)
        return loss


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
        self.loss_f = nn.L1Loss()

    def forward(self, pred, gt):
        norm = torch.norm(pred, dim=1)
        loss = self.loss_f(pred / norm[:, None], gt)
        # torch.mean(torch.mean(torch.abs(pred_norm - gt), dim=1))
        return loss


def cal_loss(pred, gt):
    ''' Calculate cosine similarity loss '''
    # B = pred.size(0)
    # dot = torch.sum(torch.mul(pred,gt),dim=1)
    # norm = torch.norm(pred,dim=1)
    # sim = torch.sum((dot / norm))
    similarity = F.cosine_similarity(pred, gt, dim=1)

    loss = torch.sum(1 - similarity)

    return loss


def cal_loss_hinge(pred, gt):
    ''' Calculate cosine similarity loss '''
    # B = pred.size(0)
    # dot = torch.sum(torch.mul(pred,gt),dim=1)
    # norm = torch.norm(pred,dim=1)
    # sim = torch.sum((dot / norm))
    # hinge_loss = F.cosine_embedding_loss()
    # loss = hinge_loss(pred,gt)

    loss_f = nn.CosineEmbeddingLoss()
    y = torch.ones(pred.size()[0]).to(pred.device)
    loss = loss_f(pred, gt, y)
    return loss


def cal_total_loss(pred, gt, use_L1=False, weights_list=None):
    if weights_list is None:
        weights_list = [0.2, 0.8]
    if use_L1:
        l1 = L1Loss()
        l1_loss = weights_list[0] * l1(pred, gt)
        cs = CosineSimilarityLoss()
        cs_loss = weights_list[1] * cs(pred, gt)
        return l1_loss + cs_loss, l1_loss, cs_loss
    else:
        l1_loss = torch.tensor(0.0)
        cs = CosineSimilarityLoss()
        cs_loss = cs(pred, gt)
        return cs_loss, l1_loss, cs_loss
