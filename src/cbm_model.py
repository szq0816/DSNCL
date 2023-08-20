import numpy as np
import numbers

import torch
import torch.nn.functional as F
from torch import nn


class SemanticCBM(nn.Module):
    def __init__(self, opt):
        super(SemanticCBM, self).__init__()
        self.opt = opt
        self.fc = nn.Linear(in_features=opt.wv_size, out_features=self.opt.num_hashing)

        self.cbm = None
        if opt.K != 0:
            self.cbm = CBM(K=opt.K, feat_dim=opt.num_hashing)
        self.cbm_start_iter = opt.cbm_start_iter

    def forward(self, x, wv, label, iter=0):

        sem_org = self.fc(wv)
        x_prev, label_prev = x, label

        if self.cbm is not None:
            self.cbm.enqueue_dequeue(x_prev.detach(), sem_org.detach(), label_prev.detach())

        if self.cbm is None:
            alpha = torch.rand(x.size(0), 1).to(sem_org.get_device())
            sem = alpha * sem_org + (1.0 - alpha) * x
        else:
            if iter < self.cbm_start_iter:
                alpha = torch.rand(x.size(0), 1).to(sem_org.get_device())
                sem = alpha * sem_org + (1.0 - alpha) * x
            else:
                x_cbm, sem_org_cbm, label_cbm = self.cbm.get()
                x = torch.cat((x, x_cbm), 0)
                sem_org = torch.cat((sem_org, sem_org_cbm), 0)
                label = torch.cat((label, label_cbm), 0)
                alpha = torch.rand(x.size(0), 1).to(sem_org.get_device())
                sem = alpha * sem_org + (1.0 - alpha) * x

        dists = self._pairwise_distance(sem.cpu(), x.cpu(), 'euclidean').cuda()
        same_identity_mask = torch.eq(label.unsqueeze(dim=1), label.unsqueeze(dim=0))
        positive_mask = torch.logical_xor(same_identity_mask[:, -x.shape[0]:],
                                          torch.eye(label.size(0), dtype=torch.bool).to(label.get_device()))

        furthest_positive, _ = torch.max(dists * (positive_mask.float()), dim=1)
        closest_negative, _ = torch.min(dists + 1e8 * (same_identity_mask.float()), dim=1)

        diff = furthest_positive - closest_negative
        diff = F.softplus(diff)

        return diff, sem

    def _pairwise_distance(self, x, sem, metric):
        diffs = x.unsqueeze(dim=1) - sem.unsqueeze(dim=0)
        if metric == 'sqeuclidean':
            return (diffs ** 2).sum(dim=-1)
        elif metric == 'euclidean':
            return torch.sqrt(((diffs ** 2) + 1e-16).sum(dim=-1))
        elif metric == 'cityblock':
            return diffs.abs().sum(dim=-1)


class CBM:
    def __init__(self, K, feat_dim):
        self.K = K
        self.feats = torch.zeros(self.K, feat_dim).cuda()
        self.sems = torch.zeros(self.K, feat_dim).cuda()
        self.targets = -1 * torch.ones(self.K, dtype=torch.long).cuda()
        self.ptr = 0

    @property
    def is_full(self):
        return self.targets[-1].item() != -1

    def get(self):
        if self.is_full:
            return self.feats, self.sems, self.targets
        else:
            return self.feats[:self.ptr], self.sems[:self.ptr], self.targets[:self.ptr]

    def enqueue_dequeue(self, feats, sems, targets):
        q_size = len(targets)
        if self.ptr + q_size > self.K:
            self.feats[-q_size:] = feats
            self.sems[-q_size:] = sems
            self.targets[-q_size:] = targets
            self.ptr = 0
        else:
            self.feats[self.ptr: self.ptr + q_size] = feats
            self.sems[self.ptr: self.ptr + q_size] = sems
            self.targets[self.ptr: self.ptr + q_size] = targets
            self.ptr += q_size
