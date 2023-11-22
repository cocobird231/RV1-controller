import numpy as np
from enum import Enum
from operator import itemgetter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils import MyLogger
from dataloader import RVDataset, RVDatasetType, GlobalKNNDataset
from criterions import RVNetCriterion, RVNetCriterionProp, RVLossType

class RVNet(nn.Module):
    def __init__(self, loaderType : RVDatasetType):
        super(RVNet, self).__init__()
        if (loaderType == RVDatasetType.GLOBAL):
            self.conv1 = torch.nn.Conv1d(6, 64, 1)
        elif (loaderType == RVDatasetType.GLOBAL_SIMP):
            self.conv1 = torch.nn.Conv1d(2, 64, 1)
        else:
            raise 'loaderType not support'

        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 512, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(512)

        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

        self.dropout = nn.Dropout(p=0.3)

        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        # x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = x.view(-1, 1, 2)
        return x

'''
KNN Method
'''

class KNNPredict():
    def __init__(self, measureSet, k : int = 3, r : float = 0.000015):
        self.k = k
        self.r = r

        # { Tensor([lat, lon], dtype=torch.float) : Tensor([steer, thr], dtype=torch.float) }
        # { Tensor([lat, lon], dtype=torch.float) : Tensor([orientx, orienty, orientz, orientw, steer, thr], dtype=torch.float) }
        self.measureSet = measureSet
    
    def getKNNFeat(self, pos : torch.Tensor):
        b, cf = pos.shape# (batch, coord + feat)
        # coord: (c, b)
        # feat: (f, b)
        coord, feat = torch.split(pos.transpose(1, 0), [2, cf - 2])
        dists = []

        nposIndicator = {}# { i : npos }
        for i, npos in enumerate(self.measureSet):
            nposIndicator[i] = npos
            npos = npos.view(-1, 1)
            nposBatch = npos
            for _i in range(b - 1) : nposBatch = torch.cat((nposBatch, npos), 1)
            dists.append(torch.norm(coord - nposBatch, p=2, dim=0).view(1, -1))
        dists, idcs = torch.sort(torch.cat(dists, dim=0), 0)
        idcs = idcs[:self.k]
        idcs = idcs.transpose(1, 0)

        ret = []
        for i in range(b):# Batch
            _eachPos = []
            for _i in idcs[i]:# KNN
                _eachPos.append(torch.cat((nposIndicator[_i.item()], self.measureSet[nposIndicator[_i.item()]])))
            _eachPos = torch.cat(_eachPos)
            ret.append(_eachPos.view(1, -1))
        ret = torch.cat((pos, torch.cat(ret, dim=0)), dim=1)
        return ret

    def predict(self, pos : torch.Tensor):
        nList = []# [[d, npos],...]
        for npos in self.measureSet:
            d = torch.norm(pos - npos, p=2)
            if (d < self.r):
                nList.append([d, npos])
        nList = sorted(nList, key=itemgetter(0))[:self.k if (len(nList) > self.k) else len(nList)]
        out = torch.tensor([0, 0], dtype=torch.float)
        for d, npos in nList:
            w = (self.r - d) / self.r
            out += w * self.measureSet[npos]
        out /= len(nList)
        return out


class LocalRVNet(nn.Module):
    def __init__(self, knnPred : KNNPredict, loaderType : RVDatasetType):
        super(LocalRVNet, self).__init__()

        self.knnPred = knnPred
        if (loaderType == RVDatasetType.GLOBAL):
            self.fc1 = nn.Linear(6 + self.knnPred.k * 8, 128)# intput: x, y, ox, oy, oz, ow; knn: xn, yn, oxn, oyn, ozn, own, sn, tn
        elif (loaderType == RVDatasetType.GLOBAL_SIMP):
            self.fc1 = nn.Linear(2 + self.knnPred.k * 4, 128)# intput: x, y; knn: xn, yn, sn, tn
        else:
            raise 'loaderType not support'

        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 2)

        self.dropout = nn.Dropout(p=0.5)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(64)

    def forward(self, x):# (batch, 6)
        x = self.knnPred.getKNNFeat(x)# (batch, 6 + k * 8)
        knnFeat = x.contiguous()
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.dropout(self.fc5(x))))
        x = self.fc6(x)
        return x, knnFeat

# Net predict knn weight to calculate steer and thr
class LocalWeightRVNet(nn.Module):
    def __init__(self, knnPred : KNNPredict, loaderType : RVDatasetType):
        super(LocalRVNet, self).__init__()

        self.knnPred = knnPred
        if (loaderType == RVDatasetType.GLOBAL):
            self.fc1 = nn.Linear(6 + self.knnPred.k * 8, 128)# intput: x, y, ox, oy, oz, ow; knn: xn, yn, oxn, oyn, ozn, own, sn, tn
        elif (loaderType == RVDatasetType.GLOBAL_SIMP):
            self.fc1 = nn.Linear(2 + self.knnPred.k * 4, 128)# intput: x, y; knn: xn, yn, sn, tn
        else:
            raise 'loaderType not support'

        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, self.knnPred.k)

        self.dropout = nn.Dropout(p=0.5)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(64)

    def forward(self, x):# (batch, 6)
        x = self.knnPred.getKNNFeat(x)# (batch, 6 + k * 8)
        knnFeat = x.contiguous()
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.dropout(self.fc5(x))))
        x = self.fc6(x)
        return x, knnFeat


