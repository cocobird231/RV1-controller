from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

class RVLossType(Enum):
    L1 = 0# L1 loss
    L2 = 1# MSE loss
    KNN = 2

class RVNetCriterionProp():
    steerW = 1.0
    thrW = 1.0
    lossType = RVLossType.L1
    netType = 'localrvnet'

def RVNetCriterion(feat, secondFeat, valid, cprop : RVNetCriterionProp):
    # secondFeat = [x, y, x1, y1, s1, t1,..., xk, yk, sk, tk]
    steerValid, thrValid = torch.split(valid.transpose(1, 0), [1, 1])

    if (cprop.netType == 'localrvnet'):
        steerPred, thrPred = torch.split(feat.transpose(1, 0), [1, 1])
        lossDict = dict()
        if (cprop.lossType == RVLossType.L1):
            lossDict['l1Loss'] = nn.L1Loss()(steerPred, steerValid) * cprop.steerW + nn.L1Loss()(thrPred, thrValid) * cprop.thrW
            return lossDict['l1Loss'], lossDict
        elif (cprop.lossType == RVLossType.L2):
            lossDict['l2Loss'] = nn.MSELoss()(steerPred, steerValid) * cprop.steerW + nn.MSELoss()(thrPred, thrValid) * cprop.thrW
            return lossDict['l2Loss'], lossDict
        elif (cprop.lossType == RVLossType.KNN):
            steerLoss = nn.L1Loss()(steerPred, steerValid) * cprop.steerW
            thrLoss = nn.L1Loss()(thrPred, thrValid) * cprop.thrW
            nSteerLoss
            return # TODO: neighbor loss
    elif (cprop.netType == 'localwrvnet'):
        nfeat = secondFeat.transpose(1, 0)