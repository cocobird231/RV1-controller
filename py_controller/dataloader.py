import os
import csv
import numpy as np
from enum import Enum

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class RVDatasetType(Enum):
    ALL = 0# input: 9, output: 3
    SIMP = 1# input: 2, output: 3
    GLOBAL = 2# input: 6, output: 2
    GLOBAL_SIMP = 3# input: 2, output: 2
    SERIES = 4

def jitter_pointcloud(pointcloud, sigma=0.0001):
    sz = pointcloud.shape[0]
    n = np.random.normal(0, sigma, 2)
    n = np.pad(n, (0, sz - 2), constant_values=(0, 0))
    pointcloud += n
    return pointcloud

#############################################################
#                       Training Dataset
#############################################################
# For normal: row = ["TS", "LAT", "LON", "ACCX", "ACCY", "GYROZ", "ORIENTX", "ORIENTY", "ORIENTZ", "ORIENTW", "STEER", "THR", "BRK"]
# For global: row = ["LAT", "LON", "ORIENTX", "ORIENTY", "ORIENTZ", "ORIENTW", "STEER", "THR"]
class RVDataset(Dataset):
    def __init__(self, csvPath : str, loaderType : RVDatasetType, isNormalized : bool = False, jitter : bool = False):
        self.inputData, self.outputData = self.load_data(csvPath, loaderType)
        self.loaderType = loaderType
        self.isNormalized = isNormalized
        self.jitterF = jitter
    
    def load_data(self, csvPath, loaderType):
        allInputData = []
        allOutputData = []
        with open(csvPath, 'r', encoding = 'utf-8', newline = '') as fp:
            csvReader= csv.reader(fp)
            for i, item in enumerate(csvReader):
                if (i == 0):
                    continue
                if ((loaderType == loaderType.ALL or loaderType == loaderType.SIMP) and len(item) != 13):
                    raise 'Dataset column size error'
                elif ((loaderType == loaderType.GLOBAL or loaderType == loaderType.GLOBAL_SIMP) and len(item) != 8):
                    raise 'Dataset column size error'

                if (loaderType == RVDatasetType.ALL):
                    allInputData.append(item[1:-3])# Input all wo ts
                    allOutputData.append(item[-3:])# Output steer, thr and brk.
                elif (loaderType == RVDatasetType.SIMP):
                    allInputData.append(item[1:3])# Just input GPS lat and lon wo ts.
                    allOutputData.append(item[-3:])# Output steer, thr and brk.
                elif (loaderType == RVDatasetType.GLOBAL):
                    allInputData.append(item[:-2])# Just input GPS lat and lon wo ts.
                    allOutputData.append(item[-2:])
                elif (loaderType == RVDatasetType.GLOBAL_SIMP):
                    allInputData.append(item[:2])# Just input GPS lat and lon wo ts.
                    allOutputData.append(item[-2:])
                else:
                    raise 'loaderType not support.'

        allInputData = np.asarray(allInputData)
        allOutputData = np.asarray(allOutputData)
        return allInputData, allOutputData
    
    def __len__(self):
        return self.inputData.shape[0]
    
    def __getitem__(self, item):
        sigma = 0.01 if (self.isNormalized) else 0.0001
        idata = jitter_pointcloud(self.inputData[item].astype('float32'), sigma=sigma) if (self.jitterF) else self.inputData[item].astype('float32')
        odata = jitter_pointcloud(self.outputData[item].astype('float32'), sigma=sigma) if (self.jitterF) else self.outputData[item].astype('float32')
        return idata, odata

class GlobalKNNDataset():
    def __init__(self, csvPath : str, loaderType : RVDatasetType):
        self.dataset = self.load_data(csvPath, loaderType)
    
    def load_data(self, csvPath, loaderType):
        outDict = {}# { Tensor([lat, lon], dtype=torch.float) : Tensor([steer, thr], dtype=torch.float) }
        with open(csvPath, 'r', encoding = 'utf-8', newline = '') as fp:
            csvReader= csv.reader(fp)
            for i, item in enumerate(csvReader):
                if (i == 0):
                    continue
                if (len(item) != 8):
                    raise 'Dataset column size error'
                if (loaderType == RVDatasetType.GLOBAL):
                    outDict[torch.tensor([float(item[0]), float(item[1])], dtype=torch.float)] = torch.tensor([float(item[i]) for i in range(2, 8)], dtype=torch.float)
                if (loaderType == RVDatasetType.GLOBAL_SIMP):
                    outDict[torch.tensor([float(item[0]), float(item[1])], dtype=torch.float)] = torch.tensor([float(item[6]), float(item[7])], dtype=torch.float)
        return outDict
    
    def getDataset(self):
        return self.dataset


#############################################################
if __name__ == '__main__':
    trainLoader = DataLoader(RVDataset(csvPath="/home/coco/Workspace/0412/datalog/dataset.csv"), batch_size=20, shuffle=True)
    cnt = 1
    for idata, odata in trainLoader:
        print("\n========idata========\n")
        print(type(idata), idata)
        print("\n========odata========\n")
        print(type(odata), odata)
        cnt += 1
        if (cnt > 5) : break