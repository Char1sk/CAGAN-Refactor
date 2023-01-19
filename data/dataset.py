import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import cv2
import numpy as np
import scipy.io as sio

import random
import os

class MyDataset(data.Dataset):
    def __init__(self, conf, isTrain):
        super(MyDataset, self).__init__()
        
        self.conf = conf
        self.isTrain = isTrain
        # [ ( input, label, inputMat, labelMat ) ]
        if self.isTrain:
            self.imageNames = getNames(os.path.join(self.conf.dataFolder, self.conf.trainFile))
        else:
            self.imageNames = getNames(os.path.join(self.conf.dataFolder, self.conf.testFile))
    
    def __getitem__(self, index):
        names = self.imageNames[index]
        
        inputPath = os.path.join(self.conf.dataFolder, names[0])
        conptPath = os.path.join(self.conf.dataFolder, names[2])
        labelPath = os.path.join(self.conf.dataFolder, names[1])
        
        inputs = getInputs(inputPath, self.conf)
        conpts = getConpts(conptPath, self.conf)
        labels = getLabels(labelPath, self.conf)
        
        inputs = torch.from_numpy(inputs)
        conpts = torch.from_numpy(conpts)
        labels = torch.from_numpy(labels)
        
        return inputs, conpts, labels
        
    
    def __len__(self):
        return len(self.imageNames)



def getNames(path):
    ret = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            names = line.split('||')
            ret.append(names)
    return ret


def getInputs(path, conf):
    # img: H*W*1 numpy
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # img = np.expand_dims(img, axis=2)
    img = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(0.5, 0.5)
    ])(img)
    img = img.numpy()
    # img: 1*H*W numpy
    # img = np.transpose(img.numpy(), (2, 0, 1))
    # img: 1*256*256 numpy
    img = zeroPadding(img, conf.targetSize)
    # numpy
    return img


def getConpts(path, conf):
    # get mat, n*256*256
    mats = sio.loadmat(path)['res_label']
    mats = mats.astype(np.float32)
    mats = np.transpose(mats, (2, 0, 1))
    mats = zeroPadding(mats, conf.targetSize)
    
    # choose 8 channels
    l0 = mats[0, :, :]
    l1 = mats[1, :, :]
    l2 = mats[7, :, :] + mats[6, :, :]
    l2 = np.where(l2 > 1, 1, l2)
    l3 = mats[5, :, :] + mats[4, :, :]
    l3 = np.where(l3 > 1, 1, l3)
    l4 = mats[2, :, :]
    l5 = mats[11, :, :] + mats[12, :, :]
    l5 = np.where(l5 > 1, 1, l5)
    l6 = mats[10, :, :]
    l7 = mats[13, :, :]
    
    # merge channels
    mat = np.zeros((1, conf.targetSize, conf.targetSize), dtype=np.float32)
    mat = np.concatenate((mat, l0.reshape(1, conf.targetSize, conf.targetSize)), axis=0)
    mat = np.concatenate((mat, l1.reshape(1, conf.targetSize, conf.targetSize)), axis=0)
    mat = np.concatenate((mat, l2.reshape(1, conf.targetSize, conf.targetSize)), axis=0)
    mat = np.concatenate((mat, l3.reshape(1, conf.targetSize, conf.targetSize)), axis=0)
    mat = np.concatenate((mat, l4.reshape(1, conf.targetSize, conf.targetSize)), axis=0)
    mat = np.concatenate((mat, l5.reshape(1, conf.targetSize, conf.targetSize)), axis=0)
    mat = np.concatenate((mat, l6.reshape(1, conf.targetSize, conf.targetSize)), axis=0)
    mat = np.concatenate((mat, l7.reshape(1, conf.targetSize, conf.targetSize)), axis=0)
    
    mat = mat[1:, :, :]
    return mat


def getLabels(path, conf):
    # img: H*W*3 (BGR) numpy
    img = cv2.imread(path)
    # img: H*W*3 (RGB) numpy
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # img: 3*H*W (RGB) numpy
    img = np.transpose(img, (2, 0, 1))
    # img: 3*256*256 numpy
    img = zeroPadding(img, conf.targetSize)
    # numpy
    return img


def zeroPadding(img, tsize):
    zeros = np.zeros((img.shape[0], tsize, tsize), dtype=np.float32)
    padT = (tsize - img.shape[1]) // 2
    padL = (tsize - img.shape[2]) // 2
    zeros[:, padT:tsize-padT, padL:tsize-padL] = img
    return zeros
