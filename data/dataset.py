import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import cv2
import numpy as np
import scipy.io as sio

import random
import os

class MyDataset(data.Dataset):
    def __init__(self, opt, isTrain):
        super(MyDataset, self).__init__()
        
        self.opt = opt
        self.isTrain = isTrain
        # [ ( input, label, inputMat, labelMat ) ]
        if self.isTrain:
            self.imageNames = getNames(os.path.join(self.opt.data_folder, self.opt.train_list))
        else:
            self.imageNames = getNames(os.path.join(self.opt.data_folder, self.opt.test_list))
    
    def __getitem__(self, index):
        names = self.imageNames[index]
        
        inputPath = os.path.join(self.opt.data_folder, names[0])
        conptPath = os.path.join(self.opt.data_folder, names[2])
        labelPath = os.path.join(self.opt.data_folder, names[1])
        
        inputs = getInputs(inputPath, self.opt.output_shape, self.opt.pad)
        conpts = getConpts(conptPath, self.opt.output_shape, self.opt.pad)
        labels = getLabels(labelPath, self.opt.output_shape, self.opt.pad)
        
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


def getInputs(path, shape, pad):
    # img: H*W numpy
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # img: 1*SHAPE*SHAPE tensor [0.0, 1.0]
    shapes = getPadShape(img.shape, shape) if pad else 0
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Pad(shapes),
        transforms.Normalize(0.6022, 0.4003)
    ])(img)
    # tensor
    return img


def getConpts(path, shape, pad):
    # get mat, 250*200*n
    mats = sio.loadmat(path)['res_label']
    mats = torch.from_numpy(mats)
    mats = mats.permute(2, 0, 1)
    shapes = getPadShape(mats.shape, shape) if pad else 0
    mats = transforms.Pad(shapes)(mats)
    
    # choose 8 channels
    l0 = mats[0, :, :]
    l1 = mats[1, :, :]
    l2 = mats[7, :, :] + mats[6, :, :]
    l2 = torch.where(l2 > 1, torch.tensor(1,dtype=torch.float32), l2)
    l3 = mats[5, :, :] + mats[4, :, :]
    l3 = torch.where(l3 > 1, torch.tensor(1,dtype=torch.float32), l3)
    l4 = mats[2, :, :]
    l5 = mats[11, :, :] + mats[12, :, :]
    l5 = torch.where(l5 > 1, torch.tensor(1,dtype=torch.float32), l5)
    l6 = mats[10, :, :]
    l7 = mats[13, :, :]
    
    # merge channels
    mat = torch.stack([l0,l1,l2,l3,l4,l5,l6,l7], dim=0)
    return mat


def getLabels(path, shape, pad):
    # img: H*W*3 (BGR) numpy
    img = cv2.imread(path)
    # img: H*W*3 (RGB) numpy
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img: 3*SHAPE*SHAPE (RGB) tensor [0.0, 1.0]
    shapes = getPadShape(img.shape, shape) if pad else 0
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Pad(shapes),
    ])(img)
    # tensor
    return img


def zeroPadding(img, tsize):
    zeros = np.zeros((img.shape[0], tsize, tsize), dtype=np.float32)
    padT = (tsize - img.shape[1]) // 2
    padL = (tsize - img.shape[2]) // 2
    zeros[:, padT:tsize-padT, padL:tsize-padL] = img
    return zeros


def getPadShape(shape, tshape):
    padHt = (tshape - shape[1]) // 2
    padHd = tshape - shape[1] - padHt
    padWl = (tshape - shape[2]) // 2
    padWr = tshape - shape[2] - padWl
    return (padWl, padHt, padWr, padHd)
