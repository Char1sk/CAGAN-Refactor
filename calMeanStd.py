import torch
from torch.utils.data.dataloader import DataLoader
import torchvision

from options.train_options import TrainOptions
from data.dataset import MyDataset


if __name__ == '__main__':
    opt = TrainOptions().parse()
    trainSet = MyDataset(opt, True)
    trainLoader = DataLoader(dataset=trainSet, batch_size=1, shuffle=False)
    
    mean = 0.0
    std = 0.0
    for inputs, _, _ in trainLoader:
        mean += inputs.mean()
        std += inputs.std()
    mean /= len(trainLoader)
    std /= len(trainLoader)
    print(mean, std)
    # tensor(0.6022) tensor(0.4003)








