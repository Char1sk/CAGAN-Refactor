import time

import torch
from torch.utils.data.dataloader import DataLoader

from data.dataset import MyDataset
from options.train_options import TrainOptions
from models.GeneratorEncoder import MyGeneratorEncoder
from models.GeneratorDecoder import MyGeneratorDecoder


def main():
    # Option
    opt = TrainOptions().parse()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using {device} device')
    
    # Data
    testSet  = MyDataset(opt, False)
    testLoader  = DataLoader(dataset=testSet,  batch_size=1, shuffle=False)
    
    # Models
    GenAppE = MyGeneratorEncoder(in_channels = opt.input_nc).to(device)
    GenComE = MyGeneratorEncoder(in_channels = opt.conpt_nc).to(device)
    GenD = MyGeneratorDecoder(out_channels = opt.output_nc).to(device)
    
    # Inference
    times = 1
    beg = time.time()
    for i,data in enumerate(testLoader,1):
        inputs, conpts, _ = [d.to(device) for d in data]
        AppFeatures = GenAppE(inputs)
        ComFeatures = GenComE(conpts)
        preds = GenD(AppFeatures, ComFeatures)
        if i == times:
            break
    end = time.time()
    print((end-beg)/times)


if __name__ == '__main__':
    main()
