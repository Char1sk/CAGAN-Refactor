import os
import torch

from options.train_options import TrainOptions
from models.Inference import InferenceModel

from ptflops import get_model_complexity_info


def main():
    opt = TrainOptions().parse()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    
    net = InferenceModel(opt, device)
    flops, params = get_model_complexity_info(net, (1+8,250,200))
    print("flops:", flops)
    print("params:", params)


if __name__ == '__main__':
    main()
