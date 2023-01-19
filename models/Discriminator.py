import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np


class MyPatchDiscriminator(nn.Module):
    def __init__(self, in_channels, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(MyPatchDiscriminator, self).__init__()
        
        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequence = [
            nn.Conv2d(in_channels, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                        kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                    kernel_size=kw, stride=1, padding=padw),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        
        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        
        self.model = nn.Sequential(*sequence)
    
    def forward(self, input):
        return self.model(input)
