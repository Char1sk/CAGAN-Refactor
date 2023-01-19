import torch
import torch.nn as nn
import torchvision.models as models


class MyGeneratorDecoder(nn.Module):
    def __init__(self, out_channels, norm_layer=nn.BatchNorm2d):
        super(MyGeneratorDecoder, self).__init__()
        
        # self.relu9  = nn.ReLU(True)
        self.relu9  = nn.LeakyReLU(0.2, True)
        self.conv9  = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.norm9  = norm_layer(512)
        
        # self.relu10 = nn.ReLU(True)
        self.relu10 = nn.LeakyReLU(0.2, True)
        self.conv10 = nn.ConvTranspose2d(1536, 512, kernel_size=4, stride=2, padding=1)
        self.norm10 = norm_layer(512)
        
        # self.relu11 = nn.ReLU(True)
        self.relu11 = nn.LeakyReLU(0.2, True)
        self.conv11 = nn.ConvTranspose2d(1536, 512, kernel_size=4, stride=2, padding=1)
        self.norm11 = norm_layer(512)
        
        # self.relu12 = nn.ReLU(True)
        self.relu12 = nn.LeakyReLU(0.2, True)
        self.conv12 = nn.ConvTranspose2d(1536, 512, kernel_size=4, stride=2, padding=1)
        self.norm12 = norm_layer(512)
        
        # self.relu13 = nn.ReLU(True)
        self.relu13 = nn.LeakyReLU(0.2, True)
        self.conv13 = nn.ConvTranspose2d(1536, 256, kernel_size=4, stride=2, padding=1)
        self.norm13 = norm_layer(256)
        
        # self.relu14 = nn.ReLU(True)
        self.relu14 = nn.LeakyReLU(0.2, True)
        self.conv14 = nn.ConvTranspose2d(768,  128, kernel_size=4, stride=2, padding=1)
        self.norm14 = norm_layer(128)
        
        # self.relu15 = nn.ReLU(True)
        self.relu15 = nn.LeakyReLU(0.2, True)
        self.conv15 = nn.ConvTranspose2d(384,  64,  kernel_size=4, stride=2, padding=1)
        self.norm15 = norm_layer(64)
        
        # self.relu16 = nn.ReLU(True)
        self.relu16 = nn.LeakyReLU(0.2, True)
        self.conv16 = nn.ConvTranspose2d(192, out_channels, kernel_size=4, stride=2, padding=1)
        self.tanh16 = nn.Tanh()
        
    
    
    def forward(self, app, com):
        
        x = torch.cat([app[7], com[7]], 1)
        
        x = self.relu9(x)
        x = self.conv9(x)
        x = self.norm9(x)
        x = torch.cat([x, app[6], com[6]], 1)
        
        x = self.relu10(x)
        x = self.conv10(x)
        x = self.norm10(x)
        x = torch.cat([x, app[5], com[5]], 1)
        
        x = self.relu11(x)
        x = self.conv11(x)
        x = self.norm11(x)
        x = torch.cat([x, app[4], com[4]], 1)
        
        x = self.relu12(x)
        x = self.conv12(x)
        x = self.norm12(x)
        x = torch.cat([x, app[3], com[3]], 1)
        
        x = self.relu13(x)
        x = self.conv13(x)
        x = self.norm13(x)
        x = torch.cat([x, app[2], com[2]], 1)
        
        x = self.relu14(x)
        x = self.conv14(x)
        x = self.norm14(x)
        x = torch.cat([x, app[1], com[1]], 1)
        
        x = self.relu15(x)
        x = self.conv15(x)
        x = self.norm15(x)
        x = torch.cat([x, app[0], com[0]], 1)
        
        x = self.relu16(x)
        x = self.conv16(x)
        x = self.tanh16(x)
        
        x = x * 255
        
        return x
