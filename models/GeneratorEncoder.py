import torch
import torch.nn as nn
import torchvision.models as models


class MyGeneratorEncoder(nn.Module):
    def __init__(self, in_channels, norm_layer=nn.BatchNorm2d):
        super(MyGeneratorEncoder, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)
        
        self.relu2 = nn.LeakyReLU(0.2, True)
        self.conv2 = nn.Conv2d( 64, 128, kernel_size=4, stride=2, padding=1)
        self.norm2 = norm_layer(128)
        
        self.relu3 = nn.LeakyReLU(0.2, True)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.norm3 = norm_layer(256)
                
        self.relu4 = nn.LeakyReLU(0.2, True)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.norm4 = norm_layer(512)
        
        self.relu5 = nn.LeakyReLU(0.2, True)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.norm5 = norm_layer(512)
        
        self.relu6 = nn.LeakyReLU(0.2, True)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.norm6 = norm_layer(512)
        
        self.relu7 = nn.LeakyReLU(0.2, True)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.norm7 = norm_layer(512)
        
        self.relu8 = nn.LeakyReLU(0.2, True)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
        
    
    def forward(self, x):
        
        x = self.conv1(x)
        temp1 = x
        
        x = self.relu2(x)
        x = self.conv2(x)
        x = self.norm2(x)
        temp2 = x
        
        x = self.relu3(x)
        x = self.conv3(x)
        x = self.norm3(x)
        temp3 = x
        
        x = self.relu4(x)
        x = self.conv4(x)
        x = self.norm4(x)
        temp4 = x
        
        x = self.relu5(x)
        x = self.conv5(x)
        x = self.norm5(x)
        temp5 = x
        
        x = self.relu6(x)
        x = self.conv6(x)
        x = self.norm6(x)
        temp6 = x
        
        x = self.relu7(x)
        x = self.conv7(x)
        x = self.norm7(x)
        temp7 = x
        
        x = self.relu8(x)
        x = self.conv8(x)
        temp8 = x
        
        return [temp1,temp2,temp3,temp4,temp5,temp6,temp7,temp8]
