import torch
import torch.nn as nn
import torch.nn.functional as F

class ImgEnhanceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.channels = [4,32,64,96,128]
        self.relu = nn.ReLU()
        self.conv0 = nn.Conv2d(3, self.channels[1], (3,3), stride=1, padding=1)
        self.conv = nn.Conv2d(3, self.channels[2], (9,9), stride=1, padding=4)
        self.conv1 = nn.Conv2d(self.channels[2], self.channels[2], (3,3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.channels[2], self.channels[4], (3,3), stride=2, padding=1) #downsampling
        self.conv3 = nn.Conv2d(self.channels[4], self.channels[4], (3,3), stride=1, padding=1)
        self.conv4 = nn.ConvTranspose2d(self.channels[4], self.channels[2], (3,3), stride=2, padding=1) #upsampling
        #conv5 is concatenation of conv4 and conv1
        self.conv6 = nn.Conv2d(self.channels[4], self.channels[2], (3,3), stride=1, padding=1)
        #conv7 is concatenation of conv6 and conv0
        self.conv8 = nn.Conv2d(self.channels[3], self.channels[2], (3,3), stride=1, padding=1)
        self.conv9 = nn.Conv2d(self.channels[2], self.channels[0], (3,3), stride=1, padding=1)
        self.RnI = nn.Sigmoid() #reflectance and illumination


    def forward(self, x):
        x0 = self.conv0(x) #conv0 saved for concat later
        x0 = self.relu(x0)
        x = self.conv(x)
        x = self.conv1(x)
        x = self.relu(x)
        x1 = x #conv1 saved for concat later
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x, output_size=x1.size())
        x = self.relu(x)
        x = torch.cat((x, x1), axis=1) #concat conv4 and conv1
                                       #should have 128 channels
        x = self.conv6(x)
        x = self.relu(x)
        x = torch.cat((x, x0), axis=1) #concat conv6 and conv0
                                       #should have 96 channels
        x = self.conv8(x)
        x = self.conv9(x)
        R = self.RnI(x[:,0:3])
        L = self.RnI(x[:,3:4])

        return R, L
