import torch
import torch.nn as nn
import torch.nn.functional as F

class ImgEnhanceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.channels = [4,32,64,96,128]
        self.relu = nn.ReLU()
        self.conv0 = nn.Conv2d(4, self.channels[1], (3,3), stride=1, padding="same")
        self.conv = nn.Conv2d(4, self.channels[2], (9,9), stride=1, padding="same")
        self.conv1 = nn.Conv2d(self.channels[2], self.channels[2], (3,3), stride=1, padding="same")
        self.conv2 = nn.Conv2d(self.channels[2], self.channels[4], (3,3), stride=2, padding=1) #downsampling
        self.conv3 = nn.Conv2d(self.channels[4], self.channels[4], (3,3), stride=1, padding="same")
        # self.conv4 = nn.ConvTranspose2d(self.channels[4], self.channels[2], (3,3), stride=2, padding=1, output_padding=1) #upsampling
        self.conv4 = nn.Conv2d(self.channels[4], self.channels[2], (3,3), stride=1, padding="same")
        #conv5 is concatenation of conv4 and conv1
        self.conv6 = nn.Conv2d(self.channels[4], self.channels[2], (3,3), stride=1, padding="same")
        #conv7 is concatenation of conv6 and conv0
        self.conv8 = nn.Conv2d(self.channels[3], self.channels[2], (3,3), stride=1, padding="same")
        self.conv9 = nn.Conv2d(self.channels[2], self.channels[0], (3,3), stride=1, padding="same")
        self.RnI = nn.Sigmoid() #reflectance and illumination


    def forward(self, x):
        #concatenation
        x_max, _ = torch.max(x, dim=1, keepdims=True)
        x = torch.cat([x_max, x], dim=1)


        x0 = self.conv0(x) #conv0 saved for concat later
        x0 = self.relu(x0)
        # print("x0 size: ", x0.size())
        x = self.conv(x)
        # print("conv x size: ", x.size())
        x = self.conv1(x)
        x = self.relu(x)
        x1 = x #conv1 saved for concat later
        # print("x1 size: ", x1.size())
        x = self.conv2(x)
        x = self.relu(x)
        # print("conv2 x size: ", x.size())
        x = self.conv3(x)
        x = self.relu(x)
        # print("conv3 x size: ", x.size())

        _, _, h, w = x1.size()
        x = F.interpolate(x, [h,w], mode="bilinear", align_corners=True)
        x = self.conv4(x)
        x = self.relu(x)
        # print("conv4 x size: ", x.size())
        x = torch.cat((x, x1), axis=1) #concat conv4 and conv1
                                       #should have 128 channels
        x = self.conv6(x)
        x = self.relu(x)
        # print("conv6 x size: ", x.size())
        x = torch.cat((x, x0), axis=1) #concat conv6 and conv0
                                       #should have 96 channels
        x = self.conv8(x)
        # print("conv8 x size: ", x.size())
        x = self.conv9(x)
        # print("conv9 x size: ", x.size())
        R = self.RnI(x[:,0:3]) 
        I = self.RnI(x[:,3:4]) 

        return R, I
