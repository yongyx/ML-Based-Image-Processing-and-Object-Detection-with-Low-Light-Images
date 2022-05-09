import os
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f

import numpy as np
import pandas as pd
import glob
import cv2
import matplotlib.pyplot as plt 
import torchvision.transforms as T
from torch.utils.data import DataLoader, sampler, random_split
from custom_dataset import CustomDataset
from loss_functions import ICE_loss
from models.ImgEnhanceNet import ImgEnhanceNet

import warnings
warnings.filterwarnings("ignore")

#parameters
NUM_TRAIN = 3000
NUM_VAL = 1800
NUM_TEST = 2563
BATCH_SIZE = 16
epochs = 50
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    global NUM_TRAIN, NUM_VAL, NUM_TEST, BATCH_SIZE, epochs, device
    path = os.getcwd()
    os.chdir("./ExDark") #change directory

    img_transform = T.Compose([ T.ToPILImage(),
                            T.RandomCrop((48,48)),
                            T.ToTensor()])
    exDark = CustomDataset(os.getcwd(), transform=img_transform)
    exDark_train, exDark_val, exDark_test = random_split(exDark, [NUM_TRAIN, NUM_VAL, NUM_TEST])

    train_loader = DataLoader(exDark_train, batch_size=BATCH_SIZE, num_workers=0,
                            sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))
    val_loader = DataLoader(exDark_val, batch_size=BATCH_SIZE, num_workers=0,
                            sampler=sampler.SubsetRandomSampler(range(NUM_VAL)))
    test_loader = DataLoader(exDark_test, batch_size=BATCH_SIZE, num_workers=0,
                            sampler=sampler.SubsetRandomSampler(range(NUM_TEST)))
    
    train(train_loader, val_loader, path)

def train(train_loader, val_loader, path):
    
    model = ImgEnhanceNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print("Starting training: ")
    for e in range(epochs):
        print("Epoch {}".format(e))
        for i, (img, classname) in enumerate(train_loader):
            model.train()
            S = img.to(device=device, dtype=torch.float32)

            optimizer.zero_grad()

            R, I = model(S)
            iceloss = ICE_loss(S, R, I)

            iceloss.backward()

            optimizer.step()

            if i % 100 == 0:
                print('Iteration %d, loss = %.4f' % (i, iceloss.item()))
                eval(val_loader, model)
    
    os.mkdir(path + '/trained_models')
    os.chdir(path + '/trained_models')
    torch.save(model, 'ICENet_model.pth')
    
    print("Done training")

def eval(val_loader, model):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for i, (img, classname) in enumerate(val_loader):
            S = img.to(device=device, dtype=torch.float32)

            R, I = model(S)

            total_loss += ICE_loss(S,R,I).item()
        total_loss /= len(val_loader)
        print("Validation loss = %.4f" % (total_loss))


if __name__== '__main__':
    main()