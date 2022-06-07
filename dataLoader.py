import torch
import cv2
import os
import glob
import random
import numpy as np
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

class BatchLoader(Dataset):
    def __init__(self, datasetRoot, imgHeight=None, imgWidth=None, shuffle=False):
        super().__init__()

        self.path = datasetRoot #dataset root path
        self.imgHeight = imgHeight #img crop height
        self.imgWidth = imgWidth #img width height
        self.imgNames = []

        classList = glob.glob(self.path + "/*")
        for class_path in classList: #for each class
            for img_path in glob.glob(class_path + "/*"):
                self.imgNames.append(img_path)
        
        self.count = len(self.imgNames)
        self.perm = list(range(self.count))
        
        if shuffle == True:
            random.shuffle(self.perm)

        self.itercount = 0

    def __len__(self):
        return self.count

    def __getitem__(self, idx):

        imName = self.imgNames[self.perm[idx]]
        imPath = '/' + imName.split("/")[-2] + '/' + imName.split("/")[-1]
        im = self.loadImage(imName)
        imLabel = imName.split("/")[-2]

        if not (self.imgHeight is None or self.imgWidth is None):
            rows, cols = im.shape[1], im.shape[2]
            gapH = (rows - self.imgHeight )
            gapW = (cols - self.imgWidth )
            rs = int(np.round(np.random.random() * gapH ) )
            cs = int(np.round(np.random.random() * gapW ) )

            im = im[:, rs:rs+self.imgHeight, cs:cs+self.imgWidth]
        
        imgDict = {
            "im" : im,
            "im_path" : imPath,
            "label" : imLabel
        }

        return imgDict


    def loadImage(self, imName):
        im = Image.open(imName)
        im = np.asarray(im)

        rows, cols = im.shape[0], im.shape[1]
        if not (self.imgHeight is None or self.imgWidth is None ):
            if rows < self.imgHeight or cols < self.imgWidth:
                scaleRow = float(rows ) / float(self.imgHeight )
                scaleCol = float(cols ) / float(self.imgWidth )
                if scaleRow > scaleCol:
                    cols = int(np.ceil(cols / scaleCol ) )
                    rows = int(np.ceil(rows / scaleCol ) )
                else:
                    cols = int(np.ceil(cols / scaleRow ) )
                    rows = int(np.ceil(rows / scaleRow ) )
                im = cv2.resize(im, (cols, rows), interpolation=cv2.INTER_LINEAR)
        
        if len(im.shape) == 2:
            # print('Warning: load a gray image')
            im = im[:, :, np.newaxis]
            im = np.concatenate([im, im, im], axis=2)
        elif len(im.shape) > 2 and im.shape[2] == 4:
            # print("Warning: load a RGBA image")
            im = im[:,:,:3]

        im = im.astype(np.float32 ) / 255.0
        im = im.transpose([2, 0, 1] )
        return im


