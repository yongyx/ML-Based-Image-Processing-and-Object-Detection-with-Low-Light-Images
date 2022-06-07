import os
from dataLoader import BatchLoader
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f

import numpy as np
import pandas as pd
import glob
import cv2
import scipy.io as io
import argparse
import matplotlib.pyplot as plt 
import torchvision.transforms as T
from torch.utils.data import DataLoader, sampler, random_split
from loss_functions import ICE_loss
import torchvision.utils as vutils
from models.ImgEnhanceNet import ImgEnhanceNet

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
# The locationi of training set
parser.add_argument('--datasetName', default='/ExDark', help='name of dataset' )
parser.add_argument('--trainRoot', default='train', help='the path to store sampled images and models' )
parser.add_argument('--evalRoot', default='eval', help='the path to store evaluated images')
parser.add_argument('--modelRoot', default='checkpoint', help='the path to store the testing results')
parser.add_argument('--numEpochs', type=int, default=200, help='the number of epochs being trained')
parser.add_argument('--batchSize', type=int, default=16, help='the size of a batch')
parser.add_argument('--patchSize', type=int, default=48, help='the patch to crop')
parser.add_argument('--gpuId', type=int, default=0, help='gpu id used for training the network' )
parser.add_argument('--initLR', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--iterationEnd', type=int, default=50000, help='the iteration to end training')

# The detail network setting
opt = parser.parse_args()
print(opt)

#parameters
NUM_TRAIN = 3000 
NUM_VAL = 1800
NUM_TEST = 2563

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("training on", device)

opt.trainRoot += '_icenet'
opt.modelRoot += '_icenet'
opt.evalRoot += '_icenet'

# Save all the codes
os.system('mkdir %s' % opt.trainRoot)
os.system('mkdir %s' % opt.modelRoot)
os.system('mkdir %s' % opt.evalRoot)

datasetRoot = os.getcwd() + opt.datasetName
exDark = BatchLoader(datasetRoot, imgHeight=opt.patchSize, imgWidth=opt.patchSize)
exDark_train, exDark_val, exDark_test = random_split(exDark, [NUM_TRAIN, NUM_VAL, NUM_TEST], generator=torch.Generator().manual_seed(6))

train_loader = DataLoader(exDark_train, batch_size=opt.batchSize, num_workers=0,
                            sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))
val_loader = DataLoader(exDark_val, batch_size=opt.batchSize, num_workers=0,
                            sampler=sampler.SubsetRandomSampler(range(NUM_VAL)))
test_loader = DataLoader(exDark_test, batch_size=opt.batchSize, num_workers=0,
                            sampler=sampler.SubsetRandomSampler(range(NUM_TEST)))

model = ImgEnhanceNet()
optimizer = optim.Adam(model.parameters(), lr=opt.initLR)

#sent everything to cuda
model = model.to(device)

lossArr = []
iteration = 0

for epoch in range(0, opt.numEpochs):
    # trainingLog = open('{0}/trainingLog_{1}.txt'.format(opt.trainRoot, epoch), 'w')
    for i, trainBatch in enumerate(train_loader):
        iteration += 1

        model.train()
        optimizer.zero_grad()
        S = trainBatch['im']
        S = S.to(device)

        R, I = model(S)
        R = R.to(device)
        I = I.to(device)

        iceloss = ICE_loss(S, R, I)
        iceloss.backward()
        optimizer.step()

        lossArr.append(iceloss.cpu().data.item())
        meanLoss = np.mean(np.array(lossArr[:] ) )

        # trainingLog.write('Epoch %d iteration %d: Loss %.5f Accumulated Loss %.5f \n' \
        #         % ( epoch, iteration, lossArr[-1], meanLoss))

        if iteration % 500 == 0:
            vutils.save_image(S.data , '%s/images_%d.png' % (opt.trainRoot, iteration ), padding=0, normalize = True)
            vutils.save_image(R.data , '%s/reflectance_%d.png' % (opt.trainRoot, iteration ), padding=0, normalize = True)
            vutils.save_image(I.data , '%s/illumination_%d.png' % (opt.trainRoot, iteration ), padding=0, normalize = True)

        if iteration == opt.iterationEnd:
            np.save('%s/loss_%d.npy' % (opt.trainRoot, epoch+1), lossArr)
            torch.save(model.state_dict(), '%s/icenet_%d.pth' % (opt.modelRoot, epoch+1))
            break
        
    print('Epoch %d iteration %d: Loss %.5f Accumulated Loss %.5f'  \
                % ( epoch, iteration, lossArr[-1], meanLoss) )
    # trainingLog.close()

    if iteration >= opt.iterationEnd:
        break

    if (epoch+1) % 10 == 0:
        # Save the accuracy
        np.save('%s/loss_%d.npy' % (opt.trainRoot, epoch+1), lossArr)
        torch.save(model.state_dict(), '%s/icenet_%d.pth' % (opt.modelRoot, epoch+1))
    
    if (epoch+1) % 20 == 0:
        with torch.no_grad():
            eval_loss = []
            for j, evalBatch in enumerate(val_loader):
                model.eval()
                S_eval = evalBatch['im']
                S_eval = S_eval.to(device)

                R_eval, I_eval = model(S_eval)
                R_eval = R_eval.to(device)
                I_eval = I_eval.to(device)

                iceloss_eval = ICE_loss(S_eval, R_eval, I_eval)
                eval_loss.append(iceloss_eval)

                if j % 50 == 0:
                    vutils.save_image(S_eval.data , '%s/images_%d.png' % (opt.evalRoot, epoch+1 ), padding=0, normalize = True)
                    vutils.save_image(R_eval.data , '%s/reflectance_%d.png' % (opt.evalRoot, epoch+1 ), padding=0, normalize = True)
                    vutils.save_image(I_eval.data , '%s/illumination_%d.png' % (opt.evalRoot, epoch+1 ), padding=0, normalize = True)




