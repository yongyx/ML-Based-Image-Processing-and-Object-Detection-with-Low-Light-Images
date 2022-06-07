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
from loss_functions import RED_loss
import torchvision.utils as vutils
from models.ImgEnhanceNet import ImgEnhanceNet
from models.REDNet import REDNet

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

opt.trainRoot += '_rednet'
opt.modelRoot += '_rednet'
opt.evalRoot += '_rednet'

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

icenet = ImgEnhanceNet()
icenet.load_state_dict(torch.load("checkpoint3_icenet/icenet_150.pth"))
rednet = REDNet()
optimizer = optim.Adam(rednet.parameters(), lr=opt.initLR)

#sent everything to cuda
icenet = icenet.to(device)
rednet = rednet.to(device)

lossArr = []
iteration = 0

for epoch in range(0, opt.numEpochs):
    # trainingLog = open('{0}/trainingLog_{1}.txt'.format(opt.trainRoot, epoch), 'w')
    for i, trainBatch in enumerate(train_loader):
        iteration += 1

        icenet.train()
        optimizer.zero_grad()
        S = trainBatch['im']
        S = S.to(device)

        #get icenet reflectance
        with torch.no_grad():
            icenet.eval()
            R_old, _ = icenet(S)
        
        #get concatenated input for rednet
        R_old_max, _ = torch.max(R_old, dim=1, keepdims=True)
        R_old_max = R_old_max.to(device)
        S_new = torch.cat([R_old_max, S], dim=1)
        S_new = S_new.to(device)

        R, I = rednet(S_new)
        R = R.to(device)
        I = I.to(device)

        redloss = RED_loss(S, R, I, R_old)
        redloss.backward()
        optimizer.step()

        lossArr.append(redloss.cpu().data.item())
        meanLoss = np.mean(np.array(lossArr[:] ) )

        # trainingLog.write('Epoch %d iteration %d: Loss %.5f Accumulated Loss %.5f \n' \
        #         % ( epoch, iteration, lossArr[-1], meanLoss))

        if iteration % 500 == 0:
            vutils.save_image(S.data , '%s/images_%d.png' % (opt.trainRoot, iteration ), padding=0, normalize = True)
            vutils.save_image(R.data , '%s/reflectance_%d.png' % (opt.trainRoot, iteration ), padding=0, normalize = True)
            vutils.save_image(I.data , '%s/illumination_%d.png' % (opt.trainRoot, iteration ), padding=0, normalize = True)

        if iteration == opt.iterationEnd:
            np.save('%s/loss_%d.npy' % (opt.trainRoot, epoch+1), lossArr)
            torch.save(rednet.state_dict(), '%s/rednet_%d.pth' % (opt.modelRoot, epoch+1))
            break
        
    print('Epoch %d iteration %d: Loss %.5f Accumulated Loss %.5f'  \
                % ( epoch, iteration, lossArr[-1], meanLoss) )
    # trainingLog.close()

    if iteration >= opt.iterationEnd:
        break

    if (epoch+1) % 10 == 0:
        # Save the accuracy
        np.save('%s/loss_%d.npy' % (opt.trainRoot, epoch+1), lossArr)
        torch.save(rednet.state_dict(), '%s/rednet_%d.pth' % (opt.modelRoot, epoch+1))
    
    if (epoch+1) % 20 == 0:
        with torch.no_grad():
            eval_loss = []
            for j, evalBatch in enumerate(val_loader):
                rednet.eval()
                icenet.eval()
                S_eval = evalBatch['im']
                S_eval = S_eval.to(device)

                R_eval_old, _ = icenet(S_eval)

                #get concatenated input for rednet
                R_eval_old_max, _ = torch.max(R_eval_old, dim=1, keepdims=True)
                R_eval_old_max = R_eval_old_max.to(device)
                S_eval_new = torch.cat([R_eval_old_max, S_eval], dim=1)
                S_eval_new = S_eval_new.to(device)

                R_eval, I_eval = rednet(S_eval_new)
                R_eval = R_eval.to(device)
                I_eval = I_eval.to(device)

                redloss_eval = RED_loss(S_eval, R_eval, I_eval, R_eval_old)
                eval_loss.append(redloss_eval)

                if j % 50 == 0:
                    np.save('%s/loss_%d.npy' % (opt.evalRoot, epoch+1), eval_loss)
                    vutils.save_image(S_eval.data , '%s/images_%d.png' % (opt.evalRoot, epoch ), padding=0, normalize = True)
                    vutils.save_image(R_eval.data , '%s/reflectance_%d.png' % (opt.evalRoot, epoch ), padding=0, normalize = True)
                    vutils.save_image(I_eval.data , '%s/illumination_%d.png' % (opt.evalRoot, epoch ), padding=0, normalize = True)




