import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np

#from LIME_package.LIME_CLI import LIME
from enlighten_inference import EnlightenOnnxModel
from argparse import Namespace
from PIL import Image


def LIME_apply(root):
  extensions = ['jpg', 'png', 'JPG']
  for fname in glob.iglob(root + '/**', recursive=True):
    if '__MACOSX' not in fname and fname[-3:] in extensions and ('Bicycle' in fname or 'Boat' in fname):
      print(fname)
      
      check = np.asarray(Image.open(fname))
      if len(check.shape) < 3:
        print('NOT ENOUGH CHANNELS')
        os.remove(fname)
        continue
      
      Image.open(fname).save(fname[:-4] + '.bmp')
      args = {
      'alpha':2,
      'filePath':fname[:-4] + '.bmp',
      'gamma':0.7,
      'iterations':10,
      'map':False,
      'output':fname,
      'rho':2,
      'strategy':2
      }
      ns = Namespace(**args)
      lime_obj = LIME(**ns.__dict__)
      lime_obj.load(fname[:-4] + '.bmp')
      lime_obj.enhance()
      plt.imsave(fname, lime_obj.R)
      os.remove(fname[:-4] + '.bmp')


def EnlightenGAN_apply(root):
  model = EnlightenOnnxModel()
  extensions = ['jpg', 'png', 'JPG']
  for fname in glob.iglob(root + '/**', recursive=True):
    if '__MACOSX' not in fname and fname[-3:] in extensions:
      print(fname)
      img = cv2.imread(fname)
      img = model.predict(img)
      Image.fromarray(img).save(fname)
      
def BGR_2_RGB(root):
  extensions = ['jpg', 'png', 'JPG']
  for fname in glob.iglob(root + '/**', recursive=True):
    if '__MACOSX' not in fname and fname[-3:] in extensions:
      print(fname)
      img = cv2.imread(fname)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      cv2.imwrite(fname, img)

# BGR_2_RGB('./ExDarkEG')
