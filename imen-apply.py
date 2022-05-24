import os
import glob
import cv2

from LIME_package.LIME_CLI import main
from enlighten_inference import EnlightenOnnxModel
from argparse import Namespace
from PIL import Image

def LIME(root):
  extensions = ['jpg', 'png', 'JPG']
  for fname in glob.iglob(root + '/**', recursive=True):
    if '__MACOSX' not in fname and fname[-3:] in extensions:
      print(fname)
      Image.open(fname).save(fname[:-4] + '.bmp')
      args = {
      'alpha':2,
      'filePath':fname[:-4] + '.bmp',
      'gamma':0.7,
      'iterations':10,
      'map':False,
      'output':fname[:-14],
      'rho':2,
      'strategy':2
      }
      ns = Namespace(**args)
      main(ns)
      Image.open(fname[:-4] + '.bmp').save(fname)
      os.remove(fname[:-4] + '.bmp')

def EnlightenGAN(root):
  model = EnlightenOnnxModel()
  extensions = ['jpg', 'png', 'JPG']
  for fname in glob.iglob(root + '/**', recursive=True):
    if '__MACOSX' not in fname and fname[-3:] in extensions:
      img = cv2.imread(fname)
      Image.fromarray(img).save(fname)