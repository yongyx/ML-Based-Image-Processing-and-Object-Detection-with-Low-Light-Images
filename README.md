# ECE-228-Low-Light-Object-Detection

ExDark Dataset can be found here: https://github.com/cs-chan/Exclusively-Dark-Image-Dataset

# Image Enhancement With LIME/EnlightenGAN
LIME and EnlightenGAN were provided from libraries and the code to use them to enhance/save images can be found in ```imen-apply.py```

The EnlightenGAN model was installed from: https://github.com/arsenyinfo/EnlightenGAN-inference

The LIME model was made into a package and originally is from: https://github.com/aeinrw/LIME

# Image Enhancement with ICENet/REDNET

To run script to train ICENet, make sure that ExDar dataset is downloaded in the same directory as `train_IN.py`, then use the command `python train_IN.py`. Arguments to pass to the console for training are within `train_IN.py`.

# Object Detection
You will have to download the ExDark dataset (or enhanced images) into your own folder and move it into this repository folder.
Your folder path should be of the form:
```
faster rcnn.ipynb
  |
img_root
  |
  +--- class1
  |     | images
  +--- class2
  |     | images
```
There is a preprocessing cell in the faster rcnn notebooks and you may need to manually remove/edit files if it doesn't take care of it itself.

## Object Detection with pre-trained Faster R-CNN
To use pre-trained Faster R-CNN, make sure you have the dataset mentioned above and run ```faster rcnn-pretrained.ipynb```

## Object Detection with some extra training
To train the last few layers of Faster R-CNN on your data, run ```faster rcnn.ipynb```
