# ECE-228-Pedestrian-Object-Detection

ExDark Dataset can be found here: https://github.com/cs-chan/Exclusively-Dark-Image-Dataset

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
## Object Detection with pre-trained Faster R-CNN
To use pre-trained Faster R-CNN, make sure you have the dataset mentioned above and run ```faster rcnn-pretrained.ipynb```

## Object Detection with some extra training
To train the last few layers of Faster R-CNN on your data, run ```faster rcnn.ipynb```
