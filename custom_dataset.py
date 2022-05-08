import torch
import cv2
import glob
import numpy as np
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, path, transform=None):
        # super(CustomDataset, self).__init__()
        self.path = path #image path
        self.img_dim = (600,400)
        self.transform = transform
        self.data = []
        self.class_map = {}
        
        class_key = 0
        filelist = glob.glob(self.path + "/*")
 
        for class_path in filelist: #for each class
            class_name = class_path.split("/")[-1]
            self.class_map[class_name] = class_key
            for img_path in glob.glob(class_path + "/*"):
                self.data.append([img_path, class_name])
            class_key += 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_path, class_name = self.data[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_dim)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        class_key = self.class_map[class_name]
        class_key = torch.tensor([class_key])

        if self.transform != None:
            img = self.transform(img)

        return img, class_key

