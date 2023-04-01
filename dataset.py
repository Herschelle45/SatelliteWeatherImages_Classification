from torch.utils.data import Dataset, DataLoader 
import numpy as np
import pandas as pd
import os
from PIL import Image

class SatelliteImagesDataset(Dataset):
    def __init__(self, rootdir, csvfile, transform=None):
        self.rootdir = rootdir 
        self.csvfile = pd.read_csv(csvfile)
        self.transform = transform
    def __len__(self):
        return len(self.csvfile)
    def __getitem__(self, index):
        if index == 0:
                  img = np.array(Image.open(os.path.join(self.rootdir, self.csvfile.iloc[index, 0]))) 
        else:
          img = np.array(Image.open(os.path.join(self.rootdir, self.csvfile.iloc[len(self.csvfile)%index, 0]))) 
        label = self.csvfile.iloc[index , 1]
        if self.transform:
            img = self.transform(img)
        img = img[0:3]
        return (img, label)
