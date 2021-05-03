import os

import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torch.nn.functional as F

from sklearn.model_selection import train_test_split

from PIL import Image
import cv2

from unet import UNet

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_PATH = 'dataset/compressed_img/'
MASK_PATH = 'dataset/compressed_mask/'

# create df with id of the dataset
def create_df(path):
    name = []
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            name.append(filename.split('.')[0])
    
    return pd.DataFrame({'id': name}, index = np.arange(0, len(name)))

df = create_df(IMAGE_PATH)
# print('Total Images: ', len(df))


#split the dataset into train, validation and test data
X_trainval, X_test = train_test_split(df['id'].values, test_size=0.1, random_state=0)
X_train, X_val = train_test_split(X_trainval, test_size=0.15, random_state=0)


# print('Train Size   : ', len(X_train)) #306
# print('Val Size     : ', len(X_val)) #54
# print('Test Size    : ', len(X_test)) #40

# img = cv2.imread(IMAGE_PATH + df['id'][0] + '.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# ## mask = Image.open(MASK_PATH + '0' + df['id'][100] + '.png')
# mask = cv2.imread(MASK_PATH + '0' + df['id'][0] + '.png', cv2.IMREAD_GRAYSCALE)
# print('Image Size', np.asarray(img).shape)
# print('Mask Size', np.asarray(mask).shape)
# mask2 = cv2.imread('dataset/processed/' + df['id'][1] + '.png', cv2.IMREAD_GRAYSCALE)
# print(mask.max())
# print(mask2.shape)
# print(mask2.max())

# plt.imshow(img)
# plt.imshow(mask, alpha=0.5)
# plt.title('Picture with Mask Appplied')
# plt.show()


class DroneDataset(Dataset):
    
    def __init__(self, img_path, mask_path, X, mean, std, transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.transform = transform
        self.mean = mean
        self.std = std
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        img = cv2.imread(self.img_path + self.X[i] + '.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_path + '0' + self.X[i] + '.png', cv2.IMREAD_GRAYSCALE)
        
        #how to transform?
        if self.transform:
            img = self.transform(img)
            mask = self.transform(img)
            
        img = Image.fromarray(img)
        trans = T.Compose(
                        [T.ToTensor(),
                        T.Normalize(self.mean, self.std)]
                        )
        img = trans(img)
        # simply calling the below will not work, you have to forward it to return something, 
        # thus, we have to use T.Compose and then call it.
        # img = T.ToTensor()
        # img = T.Normalize(self.mean, self.std)
        mask = torch.from_numpy(mask).long()
        
        # if self.patches:
        #     img, mask = self.tiles(img, mask)
            
        return img, mask
    
    
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]


#create datasets
train_set = DroneDataset(IMAGE_PATH, MASK_PATH, X_train, mean, std)
val_set = DroneDataset(IMAGE_PATH, MASK_PATH, X_val, mean, std)

#load data
batch_size= 1

# train_loader = DataLoader(train_set)
# val_loader = DataLoader(val_set)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

X_train_sample = next(iter(train_loader))
print(X_train_sample)

model = UNet()

def pixel_accuracy():
    pass

def mIoU(pred, mask):
    pass

def fit():
    pass

def predict():
    pass

