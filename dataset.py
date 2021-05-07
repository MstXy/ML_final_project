import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2

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
            augmented = self.transform(image=img, mask=mask)
            img = Image.fromarray(augmented['image'])
            mask = augmented['mask']
        else:   
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


class DroneTestDataset(Dataset):
    
    def __init__(self, img_path, mask_path, X):
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
      
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.img_path + self.X[idx] + '.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_path + '0' + self.X[idx] + '.png', cv2.IMREAD_GRAYSCALE)
               
        mask = torch.from_numpy(mask).long()
        
        return img, mask