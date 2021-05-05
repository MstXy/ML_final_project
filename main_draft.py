import os

import torch
import torch.nn as nn
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
print('Total Images: ', len(df))


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
        mask = cv2.imread(self.mask_path + self.X[i] + '.png', cv2.IMREAD_GRAYSCALE)
        
        #how to transform?
        if self.transform:
            img = self.transform(img)
            mask = self.transform(img)
            
        img = Image.fromarray(img)
        img = T.ToTensor(img)
        img = T.Normalize(self.mean, self.std)
        mask = torch.from_numpy(mask).long()
        
        if self.patches:
            img, mask = self.tiles(img, mask)
            
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


model = UNet()

def pixel_accuracy(output,label):
    accur = output == label
    return (torch.sum(accur).float() / output.nelement())
#     T_predict=0
#     Total_predict=label.shape[0]*label.shape[1]
#     for i,c in enumerate(label):
#         curr_output=output[i]
#         curr_label=label[i]
#         T_predict+=np.sum(curr_output==curr_label)
#     if T_predict==0:
#         pixel_accur=0
#     else:
#         pixel_accur=T_predict/Total_predict
#     return pixel_accur


# def MiOU(output,label,n_class):
#     to_return_MiOU=np.zeros(n_class)
#     for i in range(0,n_class):
#         pred=np.arange(output.shape[0])[output==i]
#         target=np.arange(label.shape[0])[label==i]
#         n_intersection=np.intersect1d(pred,target).shape[0]
#         n_union=np.union1d(pred,target).shape[0]
#         to_return_MiOU[i]=n_intersection/(n_union+1)
#     return np.mean(to_return_MiOU)
def MiOU(output,label,n_class,batch_size):
    pred = torch.zeros([batch_size, n_class, output.shape[1], output.shape[2]])
    target = torch.zeros([label.shape[0], n_class, label.shape[1], label.shape[2]])
    output=output.unsqueeze(0)
    label=label.unsqueeze(1)
#base on output---->determine transfer the origional pred matrix into the OneHot matrix
    #---->if some output_gray_scale_pixel==n_cls----->transfer to one row of 1-hot matrix
    pred_onehot=pred.scatter_(index=output,dim=1,value=1)
    target_onehot=target.scatter_(index=target,dim=1,value=1)
    batch_mious=[]

    #the number of 1=the value of intersection part
    multiplication=pred_onehot*target_onehot
    for i in range(batch_size):
        iou=[]
        for j in range(n_class):
            intersection = torch.sum(multiplication[i][j])
            #Inclusion–exclusion principle
            union=torch.sum(pred_onehot[i][j])+torch.sum(target_onehot[i][j])-intersection
            iou.append(intersection/union)
        miou=np.mean(iou)
        batch_mious.append(miou)
    return batch_mious

#因为不确定dataloader的structure 我先全部用dataset直接代入了
def fit(epochs, model, train_loader, val_loader, optimizer, scheduler, patch=False,n_class):
    torch.cuda.empty_cache()
    train_losses = []
    val_losses = []
    train_miou=[]
    val_miou=[]
    train_arr=[]
    val_arr=[]
    mini_loss=float('inf')
    for each in range(epochs):
        miou=0
        arr=0
        for data in range(len(train_set)):
            img=data[0]
            mask=data[1]
            prediction=model(data[0])
            loss=nn.CrossEntropyLoss(prediction,mask)#how to split mask from the train_loader
            train_losses.append(loss)
            train_miou+=MiOU(prediction,mask,n_class)
            train_arr+=pixel_accuracy(prediction,mask)

            #update learning rate...
        else:
            for data in range(len(val_set)):
                img=data[0]
                mask=data[1]
                prediction = model(data[0])
                loss = nn.CrossEntropyLoss(prediction, mask)  # how to split mask from the train_loader
                train_losses.append(loss)
                train_miou += MiOU(prediction, mask, n_class)
                train_arr += pixel_accuracy(prediction, mask)

            #determine when to stop the train through the epochs
            if mini_loss>loss:








def predict():
    pass

