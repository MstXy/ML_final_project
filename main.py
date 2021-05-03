import os
import time

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


# # get one sample
# X_train_sample, y_train_sample = next(iter(train_loader))
# print(X_train_sample, y_train_sample)

model = UNet()

def pixel_accuracy(output,label):
    T_predict=0
    Total_predict=label.shape[0]*label.shape[1]
    for i,c in enumerate(label):
        curr_output=output[i]
        curr_label=label[i]
        T_predict+=np.sum(curr_output==curr_label)
    if T_predict==0:
        pixel_accur=0
    else:
        pixel_accur=T_predict/Total_predict
    return pixel_accur


def MiOU(output,label,n_class):
    to_return_MiOU=np.zeros(n_class)
    for i in range(0,n_class):
        pred=np.arange(output.shape[0])[output==i]
        target=np.arange(label.shape[0])[label==i]
        n_intersection=np.intersect1d(pred,target).shape[0]
        n_union=np.union1d(pred,target).shape[0]
        to_return_MiOU[i]=n_intersection/(n_union+1)
    return np.mean(to_return_MiOU)

def fit(epochs, model, train_loader, val_loader, criterion, optimizer, n_class=23):
    torch.cuda.empty_cache()
    train_losses = []
    val_losses = []
    train_miou = []
    val_miou = []
    train_accuracy = []
    val_accuracy = []
    mini_loss = float('inf')
    
    model.to(device)
    train_begin = time.time()

    for epoch in range(epochs):

        begin_timer = time.time()
        total_loss = 0
        total_miou = 0
        total_accuracy = 0

        # start training
        model.train()
        for i, batch in enumerate(train_loader): # get batch
            img, mask = batch
            # !!tensor and model are different, not inplace 
            img = img.to(device)
            mask = mask.to(device)

            prediction = model(img) # pass batch
            loss = criterion(prediction,mask) # calculate loss, loss tensor
            # add to evaluation metrics
            total_miou += MiOU(prediction,mask,n_class) 
            total_accuracy += pixel_accuracy(prediction,mask)

            # back prop
            optimizer.zero_grad() # when processing a new batch, clear the gradient on start
            loss.backward() # calculate gradients
            optimizer.step() # update weights

            #TBD: update learning rate...

            total_loss += loss.item()

        # validation
        else:
            model.eval()
            total_val_loss = 0
            total_val_miou = 0
            total_val_accuracy = 0
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    img, mask = batch
                    # !!tensor and model are different, not inplace 
                    img = img.to(device)
                    mask = mask.to(device)

                    prediction = model(img)
                    loss = criterion(prediction, mask)  # how to split mask from the train_loader
                    total_val_loss += loss.item()

                    total_val_miou += MiOU(prediction, mask, n_class)
                    total_val_accuracy += pixel_accuracy(prediction, mask)

                    total_val_loss += loss.item()


                ## determine when to stop the train through the epochs
                # if mini_loss>loss:
                #     pass

                # calculate loss
                this_train_loss = total_loss/len(train_loader)
                this_val_loss = total_val_loss/len(val_loader)

                train_losses.append(this_train_loss)
                val_losses.append(this_val_loss)

                # calculate iou 
                this_miou = total_miou/len(train_loader)
                this_val_miou = total_val_miou/len(val_loader)

                train_miou.append(this_miou)                
                val_miou.append(this_val_miou)

                # calculate accuracy
                this_accuracy = total_accuracy/len(train_loader)
                this_val_accuracy = total_val_accuracy/ len(val_loader)
                train_accuracy.append(this_accuracy)
                val_accuracy.append(this_val_accuracy)

                print("Epoch:{}/{}..".format(epoch+1, epochs),
                    "Train Loss: {:.3f}..".format(this_train_loss),
                    "Val Loss: {:.3f}..".format(this_val_loss),
                    "Train mIoU:{:.3f}..".format(this_miou),
                    "Val mIoU: {:.3f}..".format(this_val_miou),
                    "Train Acc:{:.3f}..".format(this_accuracy),
                    "Val Acc:{:.3f}..".format(this_val_accuracy),
                    "Time: {:.2f}m".format((time.time()-begin_timer)/60))

    duration = time.time() - train_begin
    print('Total time: {:.2f} m' .format(duration/60))

    return train_losses, val_losses, train_miou, val_miou, train_accuracy, val_accuracy


lr = 0.001
epoch = 1
weight_decay = 0
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


train_losses, val_losses, train_miou, val_miou, train_accuracy, val_accuracy = fit(
    epoch, model, train_loader, val_loader, criterion, optimizer
)

train_log = {'train_loss' : train_losses, 'val_loss': val_losses,
        'train_miou' :train_miou, 'val_miou':val_miou,
        'train_acc' :train_accuracy, 'val_acc':val_accuracy}

def predict():
    pass



# model return size 1*24*308*564], mask.size() = 1*500*750
# need to modify