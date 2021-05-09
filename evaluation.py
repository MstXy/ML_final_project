import os
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torch.nn.functional as F

# import tensorflow as tf
import segmentation_models_pytorch as smp

# for image augmentation
import albumentations as A

from sklearn.model_selection import train_test_split

from PIL import Image
import cv2

from unet import UNet
# from dataset import DroneDataset, DroneTestDataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR = 'trained_model.pt'
IMAGE_PATH = 'val_dataset/processed_val_img/'
MASK_PATH = 'val_dataset/processed_val_mask/'

model = smp.Unet(
    'mobilenet_v2', 
    encoder_weights='imagenet', 
    classes=23, 
    encoder_depth=5, 
    decoder_channels=[256, 128, 64, 32, 16]
    )
model.load_state_dict(torch.load(MODEL_DIR))

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

class DroneTestDataset(Dataset):
    
    def __init__(self, img_path, mask_path, X):
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
      
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.img_path + self.X[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        try:
            mask = cv2.imread(self.mask_path +  self.X[idx] , cv2.IMREAD_GRAYSCALE)
                   
            mask = torch.from_numpy(mask).long()
        except:
            return img
        
        
        return img, mask

def pixel_accuracy(output,label):
    output = torch.argmax(F.softmax(output, dim=1), dim=1)
    accur = torch.eq(output, label).int()
    return (torch.sum(accur).float() / output.nelement())

def MiOU(pred_mask, mask, smooth=1e-10, n_classes=23):
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(0, n_classes): #loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0: #no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union +smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)

# evaluation
def predict_image_mask_miou(model, image, mask):
    model.eval()
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image = t(image)
    model.to(device)
    image = image.to(device)
    mask = mask.to(device)
    with torch.no_grad():
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)
        
        output = model(image)
        score = MiOU(output, mask)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked, score

def predict_image_mask_accuracy(model, image, mask):
    model.eval()
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image = t(image)
    model.to(device)
    image = image.to(device)
    mask = mask.to(device)
    with torch.no_grad():
        
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)
        
        output = model(image)
        accuracy = pixel_accuracy(output, mask).item()
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked, accuracy


# model score
def model_miou_score(model, test_set):
    score_iou = []
    for i in range(len(test_set)):
        img, mask = test_set[i]
        pred_mask, score = predict_image_mask_miou(model, img, mask)
        score_iou.append(score)
    # print(score_iou)
    return np.mean(score_iou)

def model_pixel_accuracy(model, test_set):
    accuracy = []
    for i in range(len(test_set)):
        img, mask = test_set[i]
        pred_mask, acc = predict_image_mask_accuracy(model, img, mask)
        accuracy.append(acc)
    # print(accuracy)
    return np.mean(accuracy)


# X_test = np.asarray(['3_4.png'])
# test_set = DroneTestDataset(IMAGE_PATH, MASK_PATH, X_test)
# model_miou = model_miou_score(model, test_set)
# model_accuracy = model_pixel_accuracy(model, test_set)

# print('Test Set MiOU: ', model_miou)
# print('Test Set Pixel Accuracy: ', model_accuracy)

# image, mask = test_set[0]

# # print(image, mask)
# pred_mask, score = predict_image_mask_miou(model, image, mask)
# fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20,10))
# ax1.imshow(image)
# ax1.set_title('Picture val')

# ax2.imshow(mask)
# ax2.set_title('Ground Truth Mask')
# ax2.set_axis_off()

# ax3.imshow(pred_mask)
# ax3.set_title('Predicted | MIOU {:.3f}'.format(score))
# ax3.set_axis_off()
# fig.savefig('picture_val.png')

# plt.show()



def predict_image_mask(model, image):
    model.eval()
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image = t(image)
    model.to(device)
    image = image.to(device)
    with torch.no_grad():
        image = image.unsqueeze(0)
        
        output = model(image)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked

X_test=[]
for root, dirnames, filenames in os.walk(IMAGE_PATH):
        for filename in filenames:
            X_test.append(filename)
X_test = np.asarray(X_test)
test_set = DroneTestDataset(IMAGE_PATH, MASK_PATH, X_test)

for i, data in enumerate(test_set):
    image = test_set[i]
    
    # print(image, mask)
    pred_mask = predict_image_mask(model, image)
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,10))
    ax1.imshow(image)
    ax1.set_title('Picture val')
    
    ax2.imshow(pred_mask)
    ax2.set_title('Predicted')
    ax2.set_axis_off()
    fig.savefig('val_result/picture_'+str(i)+'.png')
    
    plt.show()