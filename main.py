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
from dataset import DroneDataset, DroneTestDataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


# IMAGE_PATH = 'dataset/compressed_img/'
# MASK_PATH = 'dataset/compressed_mask/'
IMAGE_PATH = 'dataset/new_size_img/'
MASK_PATH = 'dataset/new_size_mask/'


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



    
    
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]


# adding transformation after 3rd training
# not really working
# t =  A.Compose([
#                 A.HorizontalFlip(), 
#                 A.VerticalFlip(), 
#                 A.GridDistortion(p=0.2), 
#                 A.RandomBrightnessContrast((0,0.5),(0,0.5)),
#                 A.GaussNoise()
#                 ])

#create datasets
train_set = DroneDataset(IMAGE_PATH, MASK_PATH, X_train, mean, std)
val_set = DroneDataset(IMAGE_PATH, MASK_PATH, X_val, mean, std)
#load data
batch_size= 3

# train_loader = DataLoader(train_set)
# val_loader = DataLoader(val_set)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)


# # get one sample
# X_train_sample, y_train_sample = next(iter(train_loader))
# print(X_train_sample, y_train_sample)

# model = UNet()
model = smp.Unet(
    'mobilenet_v2', 
    encoder_weights='imagenet', 
    classes=23, 
    encoder_depth=5, 
    decoder_channels=[256, 128, 64, 32, 16]
    )

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

def fit(epochs, model, train_loader, val_loader, criterion, optimizer, scheduler,  batch_size, n_class=23):
    torch.cuda.empty_cache()
    train_losses = []
    val_losses = []
    train_miou = []
    val_miou = []
    train_accuracy = []
    val_accuracy = []
    mini_loss = float('inf')
    no_progress = 0
    
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
            total_miou += MiOU(prediction,mask) 
            total_accuracy += pixel_accuracy(prediction,mask)

            # back prop
            optimizer.zero_grad() # when processing a new batch, clear the gradient on start
            loss.backward() # calculate gradients
            optimizer.step() # update weights

            #TBD: update learning rate...
            scheduler.step()

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

                    total_val_miou += MiOU(prediction,mask,n_class)
                    total_val_accuracy += pixel_accuracy(prediction, mask)

                    total_val_loss += loss.item()



                # calculate loss
                this_train_loss = total_loss/len(train_loader)
                this_val_loss = total_val_loss/len(val_loader)

                train_losses.append(this_train_loss)
                val_losses.append(this_val_loss)
            
                ## determine when to stop the train through the epochs
                if mini_loss > this_val_loss:
                    print('Loss Decreasing.. {:.3f} >> {:.3f} '.format(mini_loss, this_val_loss))
                    mini_loss = this_val_loss
                else:
                    no_progress += 1
                    mini_loss = this_val_loss
                    print(f'Loss Not Decrease for {no_progress} time')
                    if no_progress == 10:
                        print('Loss not decrease for 10 times, Stop Training')
                        break

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
epoch = 30
weight_decay = 0.0001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epoch,
                                            steps_per_epoch=len(train_loader))

train_losses, val_losses, train_miou, val_miou, train_accuracy, val_accuracy = fit(
    epoch, model, train_loader, val_loader, criterion, optimizer, scheduler, batch_size
)

train_log = {'train_loss' : train_losses, 'val_loss': val_losses,
        'train_miou' :train_miou, 'val_miou':val_miou,
        'train_acc' :train_accuracy, 'val_acc':val_accuracy}

# save the model
torch.save(model, 'Unet_5.pt')

# plot the result
def plot_loss(log):
    plt.plot(log['val_loss'], label='val', marker='o')
    plt.plot(log['train_loss'], label='train', marker='o')
    plt.title('Loss per epoch')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.savefig('loss.png')
    plt.show()
    
def plot_miou(log):
    plt.plot(log['train_miou'], label='train_mIoU', marker='*')
    plt.plot(log['val_miou'], label='val_mIoU',  marker='*')
    plt.title('Score per epoch')
    plt.ylabel('mean IoU')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.savefig('miou.png')
    plt.show()
    
def plot_accuracy(log):
    plt.plot(log['train_acc'], label='train_accuracy', marker='*')
    plt.plot(log['val_acc'], label='val_accuracy',  marker='*')
    plt.title('Accuracy per epoch')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.savefig('accuracy.png')
    plt.show()

plot_loss(train_log)
plot_miou(train_log)
plot_accuracy(train_log)


model = torch.load("Unet_5.pt")
test_set = DroneTestDataset(IMAGE_PATH, MASK_PATH, X_test)
# print(test_set[0])

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

model_miou = model_miou_score(model, test_set)
model_accuracy = model_pixel_accuracy(model, test_set)

print('Test Set MiOU: ', model_miou)
print('Test Set Pixel Accuracy: ', model_accuracy)

# visualize pic 1
image, mask = test_set[0]

# print(image, mask)
pred_mask, score = predict_image_mask_miou(model, image, mask)
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20,10))
ax1.imshow(image)
ax1.set_title('Picture 1')

ax2.imshow(mask)
ax2.set_title('Ground Truth Mask')
ax2.set_axis_off()

ax3.imshow(pred_mask)
ax3.set_title('Predicted | MIOU {:.3f}'.format(score))
ax3.set_axis_off()
fig.savefig('picture_1.png')

plt.show()


# visualize pic 2
image, mask = test_set[1]
pred_mask, score = predict_image_mask_miou(model, image, mask)
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20,10))
ax1.imshow(image)
ax1.set_title('Picture 2')

ax2.imshow(mask)
ax2.set_title('Ground Truth Mask')
ax2.set_axis_off()

ax3.imshow(pred_mask)
ax3.set_title('Predicted | MIOU {:.3f}'.format(score))
ax3.set_axis_off()
fig.savefig('picture_2.png')

plt.show()


# visualize pic 3
image, mask = test_set[2]
pred_mask, score = predict_image_mask_miou(model, image, mask)
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20,10))
ax1.imshow(image)
ax1.set_title('Picture 3')

ax2.imshow(mask)
ax2.set_title('Ground Truth Mask')
ax2.set_axis_off()

ax3.imshow(pred_mask)
ax3.set_title('Predicted | MIOU {:.3f}'.format(score))
ax3.set_axis_off()
fig.savefig('picture_3.png')

plt.show()
