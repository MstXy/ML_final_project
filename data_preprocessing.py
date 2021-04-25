import os
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt


image_path = "dataset/"

def loadImages(path):
    # Put file names into lists
    image_files = sorted([os.path.join(path, 'label_images', file)
         for file in os.listdir(path + "/label_images")])
 
    return image_files

image_files = loadImages(image_path)
print(image_files)
print(len(image_files)) #400, good.

label_df = pd.read_csv('dataset/class_dict.csv',index_col=False,skipinitialspace=True)
print(label_df)

# TEST ============================================================================
# test the first picture

test_img = Image.open(image_files[0])
test_img = np.asarray(test_img)
# print(test_img)
print(test_img.shape) # (4000,6000,3)

test_output = np.zeros(test_img.shape).astype(int) # make it integers, since class are integers
# print(test_output)
for index, data in label_df.iterrows():
    r = test_img[:,:,0]
    g = test_img[:,:,1]
    b = test_img[:,:,2]
    test_output[(r==data.r)&(g==data.g)&(b==data.b)] = np.array([index, index, index]).reshape(1,3)

test_output = test_output[:,:,0]
output_filename = image_files[0][-7:]
cv2.imwrite(os.path.join(image_path,'processed', output_filename),test_output)
# TEST END ============================================================================


def processImages(image_files):
    for i, item in enumerate(image_files):
        print("Processing:", i, item)
        rgb_mask = Image.open(item)
        rgb_mask = np.asarray(rgb_mask)

        output = np.zeros(rgb_mask.shape).astype(int)

        for index, data in label_df.iterrows():
            r = rgb_mask[:,:,0]
            g = rgb_mask[:,:,1]
            b = rgb_mask[:,:,2]
            output[(r==data.r)&(g==data.g)&(b==data.b)] = np.array([index, index, index]).reshape(1,3)
        output = output[:,:,0]
        output_filename = item[-7:]
        cv2.imwrite(os.path.join(image_path,'processed', output_filename),output)

processImages(image_files)




