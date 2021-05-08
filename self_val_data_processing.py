import os
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

IMAGE_PATH = 'self-create valid set/'
PROC_IMG_PATH = 'val_dataset/processed_val_img/'
PROC_IMG_PATH_2 = 'val_dataset/processed_val_rgb/'

def loadImages(path):
    # Put file names into lists
    image_files = [os.path.join(path, file)
         for file in os.listdir(path)]
 
    return image_files

image_files = loadImages(IMAGE_PATH)
# print(image_files)
# print(len(image_files)) 

# im = Image.open(image_files[0])
# w,h = im.size
# print(w,h)
# im = im.crop((0, 0, 750, 500))
# im.save(os.path.join(PROC_IMG_PATH, '0.png') )
WIDTH, HEIGHT = 768,512
def processing(image_files, x_start= 0, y_start=0, x_stride=1500, y_stride=1000):
    for i, file in enumerate(image_files):
        im = Image.open(image_files[i])
        w,h = im.size
        x= x_start
        count = 0
        while x + x_stride < w:
            y = y_start
            while y + y_stride < h:
                im = Image.open(image_files[i])
                im = im.crop((x, y, x+x_stride, y+y_stride))
                im = im.resize((WIDTH, HEIGHT),Image.NEAREST)
                save_name = str(i)+'_'+str(count)+'.png'
                im.save(os.path.join(PROC_IMG_PATH, save_name) )
                count += 1
                y += y_stride
            im = Image.open(image_files[i])
            im = im.crop((x, h-y_stride, x+x_stride, h))
            im = im.resize((WIDTH, HEIGHT),Image.NEAREST)
            save_name = str(i)+'_'+str(count)+'.png'
            im.save(os.path.join(PROC_IMG_PATH, save_name) )
            count += 1
            
            x += x_stride
            
        y = y_start
        while y + y_stride < h:
            im = Image.open(image_files[i])
            im = im.crop((w-x_stride, y, w, y+y_stride))
            im = im.resize((WIDTH, HEIGHT),Image.NEAREST)
            save_name = str(i)+'_'+str(count)+'.png'
            im.save(os.path.join(PROC_IMG_PATH, save_name) )
            count += 1
            y += y_stride
        im = Image.open(image_files[i])
        im = im.crop((w-x_stride, h-y_stride, w, h))
        im = im.resize((WIDTH, HEIGHT),Image.NEAREST)
        save_name = str(i)+'_'+str(count)+'.png'
        im.save(os.path.join(PROC_IMG_PATH, save_name) )
        count += 1



# processing(image_files)


# im = Image.open(os.path.join(os.getcwd(),'3_4_segmentation .png'))
# im = im.resize((WIDTH, HEIGHT),Image.NEAREST)
# save_name = '3_4_segmentation .png'
# im.save(os.path.join(PROC_IMG_PATH_2, save_name) )
