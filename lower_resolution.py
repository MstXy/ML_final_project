import os
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt


image_path = "dataset/"


"""
def loadImages(path):
    # Put file names into lists
    image_files = sorted([os.path.join(path, 'images', file)
         for file in os.listdir(path + "/images")])
 
    return image_files

image_files = loadImages(image_path)
print(image_files)
print(len(image_files)) #400, good.


# TEST ============================================================================
# test the first picture

test_img = Image.open(image_files[0])
test_img = test_img.resize((750, 500),Image.ANTIALIAS)
# print(test_img)

output_filename = image_files[0][-7:]
test_img.save(os.path.join(image_path,'compressed_img', output_filename))
# TEST END ============================================================================


def processImages(image_files):
    for i, item in enumerate(image_files):
        print("Processing:", i, item)
        img = Image.open(item)
        img = img.resize((750, 500),Image.ANTIALIAS)
        # print(test_img)

        output_filename = item[-7:]
        img.save(os.path.join(image_path,'compressed_img', output_filename))

processImages(image_files)
"""


def loadImages(path):
    # Put file names into lists
    image_files = sorted([os.path.join(path, 'processed', file)
         for file in os.listdir(path + "/processed")])
 
    return image_files

image_files = loadImages(image_path)
print(image_files)
print(len(image_files)) #400, good.


# TEST ============================================================================
# test the first picture

test_img = Image.open(image_files[0])
test_img = test_img.resize((750, 500),Image.ANTIALIAS)
# print(test_img)

output_filename = image_files[0][-7:]
test_img.save(os.path.join(image_path,'compressed_mask', output_filename))
# TEST END ============================================================================


def processImages(image_files):
    for i, item in enumerate(image_files):
        print("Processing:", i, item)
        img = Image.open(item)
        img = img.resize((750, 500),Image.ANTIALIAS)
        # print(test_img)

        output_filename = item[-7:]
        img.save(os.path.join(image_path,'compressed_mask', output_filename))

processImages(image_files)


