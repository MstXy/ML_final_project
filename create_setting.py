# -*- coding: utf-8 -*-
"""
Created on Sat May  8 00:43:30 2021

@author: sidac

This program transforms create rgb mask settings for semantic segmentation editor

"""
import pandas as pd

# print(hex(255)[2:].upper())

label_df = pd.read_csv('dataset/class_dict.csv',skipinitialspace=True)
label_df.columns = ['object', 'r', 'g', 'b']
print(label_df)

file = open('settings_part.txt', 'w')
for index, data in label_df.iterrows():
    # print(data)
    r = hex(data.r)[2:].upper()
    g = hex(data.g)[2:].upper() 
    b = hex(data.b)[2:].upper()
    if len(r) < 2:
        r = '0' + r
    if len(g) < 2:
        g = '0' + g
    if len(b) < 2:
        b = '0' + b
    color = r + g +b
    print(color)
    file.write('      {{"label": "{0}", "color": "#{1}"}},'.format(data.object, color) + '\n')
file.close()