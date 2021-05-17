#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 19:17:00 2021

@author: 
"""
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor

def file_reader(file_name):
    with open(file_name) as json_file:
        return json.load(json_file)
    
def rle_decoding(rle):
    # dict.get(key,[value,]) return a defult value if key not exists
    h, w = rle.get('size',[720,1280])
    rle_arr = rle.get('counts',[0])
    rle_arr = np.cumsum(rle_arr)
    indices = []
    extend = indices.extend
    list(map(extend, map(lambda s,e: range(s, e), rle_arr[0::2], rle_arr[1::2])));
    binary_mask = np.zeros(h*w, dtype=np.uint8)
    binary_mask[indices] = 1
    return binary_mask.reshape((w, h)).T

def show_image(img):
    img = Image.fromarray(img)               
    img.show()

def get_colors():
    COLORS = []
    for name, code in ImageColor.colormap.items():
         COLORS.append(name)
    np.random.shuffle(COLORS)
    return COLORS

COLORS = get_colors()

def draw_bbox(image, box=None, conf=None, class_id=None, mask=None, class_dict=None, mode='return'):
    if np.any(class_id is None) and np.any(box is not None):
        class_id = np.arange(len(box))
    if class_dict is not None and np.any(class_id is not None) :
        class_id = [class_dict[int(c)] for c in class_id]
    img = Image.fromarray(np.array(image*255,dtype=np.uint8))                   
    draw = ImageDraw.Draw(img)   
    if np.any(box is not None):
        for i,(bb,cc) in enumerate(zip(box, class_id)):
            if np.any(bb>0):
                draw.rectangle(list(bb), outline = COLORS[i%len(COLORS)]) 
                xy = ((bb[2]+bb[0])*0.5, (bb[3]+bb[1])*0.5)
                if not type(cc) is str:
                    cc = str(np.round(cc,3))
                draw.text(xy, cc, \
                          font=ImageFont.truetype("./other/arial.ttf"), \
                          fontsize = 15, fill=COLORS[i%len(COLORS)])
    img = np.array(img)
    if np.any(mask is not None):
        for i, mm in enumerate(mask):
            if np.any(mm>0):
                mm = np.array(Image.new('RGB', (img.shape[1], img.shape[0]), \
                                        color = COLORS[i%len(COLORS)])) * \
                                        np.tile(mm[...,None],(1,1,3))
                img[mm>0] = img[mm>0]*0.5+mm[mm>0]*0.5
    if mode == 'PIL':
        show_image(img)
    else:
        return img