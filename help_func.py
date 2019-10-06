# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 10:27:49 2019

@author: Leo
"""
# 协助函数
from pickle import load
from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.models import Model
from keras.applications.vgg16 import VGG16,preprocess_input
from keras.preprocessing.image import load_img,img_to_array
from os import listdir

def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

# 获取图片名称集合
def load_set(filename):
    txt = load_doc(filename)
    namelist = []
    txt = txt.split('\n')
    for line in txt:
        if len(line)<1:                                 # txt文件最后一行为空
            continue
        namelist.append(line[:-4])                      # 去尾缀
    return set(namelist)                                # 设置为集合，排除重复项

# 加载名单中的图片描述
def load_descriptions(filename, namelist):
    descriptions = {}
    txt =load_doc(filename)
    txt = txt.split('\n')
    for line in txt:
        line = line.split()
        img_id, img_dsc = line[0], line[1:]             # 提取图片名和图片描述
        if img_id in namelist:
            # 添加首尾标志'startseq'和'endseq'
            img_dsc = 'startseq ' + ' '.join(img_dsc) + ' endseq'
            if img_id not in descriptions:              # 判断key是否存在
                descriptions[img_id] = []               # 添加键值对
            descriptions[img_id].append(img_dsc)        # 将描述写入字典
    return descriptions

# 获取名单里图片的特征（加载.pkl文件）
def load_photo_features(filename, namelist):
    file = open(filename,'rb')
    all_features = load(file)
    features = {k: all_features[k] for k in namelist}   # 筛选名单内容
    file.close()
    return features

# 将descriptions由字典转为list
def to_lines(descriptions):
    all_dsc = []
    for i in range(len(descriptions.keys())):
        img_id = list(descriptions.keys())[i]
        for j in range(len(descriptions[img_id])):
            img_dsc = list(descriptions[img_id])[j]
            all_dsc.append(img_dsc)
    return all_dsc

# 为图片描述创建tokenizer
def create_tokenizer(descriptions):
    lines = to_lines(descriptions)                      # 字典转列表
    tokenizer = Tokenizer()                             # 创建tokenizer
    tokenizer.fit_on_texts(lines)                       # fit
    return tokenizer

# 计算最长的一段描述包含多少单词
def max_length(descriptions):
    lines = to_lines(descriptions)
    max_length = max(len(line.split()) for line in lines)
    return max_length