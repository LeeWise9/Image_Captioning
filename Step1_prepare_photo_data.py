# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 11:32:01 2019

@author: Leo
"""
# 准备图像数据（图片特征）
# 加载模型vgg16，去除最后一层
# 用该模型提取图片特征（1×4096的矩阵）
# 将提取的特征保存到名为 “ features.pkl ” 的文件中
import numpy as np
from keras.models import Model
from keras.applications.vgg16 import VGG16,preprocess_input
from keras.preprocessing.image import load_img,img_to_array
from os import listdir
from pickle import dump

# 用vgg16提取图片特征 ,从特定目录
def extract_features(directory):
    model = VGG16()          # 加载模型VGG16
    model.layers.pop()       # 去掉最后一层
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    model.summary()          # 输出summarize
    
    features = {}            # 从图片中提取特征,先构建一个空字典
    img_namelist = listdir(directory)          # 创建图像名列表
    for img_name in img_namelist:
        name = directory +'/'+ img_name        # 创建文件名
        # 加载图片
        img = load_img(name,target_size=(224, 224))
        img = img_to_array(img)                # 转换为矩阵
        img = np.expand_dims(img,axis=0)       # 矩阵增维
        img = preprocess_input(img)            # 预处理：均值化
       
        feature = model.predict(img,verbose=0) # 提取特征并保存
        features[img_name[:-4]] = feature      # 将向量写入字典
        print('>正在处理：',img_name)
    return features

directory = 'Flickr8k_Dataset'
features  = extract_features(directory)
dump(features, open('features.pkl', 'wb'))     # 保存文件
print('图片特征提取完成，文件已保存！')
