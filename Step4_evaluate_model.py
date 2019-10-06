# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 10:20:27 2019

@author: Leo
"""
# 使用bleu指标评估模型
import numpy as np
from numpy import argmax
from pickle import load,dump
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model,Model
from nltk.translate.bleu_score import corpus_bleu
import help_func as func
from keras.applications.vgg16 import VGG16,preprocess_input
from keras.preprocessing.image import load_img,img_to_array

# 根据tokenizer将一个整数转为单词
def word_for_id(integer, tokenizer):
    for word,word_id in tokenizer.word_index.items():      # 遍历整个tokenizer
        if word_id == integer:                             # 寻找匹配项并输出word
            return word
    return None

# 根据图像特征为单张图片生成一段描述
def generate_dsc(model, tokenizer, photo, max_length):
    in_text = 'startseq'                                        # 文件头
    for i in range(max_length):
        input_seq = tokenizer.texts_to_sequences([in_text])[0]  # 使用tokenizer处理（生成数字）
        input_seq = pad_sequences([input_seq],maxlen=max_length)# 按照最大长度充0补齐
        output_seq = model.predict([photo,input_seq], verbose=0)# 使用模型预测输出（生成数字）
        output_int = argmax(output_seq)                         # 将预测输出转为整数数字
        output_word = word_for_id(output_int,tokenizer)         # 将预测值（数字）转为单词
        if output_word == None:                                 # 排除特殊情况None
            break
        in_text = in_text + ' ' + output_word                   # 逐个将单词拼接为句子
        if output_word == 'endseq':                             # 遇到结束词就退出循环
            break
    return in_text

# 用bleu评估模型
def evaluate_model(model, descriptions, photos, tokenizer, max_length):
    '''            模型↑   图片描述↑     特征↑            '''
    y_tag, y_pdc = [],[]                                   # 定义标签值和预测值列表
    for img_id, dsc_list in descriptions.items():          # 遍历整个descriptions
        yhat = generate_dsc(model, tokenizer, photos[img_id], max_length)
        references = [d.split() for d in dsc_list]
        y_tag.append(references)
        y_pdc.append(yhat.split())
    print('BLEU-1: %f' % corpus_bleu(y_tag, y_pdc, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(y_tag, y_pdc, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(y_tag, y_pdc, weights=(0.34, 0.33, 0.33, 0)))
    print('BLEU-4: %f' % corpus_bleu(y_tag, y_pdc, weights=(0.25, 0.25, 0.25, 0.25)))
    return None

# 用vgg16提取单张图片特征
def extract_features(filename):
	model = VGG16()
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	img = load_img(filename, target_size=(224, 224))
	img = img_to_array(img)
	img = np.expand_dims(img,axis=0)
	img = preprocess_input(img)
	feature = model.predict(img, verbose=0)
	return feature

# 加载训练数据(6K)
filename = 'Flickr8k_text/Flickr_8k.trainImages.txt'
train = func.load_set(filename)
print('Namelist of train data：%d' % len(train))
# 生成训练图片的描述文件
train_descriptions = func.load_descriptions('descriptions.txt', train)
print('Descriptions of train data：%d' % len(train_descriptions))
# 在训练数据上创建tokenizer，计算词汇表长度
tokenizer = func.create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size：%d' % vocab_size)
# 计算描述语段最大词长度
max_length = func.max_length(train_descriptions)
print('Description Length：%d' % max_length)
# 保存tokenizer到文件
dump(tokenizer, open('tokenizer.pkl', 'wb'))

# 加载测试数据(1K)
filename = 'Flickr8k_text/Flickr_8k.testImages.txt'
test = func.load_set(filename)
print('Namelist of test data：%d' % len(test))
# 生成测试图片的描述文件
test_descriptions = func.load_descriptions('descriptions.txt', test)
print('Descriptions of test data：%d' % len(test_descriptions))
# 加载测试图片的特征文件
test_features = func.load_photo_features('features.pkl', test)
print('Photo features of test data：%d' % len(test_features))

# 加载训练好的模型并评估
filename = 'model_0.h5'
model = load_model(filename)
evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)

# 处理单张图片，先加载tokenizer，再为其添加描述
#tokenizer = load(open('tokenizer.pkl', 'rb'))
photo = extract_features('example.jpg')
description = generate_dsc(model, tokenizer, photo, max_length)
print(description[9:-7])              # 输出描述时去除首尾标识符    