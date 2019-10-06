# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 12:30:19 2019

@author: Leo
"""
# 准备文本数据（图片描述）
# 加载原始txt数据，清洗后保存
import string
import help_func as func

# 清洗文本
def clean_dsc(dsc):
    dsc = [word.lower() for word in dsc]           # 转小写
    dsc = [word for word in dsc if len(word)>1]    # 去除单字符
    dsc = ' '.join(dsc)                            # list2str
    for i in (string.punctuation + string.digits):
        dsc = dsc.replace(i,'')                    # 去除字符、数字 (单词内部的)
    return dsc

# 保存txt文件
def save_descriptions(txt, filename):
    txt = txt.split('\n')                          # 按换行符分割
    descriptions = []
    for line in txt:
        line = line.split()                        # 再次分割
        if len(line)<2:
            continue
        img_id, img_dsc = line[0][:-6], line[1:]   # 提取图片名和图片描述
        img_dsc = clean_dsc(img_dsc)               # 清洗描述文本
        descriptions.append(img_id + ' ' + img_dsc)# 拼接图片名和图片描述
    data = '\n'.join(descriptions)                 # 每条描述间用换行符分隔
    file = open(filename, 'w')
    file.write(data)
    file.close()
    return print('Descriptions saved!')

filename = 'Flickr8k_text/Flickr8k.token.txt'
txt = func.load_doc(filename)
save_descriptions(txt, 'descriptions.txt')