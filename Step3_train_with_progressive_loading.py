# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 17:06:01 2019

@author: Leo
"""
# 逐步加载训练，避免 memory error
# 加载训练数据：图片描述，图片特征
# 创建tokenizer
# 训练模型
from numpy import array
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input,Dense,LSTM,Embedding,Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
import help_func as func

# 为神经网络创建输入和标签序列
# X1--图片特征
# X2--图片描述（上文）
# y---标签（图片描述/下文）
def create_sequences(tokenizer, max_length, dsc_list, photos, vocab_size):
    X1,X2,y = [],[],[]
    for line in dsc_list:
        seq = tokenizer.texts_to_sequences([line])[0]
        # 每一段描述都需要构成若干组上下文数据对
        # 比如：一段包含n个词的句子，需要构建n-1个数据对
        for i in range(1,len(seq)):
            in_seq, out_seq = seq[:i], seq[i]           # 输入取前，输出取后
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]   # 补齐
            # 转换为独热编码
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            X1.append(photos)
            X2.append(in_seq)
            y.append(out_seq)
    return array(X1), array(X2), array(y)

# 构建模型，模型是词汇量和句子长度的函数
def define_model(vocab_size, max_length):
    # 图像特征
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    # 图像描述
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    # 融合层
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs  = Dense(vocab_size, activation='softmax')(decoder2)
    # 输入输出
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    # 编译模型
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # 输出模型结构
    model.summary()
    return model

# 数据生成器
def data_generator(descriptions, photos, tokenizer, max_length, vocab_size):
    while True: # 只要调用该函数，就源源不断生成数据，所以这是一个死循环
        for img_id,dsc_list in descriptions.items():
            photo = photos[img_id][0]
            in_img, in_seq, out_word = create_sequences(tokenizer, max_length, dsc_list, photo, vocab_size)
            yield [[in_img, in_seq], out_word]

# 加载训练数据
filename = 'Flickr8k_text/Flickr_8k.trainImages.txt'
train = func.load_set(filename)
print('Namelist of train data：%d' % len(train))
# 生成训练图片的描述文件
train_descriptions = func.load_descriptions('descriptions.txt', train)
print('Descriptions of train data：%d' % len(train_descriptions))
# 加载训练图片的特征文件
train_features = func.load_photo_features('features.pkl', train)
print('Photo features of train data：%d' % len(train_features))
# 在训练数据上创建tokenizer，计算词汇表长度
tokenizer = func.create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size：%d' % vocab_size)
# 计算描述语段最大词长度
max_length = func.max_length(train_descriptions)
print('Description Length：%d' % max_length)

# 定义模型
model = define_model(vocab_size, max_length)
# 训练模型
epochs = 20
steps = len(train_descriptions)
for i in range(epochs):
    generator = data_generator(train_descriptions, train_features, tokenizer, max_length, vocab_size)
    model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
    model.save('model_' + str(i) + '.h5')