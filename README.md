# Image_Captioning
This is a neural network project. The expected function is to generate descriptions for pictures. That is, to provide a picture, the neural network can automatically generate a paragraph of text for the picture to describe the content of it. The project is based on LSTM and CNN, and uses VGG16 to extract image features.

这是一个神经网络项目，期望实现看图说话功能，即给定一张图片，神经网络能够生成一段文字来描述图片内容，讲述图片中的故事。这个项目以LSTM和CNN为基础，并且使用VGG16模型来提取图片特征。

看图说话是典型的端到端学习，输入端为图片特征和一部分文字（作为引子），期望得到的输出则是图片对应的内容描述。数据集来自Flickr8k_Dataset，读者可以在kaggle搜索下载：Kaggle_Flickr8k。该数据集中包含相对应的图片数据和文本数据。

Step1 提取图片特征

这里使用VGG16来提取图片特征

