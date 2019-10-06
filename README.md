# Image_Captioning
This is a neural network project. The expected function is to generate descriptions for pictures. That is, to provide a picture, the neural network can automatically generate a paragraph of text for the picture to describe the content of it. The project is based on LSTM and CNN, and uses VGG16 to extract image features.

这是一个神经网络项目，期望实现看图说话功能，即给定一张图片，神经网络能够生成一段文字来描述图片内容，讲述图片中的故事。这个项目以LSTM和CNN为基础，并且使用VGG16模型来提取图片特征。

看图说话是典型的端到端学习，输入端为图片特征和一部分文字（作为引子），期望得到的输出则是图片对应的内容描述。所以要求数据集中既包含图片，又包含对应的描述。该项目数据集来自Flickr8k_Dataset，读者可以在kaggle搜索下载：[Kaggle_Flickr8k](https://www.kaggle.com/shadabhussain/flickr8k)。

本项目主要包含四个部分：1.使用VGG16提取图片特征并保存为文件；2.预处理数据集中的描述文本并保存为文件；3.构建模型并训练；4.评估模型并为图片生成描述。

## Step1 提取图片特征<br>
本项目使用预先训练好的VGG16来提取图片特征。为了适配该项目，VGG16模型最后一层被去掉，输出为一个1×4096的向量。<br>
![VGG16](https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1570385062326&di=c2aaf62c394a72b2a97b1793d9b2de26&imgtype=0&src=http%3A%2F%2Fws1.sinaimg.cn%2Flarge%2F662f5c1fgy1frnjdmk4n5j21710pc46d.jpg)
