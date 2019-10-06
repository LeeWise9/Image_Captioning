# Image_Captioning 看图说话
This is a neural network project. The expected function is to generate descriptions for pictures. That is, to provide a picture, the neural network can automatically generate a paragraph of text for the picture to describe the content of it. The project is based on LSTM and CNN, and uses VGG16 to extract image features.

这是一个神经网络项目，期望实现看图说话功能，即给定一张图片，神经网络能够生成一段文字来描述图片内容，讲述图片中的故事。这个项目以LSTM和CNN为基础，并且使用VGG16模型来提取图片特征。

看图说话是典型的端到端学习，输入端为图片特征和一部分文字（作为引子），期望得到的输出则是图片对应的内容描述。所以要求数据集中既包含图片，又包含对应的描述。该项目数据集来自Flickr8k_Dataset，读者可以在kaggle搜索下载：[Kaggle_Flickr8k](https://www.kaggle.com/shadabhussain/flickr8k)。

本项目主要包含四个部分：1.使用VGG16提取图片特征并保存为文件；2.预处理数据集中的描述文本并保存为文件；3.构建模型并训练；4.评估模型并为图片生成描述。

## Step1 提取图片特征<br>
本项目使用预先训练好的VGG16来提取图片特征。为了适配该项目，至少要注意两点面：1.要对图片做预处理，包括图片缩放、增维和去均值化；2.去掉VGG16的最后一层（1000的全连接层和softmax激活层），使输出为一个1×4096的向量。<br>
![VGG16](https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1570385062326&di=c2aaf62c394a72b2a97b1793d9b2de26&imgtype=0&src=http%3A%2F%2Fws1.sinaimg.cn%2Flarge%2F662f5c1fgy1frnjdmk4n5j21710pc46d.jpg)
提取到的特征可以用字典形式储存。当目标文件夹中的所有图片都完成了特征提取操作之后，结果将被保存为.pkl文件，以方便后续调用。<br>
除了使用VGG16，读者还可以尝试使用其他的预训练网络进行尝试，比如ResNet50。

## Step2 描述文本预处理<br>
Flickr8k_Dataset数据集中包含图片名和对应的描述文本，用空格分割，以.txt形式保存。其中，一张图片有5段不同文字描述。<br>
>1000268201_693b08cb0e.jpg#0	A child in a pink dress is climbing up a set of stairs in an entry way .
>1000268201_693b08cb0e.jpg#1	A girl going into a wooden building .
>1000268201_693b08cb0e.jpg#2	A little girl climbing into a wooden playhouse .
>1000268201_693b08cb0e.jpg#3	A little girl climbing the stairs to her playhouse .
>1000268201_693b08cb0e.jpg#4	A little girl in a pink dress going into a wooden cabin .
>1001773457_577c3a7d70.jpg#0	A black dog and a spotted dog are fighting
>1001773457_577c3a7d70.jpg#1	A black dog and a tri-colored dog playing with each other on the road .
>1001773457_577c3a7d70.jpg#2	A black dog and a white dog with brown spots are staring at each other in the street .
>1001773457_577c3a7d70.jpg#3	Two dogs of different breeds looking at each other on the road .
>1001773457_577c3a7d70.jpg#4	Two dogs on pavement moving toward each other .
>1002674143_1b742ab4b8.jpg#0	A little girl covered in paint sits in front of a painted rainbow with her hands in a bowl .
>1002674143_1b742ab4b8.jpg#1	A little girl is sitting in front of a large painted rainbow .
>...
预处理工作需要去除图片名称尾缀，并且清洗描述文本，比如将拼写转为小写，去除停顿词、数字、单个字母等。<br>
>1000268201_693b08cb0e child in pink dress is climbing up set of stairs in an entry way
>1000268201_693b08cb0e girl going into wooden building
>1000268201_693b08cb0e little girl climbing into wooden playhouse
>1000268201_693b08cb0e little girl climbing the stairs to her playhouse
>1000268201_693b08cb0e little girl in pink dress going into wooden cabin
>1001773457_577c3a7d70 black dog and spotted dog are fighting
>1001773457_577c3a7d70 black dog and tricolored dog playing with each other on the road
>1001773457_577c3a7d70 black dog and white dog with brown spots are staring at each other in the street
>1001773457_577c3a7d70 two dogs of different breeds looking at each other on the road
>1001773457_577c3a7d70 two dogs on pavement moving toward each other
>1002674143_1b742ab4b8 little girl covered in paint sits in front of painted rainbow with her hands in bowl
>1002674143_1b742ab4b8 little girl is sitting in front of large painted rainbow
>...

## Step3 构建模型并训练模型
