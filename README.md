# Image_Captioning 看图说话
This is a neural network project. The expected function is to generate descriptions for pictures. That is, to provide a picture, the neural network can automatically generate a paragraph of text for the picture to describe the content of it. The project is based on LSTM and CNN, and uses VGG16 to extract image features.

这是一个神经网络项目，期望实现看图说话功能，即给定一张图片，神经网络能够生成一段文字来描述图片内容，讲述图片中的故事。这个项目以LSTM和CNN为基础，并且使用VGG16模型来提取图片特征。

看图说话是典型的端到端学习，输入端为图片特征（Step3中会详细讲解），期望得到的输出则是图片对应的内容描述。所以要求数据集中既包含图片，又包含对应的描述用于训练。该项目数据集来自Flickr8k_Dataset，读者可以在kaggle搜索下载：[Kaggle_Flickr8k](https://www.kaggle.com/shadabhussain/flickr8k)。

本项目主要包含四个部分：1.使用VGG16提取图片特征并保存为文件；2.预处理数据集中的描述文本并保存为文件；3.构建模型并训练；4.评估模型并为图片生成描述。

## Step1 提取图片特征<br>
本项目使用预先训练好的VGG16来提取图片特征。为了适配该项目，至少要注意两点面：1.要对图片做预处理，包括图片缩放、增维和去均值化；2.去掉VGG16的最后一层（1000的全连接层和softmax激活层），使输出为一个1×4096的向量。

![VGG16](https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1570385062326&di=c2aaf62c394a72b2a97b1793d9b2de26&imgtype=0&src=http%3A%2F%2Fws1.sinaimg.cn%2Flarge%2F662f5c1fgy1frnjdmk4n5j21710pc46d.jpg {width=250px})

提取到的特征可以用字典形式储存。当目标文件夹中的所有图片都完成了特征提取操作之后，结果将被保存为.pkl文件，以方便后续调用。<br>
除了使用VGG16，读者还可以尝试使用其他的预训练网络进行尝试，比如ResNet50。

## Step2 描述文本预处理<br>
Flickr8k_Dataset数据集中包含图片名和对应的描述文本，用空格分割，以.txt形式保存。其中，一张图片有5段不同文字描述。<br>
>1000268201_693b08cb0e.jpg#0	A child in a pink dress is climbing up a set of stairs in an entry way .<br>
>1000268201_693b08cb0e.jpg#1	A girl going into a wooden building .<br>
>1000268201_693b08cb0e.jpg#2	A little girl climbing into a wooden playhouse .<br>
>1000268201_693b08cb0e.jpg#3	A little girl climbing the stairs to her playhouse .<br>
>1000268201_693b08cb0e.jpg#4	A little girl in a pink dress going into a wooden cabin .<br>
>1001773457_577c3a7d70.jpg#0	A black dog and a spotted dog are fighting<br>
>1001773457_577c3a7d70.jpg#1	A black dog and a tri-colored dog playing with each other on the road .<br>
>1001773457_577c3a7d70.jpg#2	A black dog and a white dog with brown spots are staring at each other in the street .<br>
>...

预处理工作需要去除图片名称尾缀，并且清洗描述文本，比如将拼写转为小写，去除停顿词、数字、单个字母等。<br>

>1000268201_693b08cb0e child in pink dress is climbing up set of stairs in an entry way<br>
>1000268201_693b08cb0e girl going into wooden building<br>
>1000268201_693b08cb0e little girl climbing into wooden playhouse<br>
>1000268201_693b08cb0e little girl climbing the stairs to her playhouse<br>
>1000268201_693b08cb0e little girl in pink dress going into wooden cabin<br>
>1001773457_577c3a7d70 black dog and spotted dog are fighting<br>
>1001773457_577c3a7d70 black dog and tricolored dog playing with each other on the road<br>
>1001773457_577c3a7d70 black dog and white dog with brown spots are staring at each other in the street<br>
>...

处理好的文本同样使用.txt文件形式保存，方便后续调用。

## Step3 构建模型并训练
网络包含两部分输入：图片特征和描述文本。<br>
 <div align="center">![网络结构](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2017/09/Schematic-of-the-Merge-Model-For-Image-Captioning.png)<br>

你可能难以理解为什么要将描述文本作为输入的一部分，为什么不直接把图片特征作为X_train，图片描述作为y_train。原因是这样预测效果并不好。描述文本中的词语包含着内在的先后顺序，但是图片特征不包含。要想让神经网络看到图片“说人话”，还得考虑使用LSTM处理一下描述文本，加入训练。

通常来说，LSTM的输入不能为空，且在获取输入之后，每一次都只输出一个单词。为了让输入不为空，考虑在文本首端添加统一标识符“startseq”，为了使LSTM在适当的时侯停止输出，输入的描述文本需要添加统一尾端标识符“endseq”。为了让神经网络习得语言顺序的精髓，需要构建上下文结构，即拆分句子。不算首尾标识符，一段包含n个单词的句子需要构建n+1对上下文结构。

![上下文结构](https://github.com/LeeWise9/Img_repositories/blob/master/%E4%B8%8A%E4%B8%8B%E6%96%87%E7%BB%93%E6%9E%84.jpg)

最终的网络结构如下图所示。<br>
![网络结构](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2017/09/Plot-of-the-Caption-Generation-Deep-Learning-Model.png)

读者们可以根据训练情况，自行设计规模更大、参数更多，结构更复杂的模型。












