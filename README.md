# Image_Captioning 看图说话

这是一个神经网络项目，期望实现看图说话功能，即给定一张图片，神经网络能够生成一段文字来描述图片内容，讲述图片中的故事。这个项目以LSTM和CNN为基础，并且使用VGG16模型来提取图片特征。

本项目基于keras编写，上手简单，支持GPU加速。

看图说话是典型的端到端学习，输入端为图片特征（Step3中会详细讲解），期望得到的输出则是图片对应的内容描述。所以要求数据集中既包含图片，又包含对应的描述用于训练。该项目数据集来自Flickr8k_Dataset，读者可以在kaggle搜索下载：[Kaggle_Flickr8k](https://www.kaggle.com/shadabhussain/flickr8k)。

本项目主要包含四个部分：<br>
* 1.使用VGG16提取图片特征并保存为文件；<br>
* 2.预处理数据集中的描述文本并保存为文件；<br>
* 3.构建模型并训练；<br>
* 4.评估模型并为图片生成描述。<br>

## Step1 提取图片特征<br>
本项目使用预先训练好的VGG16来提取图片特征。为了适配该项目，至少要注意两点面：1.要对图片做预处理，包括图片缩放、增维和去均值化；2.去掉VGG16的最后一层（1000的全连接层和softmax激活层），使输出为一个1维的长度为4096的向量。<br>
<p align="center">
	<img src="https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1570385062326&di=c2aaf62c394a72b2a97b1793d9b2de26&imgtype=0&src=http%3A%2F%2Fws1.sinaimg.cn%2Flarge%2F662f5c1fgy1frnjdmk4n5j21710pc46d.jpg" alt="Sample"  width="500">
</p>

提取到的特征可以用字典形式储存。当目标文件夹中的所有图片都完成了特征提取操作之后，结果将被保存为.pkl文件，以方便后续调用。<br>
除了使用VGG16，读者还可以尝试使用其他的预训练网络进行尝试，比如ResNet50等。

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

## Step3 构建模型并训练<br>
网络包含两部分输入：图片特征和描述文本（单词串）。<br>
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84.jpg" alt="Sample"  width="500">
</p>

你可能难以理解为什么要将描述文本作为输入的一部分，为什么不直接把图片特征作为X_train，图片描述作为y_train。原因是这样预测效果并不好。描述文本中的单词包含着内在的先后顺序，但是图片特征不包含。要想让神经网络看到图片“说人话”，还得考虑使用LSTM处理一下描述文本，加入训练。那么图片描述是不是既要作为X_train又要作为y_train呢？这需要你先了解LSTM的工作原理。

通常来说，LSTM的输入不能为空，且在获取输入之后，每一次都只输出一个单词。为了让输入不为空，考虑在文本首端添加标识符“startseq”，为了使LSTM在适当的时侯停止输出，输入的描述文本需要添加尾端标识符“endseq”。为了让神经网络习得语言顺序，需要构建上下文结构，即拆分句子。不算首尾标识符，一段包含n个单词的句子需要构建n+1对上下文结构，如下表所示。<br>
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/%E4%B8%8A%E4%B8%8B%E6%96%87%E7%BB%93%E6%9E%84.jpg" alt="Sample"  width="500">
</p>

任何一段句子，其第一个输入词一定是“startseq”，再陆续输入每个单词。上述句子“a cat sits on the table”，按照上下文规则分别进行了7次输入和输出，最后一次的输出为“endseq”。这样不会造成数据泄露，出现在输入中的词汇一定是在输出端作为标签先出现的。比如“startseq a cat sits”这段输入中的“sits”这个单词，上一轮的output中已经作为y_train出现过了。

上述部分是为了解释了为什么将描述文本作为输入。下面讲解如何合并图片特征和图片描述。先来看一下网络结构，如下图所示。<br>
<p align="center">
	<img src="https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2017/09/Plot-of-the-Caption-Generation-Deep-Learning-Model.png" alt="Sample"  width="500">
</p>

输入的左支（input_2）为描述文本，输入长度为34，因为最长的句子包含了34个单词，而不足34词的句子可以补零，所以统一长度为34。输入的右支（input_1）为图片特征，由VGG16提取得到，长度为4096。为了融合两支，需要先统一输入的长度。左支采用嵌入层加LSTM方法：先对输入词做增维，再输入到LSTM，使输出长度为256，其中还使用了Droupout方法避免过拟合。右支使用全连接层直接将输出长度减少到256，同样使用了Droupout方法避免过拟合。这样两支的输出可以直接使用add相加，再做后续操作。读者们可以根据实际情况，自行设计规模更大、参数更多，结构更复杂的模型。

实际训练中，需要加载大量数据，为了减少内存开销，本项目采取逐步加载训练数据的方法。如果读者的计算机有32GB及以上的内存，可以考虑将代码改写为一次性读入所有数据，这样可以大幅提升训练速度减少训练时间。

模型的训练结果以.h5文件形式保存，方便后续调用。

## Step4.评估模型并为图片生成描述<br>
常规的评价模型性能的方法很多，对于语言模型，常用的评估方法为[BLEU](https://blog.csdn.net/allocator/article/details/79657792)。BLEU是一个非常简单快速粗略的评估指标, 最初被用于评价机器翻译效果，当面对多个翻译模型且需要快速选择模型的场景, 可以使用这个指标来评估模型的好坏。BLEU的本质是计算两段的相似程度，可以用来评估看图说话模型的好坏。其评价值介于0到1之间，值越接近于1则两个句子越相似。

另一种评估方法是对一张图片生成一段描述，让读者主观的判断生成的句子是否合理。

比如对于下图：<br>
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/example.jpg" alt="Sample"  width="500">
</p>

模型的输出为：“dog is running through the water”。<br>
你觉得怎么样呢？
