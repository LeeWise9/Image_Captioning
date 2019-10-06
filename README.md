# Image_Captioning
This is a neural network project. The expected function is to generate descriptions for pictures. That is, to provide a picture, the neural network can automatically generate a paragraph of text for the picture to describe the content of it. The project is based on LSTM and CNN, and uses VGG16 to extract image features.

这是一个神经网络项目，期望实现看图说话功能，即给定一张图片，神经网络能够生成一段文字来描述图片内容，讲述图片中的故事。这个项目以LSTM和CNN为基础，并且使用VGG16模型来提取图片特征。

看图说话是典型的端到端学习，输入端为图片特征和一部分文字（作为引子），期望得到的输出则是图片对应的内容描述。所以要求数据集中既包含图片，又包含对应的描述。该项目数据集来自Flickr8k_Dataset，读者可以在kaggle搜索下载：[Kaggle_Flickr8k](https://www.kaggle.com/shadabhussain/flickr8k)。

本项目主要包含四个部分：1.使用VGG16提取图片特征并保存为文件；2.预处理数据集中的描述文本并保存为文件；3.构建模型并训练；4.评估模型并为图片生成描述。

## Step1 提取图片特征<br>
本项目使用预先训练好的VGG16来提取图片特征。为了适配该项目，VGG16模型最后一层被去掉，输出为一个1×4096的向量。<br>
![VGG16](https://image.baidu.com/search/detail?ct=503316480&z=0&ipn=d&word=vgg16&step_word=&hs=0&pn=3&spn=0&di=7150&pi=0&rn=1&tn=baiduimagedetail&is=0%2C0&istype=0&ie=utf-8&oe=utf-8&in=&cl=2&lm=-1&st=undefined&cs=3696184893%2C61316082&os=2494690748%2C1936766418&simid=0%2C0&adpicid=0&lpn=0&ln=91&fr=&fmq=1570374575871_R&fm=&ic=undefined&s=undefined&hd=undefined&latest=undefined&copyright=undefined&se=&sme=&tab=0&width=undefined&height=undefined&face=undefined&ist=&jit=&cg=&bdtype=0&oriquery=&objurl=http%3A%2F%2Fws1.sinaimg.cn%2Flarge%2F662f5c1fgy1frnjdmk4n5j21710pc46d.jpg&fromurl=ippr_z2C%24qAzdH3FAzdH3Fks52_z%26e3Bvf1g_z%26e3BgjpAzdH3FoitsjaAzdH3Fw6ptvsjAzdH3F1jpwtsfAzdH3Fblnmadbm&gsm=&rpstart=0&rpnum=0&islist=&querylist=&force=undefined"VGG16")
