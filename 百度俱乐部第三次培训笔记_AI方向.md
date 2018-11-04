先把任务粘在这里，以备他日之需：

```
Task：阅读AlexNet的论文，学习深度学习相关知识，理解网络结构，并使用pytorch或tensorflow（或其他深度学习框架），使用AlexNet 完成对MNIST数据集的分类
要求：不能使用框架内的预训练模型，代码不允许copy
```

**10月27日**

[深度学习笔记目录](http://www.ai-start.com/dl2017/)

[吴恩达网易云课堂（源自Coursera）深度学习课程](https://mooc.study.163.com/university/deeplearning_ai#/c)

[TensorFlow实战：Chapter-4（CNN-2-经典卷积神经网络（AlexNet、VGGNet））](https://blog.csdn.net/WangR0120/article/details/80221098)

[深度学习系列——AlxeNet实现MNIST手写数字体识别](https://blog.csdn.net/qq_30666517/article/details/79686877)

**10月28日**

- Alexnet论文所采用的LRN事后证明并没有什么用，对准确率提升的帮助不大，还使运算时间翻倍甚至翻几倍。我就不引入了
- Alexnet论文是针对Imagenet数据集做的处理，这次的任务针对MNIST，故有一些参数需要调整

**修改后卷积层如下：**

| 连接层                  | 计算流程                       |
| ----------------------- | ------------------------------ |
| 第一卷积层              | 输入–>卷积–>ReLUs–>max-pooling |
| 第二卷积层              | 卷积–>ReLUs–>max-pooling       |
| 第三卷积层              | 卷积–>ReLUs                    |
| 第四卷积层              | 卷积–>ReLUs                    |
| 第五卷积层              | 卷积–>ReLUs                    |
| 第一全连接层            | 矩阵乘法–>ReLUs–>dropout       |
| 第二全连接层            | 矩阵乘法–>ReLUs–>dropout       |
| 第三全连接层(softmax层) | 矩阵乘法–>ReLUs–>softmax       |

**遇到的问题：**

1、ssl：certificate_verify_failed

Solution：全局取消证书验证（当项目对安全性问题不太重视时，推荐使用，可以全局取消证书的验证，简易方便）[关于python出现ssl：certificate_verify_failed问题](https://blog.csdn.net/yixieling4397/article/details/79861379)

```python
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```



2、ValueError: Only call `softmax_cross_entropy_with_logits` with named arguments (labels=..., logits=..., ...)

tensorflow中不同版本该函数写法不一样

solution：[调用tf.softmax_cross_entropy_with_logits函数出错解决](https://blog.csdn.net/caimouse/article/details/61208940)

3、***A fatal problem.*** I'm currently using MacBook Pro, but for some reasons Tensorflow has stopped supporting using GPU on apple devices. Running on CPU is extremely slow, so all I have to do is to wait, wait and wait.

Maybe some days later I'll consider renting a VPS on AWS or Google Cloud. There are probably free trials for new users, so maybe money will not become another huge boundary to me.

[Amazon AWS 的几个坑](https://blog.csdn.net/csdnhxs/article/details/80219468)

4、Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA

Solution：
Change to another version of tensorflow which corresponds to my CPU.

[Install TensorFlow with pip](https://www.tensorflow.org/install/pip)

[PyCharm调用whl方式导入本地库](https://blog.csdn.net/qq_32300143/article/details/79961307)

**11月3日**
[Tensorflow的基本概念与常用函数](http://www.cnblogs.com/focusonepoint/p/7544369.html)

收尾？

[Google Colab——用谷歌免费GPU跑你的深度学习代码](https://www.jianshu.com/p/000d2a9d36a0)

[codalab](https://worksheets.codalab.org/)