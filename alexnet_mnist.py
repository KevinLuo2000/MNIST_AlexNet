import tensorflow as tf
# 输入数据,mnist.py我拷贝到了此py文件所在的目录
from mnist import read_data_sets
# ssl：certificate_verify_failed
# Solution：全局取消证书验证（当项目对安全性问题不太重视时，推荐使用，可以全局取消证书的验证，简易方便）
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
mnist = read_data_sets("MNIST_data", one_hot=True)
# 定义网络的一参数
learning_rate = 0.001
training_iters = 200000
batch_size = 64
display_step = 10
# 定义网络的参数
n_input = 784  # 输入的维度(img shape: 28×28)
n_classes = 10  # 标记的维度 (0-9 digits)
dropout = 0.9  # Dropout 的概率，输出的可能性
# 输入占位符
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)  # dropout


# 卷积
def conv2d(name, x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x, name=name)  # 使用relu 激活函数


# 池化
def maxpool2d(name, x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, 1, 1, 1],
                          padding='SAME', name=name)


# LRN层--去掉不用
# def norm(name, l_input, lsize=4):
#    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0,
#                     beta=0.75, name=name)
# 定义所有的网络参数
weights = {
    'wc1': tf.Variable(tf.random_normal([3, 3, 1, 64])),
    'wc2': tf.Variable(tf.random_normal([3, 3, 64, 64])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 64, 128])),
    'wc4': tf.Variable(tf.random_normal([3, 3, 128, 128])),
    'wc5': tf.Variable(tf.random_normal([3, 3, 128, 256])),
    'wd1': tf.Variable(tf.random_normal([28 * 28 * 256, 1024])),
    'wd2': tf.Variable(tf.random_normal([1024, 1024])),
    'out': tf.Variable(tf.random_normal([1024, 10]))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([64])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bc3': tf.Variable(tf.random_normal([128])),
    'bc4': tf.Variable(tf.random_normal([128])),
    'bc5': tf.Variable(tf.random_normal([256])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'bd2': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# 定义AlexNet 的网络模型：
# 定义整个网络
def alex_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    # 第一层卷积
    # 卷积
    conv1 = conv2d('conv1', x, weights['wc1'], biases['bc1'])
    # 下采样
    pool1 = maxpool2d('pool1', conv1, k=2)
    # 规范化
    # norm1 = norm('norm1', pool1, lsize=4)
    # 第二层卷积
    # 卷积
    conv2 = conv2d('conv2', pool1, weights['wc2'], biases['bc2'])
    # 最大池化（向下采样）
    pool2 = maxpool2d('pool2', conv2, k=2)
    # 规范化
    # norm2 = norm('norm2', pool2, lsize=4)
    # 第三层卷积
    # 卷积
    conv3 = conv2d('conv3', pool2, weights['wc3'], biases['bc3'])
    # 下采样
    # pool3 = maxpool2d('pool3', conv3, k=2)
    # 规范化
    # norm3 = norm('norm3', pool3, lsize=4)
    # 第四层卷积
    conv4 = conv2d('conv4', conv3, weights['wc4'], biases['bc4'])
    # 第五层卷积
    conv5 = conv2d('conv5', conv4, weights['wc5'], biases['bc5'])
    # 下采样
    pool5 = maxpool2d('pool5', conv5, k=2)
    # 规范化
    # norm5 = norm('norm5', pool5, lsize=4)
    # 全连接层1
    # shape = pool5.get_shape()
    fc1 = tf.reshape(pool5, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # dropout
    fc1 = tf.nn.dropout(fc1, dropout)
    # 全连接层2
    fc2 = tf.reshape(fc1, [-1, weights['wd2'].get_shape().as_list()[0]])
    fc2 = tf.add(tf.matmul(fc2, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2)
    # dropout
    fc2 = tf.nn.dropout(fc2, dropout)
    # 输出层
    out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    return out


# 构建模型，定义损失函数和优化器，并构建评估函数：
# 构建模型
pred = alex_net(x, weights, biases, keep_prob)
# 定义损失函数和优化器
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
# pred
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# 评估函数
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# 训练模型和评估模型的代码如下：
# 初始化变量
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # 开始训练，直到达到training_iters，即200000
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # 计算损失值和准确度，输出
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " +
                  "{:.6f}".format(loss) + ", Training Accuracy= " +
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

# 计算测试集的准确度
    print("Testing Accuracy:",
          sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256],
                                        keep_prob: 1.}))
