import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Linear,Conv2D,Pool2D
# from paddle.fluid.dygraph.nn import FC
import numpy as np
import os
import gzip
import json
import random


def load_data(mode='train'):
    # 数据文件
    datafile = 'mnist.json.gz'
    print('loading mnist dataset from {} ......'.format(datafile))
    data = json.load(gzip.open(datafile))
    # 读取到的数据可以直接区分训练集，验证集，测试集
    train_set, val_set, eval_set = data

    # 数据集相关参数，图片高度IMG_ROWS, 图片宽度IMG_COLS
    IMG_ROWS = 28
    IMG_COLS = 28
    # 获得数据
    if mode == 'train':
        imgs = train_set[0]
        labels = train_set[1]
    elif mode == 'valid':
        imgs = val_set[0]
        labels = val_set[1]
    elif mode == 'eval':
        imgs = eval_set[0]
        labels = eval_set[1]
    else:
        raise Exception("mode can only be one of ['train', 'valid', 'eval']")

    imgs_length = len(imgs)

    assert len(imgs) == len(labels), \
        "length of train_imgs({}) should be the same as train_labels({})".format(
            len(imgs), len(labels))

    index_list = list(range(imgs_length))

    # 读入数据时用到的batchsize
    BATCHSIZE = 100

    # 定义数据生成器
    def data_generator():
        if mode == 'train':
            # 训练模式下，将训练数据打乱
            random.shuffle(index_list)
        imgs_list = []
        labels_list = []

        for i in index_list:
            img = np.reshape(imgs[i], [1, IMG_ROWS, IMG_COLS]).astype('float32')
            # img = np.array(imgs[i]).astype('float32')

            label = np.reshape(labels[i], [1]).astype('float32')
            imgs_list.append(img)
            labels_list.append(label)
            if len(imgs_list) == BATCHSIZE:
                # 产生一个batch的数据并返回
                yield np.array(imgs_list), np.array(labels_list)
                # 清空数据读取列表
                imgs_list = []
                labels_list = []

        # 如果剩余数据的数目小于BATCHSIZE，
        # 则剩余数据一起构成一个大小为len(imgs_list)的mini-batch
        if len(imgs_list) > 0:
            yield np.array(imgs_list), np.array(labels_list)

    return data_generator

class MNIST(fluid.dygraph.Layer):
    def __init__(self,name_scope):
        super(MNIST,self).__init__(name_scope)
        name_scope = self.full_name()
        # 定义卷积层，输出特征通道num_filters设置为20，卷积核的大小filter_size为5，卷积步长stride=1，padding=2
        # 激活函数使用relu
        self.conv1 = Conv2D(num_channels=1,num_filters=20, filter_size=5, stride=1, padding=2, act='relu')
        # 定义池化层，池化核pool_size=2，池化步长为2，选择最大池化方式
        self.pool1 = Pool2D( pool_size=2, pool_stride=2, pool_type='max')
        # 定义卷积层，输出特征通道num_filters设置为20，卷积核的大小filter_size为5，卷积步长stride=1，padding=2
        self.conv2 = Conv2D(num_channels=20, num_filters=20, filter_size=5, stride=1, padding=2, act='relu')
        # 定义池化层，池化核pool_size=2，池化步长为2，选择最大池化方式
        self.pool2 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
        # 定义一层全连接层，输出维度是1，不使用激活函数
        self.linear = Linear(20*7*7, 1, act=None)
    def forward(self, inputs):
         x = self.conv1(inputs)
         x = self.pool1(x)
         x = self.conv2(x)
         x = self.pool2(x)
         x = x.numpy()
         x = np.reshape(x,[100,-1])
         x = fluid.dygraph.to_variable(x)
         # x = fluid.layers.reshape(x,shape=[100.-1])
         x = self.linear(x)
         return x

# 异步处理需要添加三行代码

#异步代码一
place = fluid.CPUPlace()

with fluid.dygraph.guard(place):
    model = MNIST("mnist")
    model.train()
    train_loader = load_data('train')

    # 异步代码二 定义DataLoader对象用于加载Python生成器产生的数据
    data_loader = fluid.io.DataLoader.from_generator(capacity=5, return_list=True)
    # 异步代码三 设置数据生成器
    data_loader.set_batch_generator(train_loader, places=place)

    # linears= Linear(784,1,dtype='float32')
    optimizer=fluid.optimizer.SGDOptimizer(learning_rate=0.001,parameter_list=model.parameters())
    EPOCH_NUM = 10
    for epoch_id in range(EPOCH_NUM):
        # for batch_id,data in enumerate(data_generator("train")):
        for batch_id,data in enumerate(train_loader()):

            image_data,label_data = data
            # print("shape",image_data.shape,label_data.shape)
            image = fluid.dygraph.to_variable(image_data)
            label = fluid.dygraph.to_variable(label_data)
            predict = model.forward(image)
            loss = fluid.layers.square_error_cost(predict,label)
            avg_loss = fluid.layers.mean(loss)
            if batch_id%200 ==0:
                print("epoch:{},batch:{},loss is :{}".format(epoch_id,batch_id,avg_loss.numpy()))
            avg_loss.backward()
            optimizer.minimize(avg_loss)
            model.clear_gradients()
    fluid.save_dygraph(model.state_dict(),'mnist')
