import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Linear,Conv2D,Pool2D
# from paddle.fluid.dygraph.nn import FC
import numpy as np
import os
import gzip
import json
import random
from tb_paddle import SummaryWriter
data_writer = SummaryWriter(logdir='log/data')

#计算分类准确率，观察模型训练结果
#检查模型训练过程，识别潜在问题
#加入校验和测试，更好评价模型效果
#加入正则化，避免模型过拟合
#可视化分析

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

            # label = np.reshape(labels[i], [1]).astype('float32')
            label = np.reshape(labels[i],[1]).astype('int64')
            print("label",label.shape,label)
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

        # self.linear = Linear(20*7*7, 1, act=None)
        self.linear = Linear(20*7*7, 10, act='softmax')

    def forward(self, inputs,label = None,check_shape = False,check_content=False):
         outputs1 = self.conv1(inputs)
         outputs2 = self.pool1(outputs1)
         outputs3 = self.conv2(outputs2)
         outputs4 = self.pool2(outputs3)
         outputs4 = outputs4.numpy()
         outputs4 = np.reshape(outputs4,[100,-1])
         outputs4 = fluid.dygraph.to_variable(outputs4)
         outputs5 = self.linear(outputs4)

         if check_shape:
             # 打印每层网络设置的超参数-卷积核尺寸，卷积步长，卷积padding，池化核尺寸
             print("\n########## print network layer's superparams ##############")
             print("\n########## print network layer's superparams ##############")
             print("conv1-- kernel_size:{}, padding:{}, stride:{}".format(self.conv1.weight.shape, self.conv1._padding, self.conv1._stride))
             print("conv2-- kernel_size:{}, padding:{}, stride:{}".format(self.conv2.weight.shape, self.conv2._padding, self.conv2._stride))
             print("pool1-- pool_type:{}, pool_size:{}, pool_stride:{}".format(self.pool1._pool_type, self.pool1._pool_size, self.pool1._pool_stride))
             print("pool2-- pool_type:{}, poo2_size:{}, pool_stride:{}".format(self.pool2._pool_type, self.pool2._pool_size, self.pool2._pool_stride))
             print("fc-- weight_size:{}, bias_size_{}, activation:{}".format(self.linear.weight.shape, self.linear.bias.shape, self.linear._act))

             # 打印每层的输出尺寸
             print("\n########## print shape of features of every layer ###############")
             print("inputs_shape: {}".format(inputs.shape))
             print("outputs1_shape: {}".format(outputs1.shape))
             print("outputs2_shape: {}".format(outputs2.shape))
             print("outputs3_shape: {}".format(outputs3.shape))
             print("outputs4_shape: {}".format(outputs4.shape))
             print("outputs5_shape: {}".format(outputs5.shape))
         # 选择是否打印训练过程中的参数和输出内容，可用于训练过程中的调试
         if check_content:
             print("\n########## print convolution layer's kernel ###############")
             print("conv1 params -- kernel weights:", self.conv1.weight[0][0])
             print("conv2 params -- kernel weights:", self.conv2.weight[0][0])

             # 创建随机数，随机打印某一个通道的输出值
             idx1 = np.random.randint(0, outputs1.shape[1])
             idx2 = np.random.randint(0, outputs3.shape[1])
             # 打印卷积-池化后的结果，仅打印batch中第一个图像对应的特征
             print("\nThe {}th channel of conv1 layer: ".format(idx1), outputs1[0][idx1])
             print("The {}th channel of conv2 layer: ".format(idx2), outputs3[0][idx2])
             print("The output of last layer:", outputs5[0], '\n')


         if label is not None:
             acc = fluid.layers.accuracy(input=outputs5,label=label)
             return outputs5,acc
         else:
            return outputs5

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
    optimizer=fluid.optimizer.SGDOptimizer(learning_rate=0.01,parameter_list=model.parameters(),regularization=fluid.regularizer.L2Decay(regularization_coeff=0.1))
    EPOCH_NUM = 10
    iter = 0
    for epoch_id in range(EPOCH_NUM):
        # for batch_id,data in enumerate(data_generator("train")):
        for batch_id,data in enumerate(train_loader()):

            image_data,label_data = data
            # print("shape",image_data.shape,label_data.shape)
            image = fluid.dygraph.to_variable(image_data)
            label = fluid.dygraph.to_variable(label_data)

            if batch_id == 0 and epoch_id==0:
                predict, avg_acc = model.forward(image, label,check_shape=True,check_content=False)
            elif batch_id == 401:
                predict, avg_acc = model.forward(image, label, check_shape=False, check_content=True)
            else:
                predict, avg_acc = model.forward(image, label,)

            # loss = fluid.layers.square_error_cost(predict,label)
            loss = fluid.layers.cross_entropy(predict,label)
            avg_loss = fluid.layers.mean(loss)
            if batch_id%100 ==0:
                print("epoch:{},batch:{},loss is :{},acc:{}".format(epoch_id,batch_id,avg_loss.numpy(),avg_acc.numpy()))
                data_writer.add_scalar("train/loss",avg_loss.numpy(),iter)
                data_writer.add_scalar("train/accuracy",avg_acc.numpy(),iter)
                iter +=100
            avg_loss.backward()
            optimizer.minimize(avg_loss)
            model.clear_gradients()
    fluid.save_dygraph(model.state_dict(),'mnist')
