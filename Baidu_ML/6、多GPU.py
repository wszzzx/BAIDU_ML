import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Linear,Conv2D,Pool2D
# from paddle.fluid.dygraph.nn import FC
import numpy as np
import os
import gzip
import json
import random
import matplotlib.pyplot as plt


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
# place = fluid.CPUPlace()

################## CPU or 单个GPU
# place = fluid.CPUPlace()
# place = fluid.CUDAPlace(0)
# use_gpu = False
# place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()


##################分布式训练  修改四个地方

#修改1 -----从环境变量获取设备的ID，并指定给CUDAPlace
device_id = fluid.dygraph.parallel.Env().dev_id
place = fluid.CUDAPlace(device_id)

with fluid.dygraph.guard(place):
    model1 = MNIST("mnist")
    model2 = MNIST("mnist")
    model3 = MNIST("mnist")
    model4 = MNIST("mnist")
    model5 = MNIST("mnist")
    model6 = MNIST("mnist")
    model7 = MNIST("mnist")
    model8 = MNIST("mnist")
    model9 = MNIST("mnist")
    model10 = MNIST("mnist")
    model11 = MNIST("mnist")
    model12 = MNIST("mnist")


    optimizer1_1=fluid.optimizer.SGDOptimizer(learning_rate=0.1,parameter_list=model1.parameters())
    optimizer1_2 = fluid.optimizer.SGDOptimizer(learning_rate=0.01, parameter_list=model2.parameters())
    optimizer1_3 = fluid.optimizer.SGDOptimizer(learning_rate=0.001, parameter_list=model3.parameters())

    optimizer2_1=fluid.optimizer.MomentumOptimizer(learning_rate=0.1,momentum = 0.9,parameter_list=model4.parameters())
    optimizer2_2 = fluid.optimizer.MomentumOptimizer(learning_rate=0.01, momentum = 0.9,parameter_list=model5.parameters())
    optimizer2_3 = fluid.optimizer.MomentumOptimizer(learning_rate=0.001, momentum = 0.9,parameter_list=model6.parameters())


    optimizer3_1=fluid.optimizer.AdagradOptimizer(learning_rate=0.1,parameter_list=model7.parameters())
    optimizer3_2 = fluid.optimizer.AdagradOptimizer(learning_rate=0.01, parameter_list=model8.parameters())
    optimizer3_3 = fluid.optimizer.AdagradOptimizer(learning_rate=0.001, parameter_list=model9.parameters())

    optimizer4_1=fluid.optimizer.AdamOptimizer(learning_rate=0.1,parameter_list=model10.parameters())
    optimizer4_2 = fluid.optimizer.AdamOptimizer(learning_rate=0.01, parameter_list=model11.parameters())
    optimizer4_3 = fluid.optimizer.AdamOptimizer(learning_rate=0.001, parameter_list=model12.parameters())

    optimizers = [optimizer1_1,optimizer1_2,optimizer1_3,optimizer2_1,optimizer2_2,optimizer2_3,optimizer3_1,optimizer3_2,optimizer3_3,optimizer4_1,optimizer4_2,optimizer4_3,]
    models = [model1,model2,model3,model4,model5,model6,model7,model8,model9,model10,model11,model12,]
    avg_loss_lists = []
    for optimizer in optimizers:
        index = optimizers.index(optimizer)
        avg_loss_list = []

        ##修改2-对原模型做并行化预处理
        strategy = fluid.dygraph.parallel.prepare_context()
        model = models[index]
        model = fluid.dygraph.parallel.DataParallel(model,strategy)
        model.train()

        ###############修改3，多GPU数据读取，必须确保每个进程读取的数据是不同的

        train_loader = load_data('train')
        train_loader = fluid.contrib.reader.distributed_batch_reader(train_loader)

        # 异步代码二 定义DataLoader对象用于加载Python生成器产生的数据
        data_loader = fluid.io.DataLoader.from_generator(capacity=5, return_list=True)
        # 异步代码三 设置数据生成器
        data_loader.set_batch_generator(train_loader, places=place)

        EPOCH_NUM = 10
        for epoch_id in range(EPOCH_NUM):
            # for batch_id,data in enumerate(data_generator("train")):
            for batch_id,data in enumerate(train_loader()):

                image_data,label_data = data
                # print("shape",image_data.shape,label_data.shape)
                image = fluid.dygraph.to_variable(image_data)
                label = fluid.dygraph.to_variable(label_data)
                predict = model.forward(image)
                # loss = fluid.layers.square_error_cost(predict,label)
                loss = fluid.layers.cross_entropy(predict,label)
                avg_loss = fluid.layers.mean(loss)
                # 修改4-多GPU训练需要对Loss做出调整，并聚合不同设备上的参数梯度
                avg_loss = mnist.
                if batch_id%200 ==0:
                    print("index:{},epoch:{},batch:{},loss is :{}".format(index,epoch_id,batch_id,avg_loss.numpy()))
                    avg_loss_list.append(avg_loss.numpy()[0])
                avg_loss.backward()
                optimizer.minimize(avg_loss)
                model.clear_gradients()
        fluid.save_dygraph(model.state_dict(),'mnist' + '_' + str(index+1))
        avg_loss_lists.append(avg_loss_list)

for i in range(len(avg_loss_lists)):
    print(i,avg_loss_lists[i])
    plt.plot(avg_loss_lists[i],label = str(i))
plt.show()