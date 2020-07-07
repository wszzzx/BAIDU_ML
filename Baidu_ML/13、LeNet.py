import paddle
import paddle.fluid as fluid
import numpy as np
from paddle.fluid.dygraph.nn import Conv2D,Pool2D,Linear

import os
import random


class LeNet(fluid.dygraph.Layer):
    def __init__(self,num_classes = 1):
        super(LeNet, self).__init__()
        self.conv1 = Conv2D(num_channels=1,num_filters=6,filter_size=5,act='sigmoid')
        self.pool1 = Pool2D(pool_size=2,pool_stride=2,pool_type='max')
        self.conv2 = Conv2D(num_channels=6,num_filters=16,filter_size=5,act='sigmoid')
        self.pool2 = Pool2D(pool_size=2,pool_stride=2,pool_type='max')
        self.conv3 = Conv2D(num_channels=16,num_filters=120,filter_size=4,act='sigmoid')

        self.fc1 = Linear(input_dim=120,output_dim=64,act='sigmoid')
        self.fc2 = Linear(input_dim=64,output_dim=num_classes)

    def forward(self, input,label = None):
        conv1 = self.conv1(input)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        conv3_reshape = fluid.layers.reshape(conv3,[conv3.shape[0],-1])
        fc1 = self.fc1(conv3_reshape)
        fc2 = self.fc2(fc1)
        if label is not None:
            acc = fluid.layers.accuracy(input=fc2, label=label)
            return fc2, acc
        else:
            return fc2
        # return fc2
#### 查看 LeNet
# x = np.random.randn(3,1,28,28).astype('float32')
# with fluid.dygraph.guard():
#     m = LeNet(num_classes=10)
#     print(m.sublayers())
#     x = fluid.dygraph.to_variable(x)
#     for item in m.sublayers():
#         try:
#             x = item(x)
#         except:
#             x = fluid.layers.reshape(x,[x.shape[0],-1])
#             x = item(x)
#         print("len",len(item.parameters()))
#         if len(item.parameters())==2:
#             print(item.full_name(),x.shape,item.parameters()[0].shape,item.parameters()[1].shape)
#         else:
#             print(item.full_name(),x.shape)

def train(model):
    print('start training')
    model.train()
    epoch_num = 5
    # opt = fluid.optimizer.Momentum(learning_rate=0.001,momentum=0.9,parameter_list=model.parameters())
    # opt = fluid.optimizer.AdamOptimizer(learning_rate=0.01, parameter_list=model.parameters(),
    #                                           regularization=fluid.regularizer.L2Decay(regularization_coeff=0.1))
    # opt = fluid.optimizer.SGDOptimizer(learning_rate=0.1, parameter_list=model.parameters(),)
    opt = fluid.optimizer.AdamOptimizer(learning_rate=0.1, parameter_list=model.parameters(), )
    train_loader = paddle.batch(paddle.dataset.mnist.train(),batch_size=10)
    valid_loader = paddle.batch(paddle.dataset.mnist.test(),batch_size=10)
    for epoch in range(epoch_num):
        for batch_id,data in enumerate(train_loader()):
            x_data = np.array([item[0] for item in data],dtype='float32').reshape(-1,1,28,28)
            y_data = np.array([item[1] for item in data],dtype='int64').reshape(-1,1)
            img = fluid.dygraph.to_variable(x_data)
            label = fluid.dygraph.to_variable(y_data)
            logits = model.forward(img)
            loss = fluid.layers.softmax_with_cross_entropy(logits,label)
            avg_loss = fluid.layers.mean(loss)
            if batch_id%1000 == 0:
                print("epoch:{},batch_id:{},loss is :{}".format(epoch,batch_id,avg_loss.numpy()))
            avg_loss.backward()
            opt.minimize(avg_loss)
            model.clear_gradients()
        model.eval()
        accuracies = []
        losses = []
        for batch_id, data in enumerate(valid_loader()):
            # 调整输入数据形状和类型
            x_data = np.array([item[0] for item in data], dtype='float32').reshape(-1, 1, 28, 28)
            y_data = np.array([item[1] for item in data], dtype='int64').reshape(-1, 1)
            # 将numpy.ndarray转化成Tensor
            img = fluid.dygraph.to_variable(x_data)
            label = fluid.dygraph.to_variable(y_data)
            # 计算模型输出
            logits = model(img)
            pred = fluid.layers.softmax(logits)
            # 计算损失函数
            loss = fluid.layers.softmax_with_cross_entropy(logits, label)
            acc = fluid.layers.accuracy(pred, label)
            accuracies.append(acc.numpy())
            losses.append(loss.numpy())
        print("[validation] accuracy/loss: {}/{}".format(np.mean(accuracies), np.mean(losses)))
        model.train()
    fluid.save_dygraph(model.state_dict(),'mnist_LeNet')
# def train(model):
#     print('start training ... ')
#     model.train()
#     epoch_num = 5
#     opt = fluid.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameter_list=model.parameters())
#     # 使用Paddle自带的数据读取器
#     train_loader = paddle.batch(paddle.dataset.mnist.train(), batch_size=10)
#     valid_loader = paddle.batch(paddle.dataset.mnist.test(), batch_size=10)
#     for epoch in range(epoch_num):
#         for batch_id, data in enumerate(train_loader()):
#             # 调整输入数据形状和类型
#             x_data = np.array([item[0] for item in data], dtype='float32').reshape(-1, 1, 28, 28)
#             y_data = np.array([item[1] for item in data], dtype='int64').reshape(-1, 1)
#             # 将numpy.ndarray转化成Tensor
#             img = fluid.dygraph.to_variable(x_data)
#             label = fluid.dygraph.to_variable(y_data)
#             # 计算模型输出
#             logits = model(img)
#             # 计算损失函数
#             loss = fluid.layers.softmax_with_cross_entropy(logits, label)
#             avg_loss = fluid.layers.mean(loss)
#             if batch_id % 1000 == 0:
#                 print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, avg_loss.numpy()))
#             avg_loss.backward()
#             opt.minimize(avg_loss)
#             model.clear_gradients()
#
#         model.eval()
#         accuracies = []
#         losses = []
#         for batch_id, data in enumerate(valid_loader()):
#             # 调整输入数据形状和类型
#             x_data = np.array([item[0] for item in data], dtype='float32').reshape(-1, 1, 28, 28)
#             y_data = np.array([item[1] for item in data], dtype='int64').reshape(-1, 1)
#             # 将numpy.ndarray转化成Tensor
#             img = fluid.dygraph.to_variable(x_data)
#             label = fluid.dygraph.to_variable(y_data)
#             # 计算模型输出
#             logits = model(img)
#             pred = fluid.layers.softmax(logits)
#             # 计算损失函数
#             loss = fluid.layers.softmax_with_cross_entropy(logits, label)
#             acc = fluid.layers.accuracy(pred, label)
#             accuracies.append(acc.numpy())
#             losses.append(loss.numpy())
#         print("[validation] accuracy/loss: {}/{}".format(np.mean(accuracies), np.mean(losses)))
#         model.train()
#
#     # 保存模型参数
#     fluid.save_dygraph(model.state_dict(), 'mnist')

if __name__ == '__main__':
    with fluid.dygraph.guard():
        model = LeNet(num_classes=10)
        train(model)





