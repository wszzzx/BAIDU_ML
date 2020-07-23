import paddle.fluid as fluid
# #创建变量
# x = fluid.data(name='x',shape=[3,None],dtype='int64')
# batched_x = fluid.data(name="batch_x",shape=[None,3,None],dtype='int64')
# #创建常量
# data = fluid.layers.fill_constant(shape=[3,4],value=16,dtype='int64')
# # print(data)
# # data = fluid.layers.Print(data,message='Print data:')
# # place = fluid.CPUPlace()
# # exe = fluid.Executor(place)
# # exe.run(fluid.default_startup_program())
# # exe.run()
# a = fluid.data(name='a',shape=[None,1],dtype='int64')
# b = fluid.data(name='b',shape=[None,1],dtype='int64')
# result = fluid.layers.elementwise_add(a,b)
# cpu = fluid.CPUPlace()
# exe = fluid.Executor(cpu)
# exe.run(fluid.default_startup_program())
#
# import numpy as np
# # data_1 = int(1).astype('int64')
# # data_2 = int(2).astype('int64')
# data_1 = np.int64(input("Please enter an integer: a="))
# data_2 = np.int64(input("Please enter an integer: b="))
# x = np.array([[data_1]])
# y = np.array([[data_2]])
# outs = exe.run(feed={'a':x,'b':y},fetch_list=[a,b,result])
# print("outs",outs)

import numpy

train_data=numpy.array([[1.0], [2.0], [3.0], [4.0]]).astype('float32')
# 定义期望预测的真实值y_true
y_true = numpy.array([[2.0], [4.0], [6.0], [8.0]]).astype('float32')

x = fluid.data(name='x',shape=[None,1],dtype='float32')
y = fluid.data(name='y',shape=[None,1],dtype='float32')
y_predict = fluid.layers.fc(input=x,size=1,act=None)

cost = fluid.layers.square_error_cost(input=y_predict,label=y)
avg_cost = fluid.layers.mean(cost)

sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.01)
sgd_optimizer.minimize(avg_cost)

cpu = fluid.CPUPlace()
exe = fluid.Executor(cpu)
exe.run(fluid.default_startup_program())

for i in range(100):
    outs = exe.run(
        feed={'x':train_data, 'y':y_true},
        fetch_list=[y_predict, avg_cost])

# 输出训练结果
print("outs",outs)


