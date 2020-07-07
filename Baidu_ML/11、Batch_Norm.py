import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import BatchNorm
data = np.array([[1,2,3],[4,5,6],[7,8,9]]).astype('float32')
with fluid.dygraph.guard():
    bn = BatchNorm(num_channels=3)
    x = fluid.dygraph.to_variable(data)
    y = bn(x)
    print("output of BatchNorm Layer:\n{}".format(y.numpy()))