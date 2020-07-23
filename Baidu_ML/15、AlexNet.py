import paddle
import paddle.fluid as fluid
import numpy as np
from paddle.fluid.dygraph.nn import Conv2D,Pool2D,Linear

class AlexNet(fluid.dygraph.Layer):
    def __init__(self,num_classes =1):
        super(AlexNet,self).__init__()

        self.conv1 = Conv2D(num_channels=3,num_filters=96,filter_size=11,stride=4,padding=5,act='relu')
        self.pool1 = Pool2D(pool_size=2,pool_stride=2,pool_type='max')
        self.conv2 = Conv2D(num_channels=96,num_filters=256,filter_size=5,stride=1,padding=2,act='relu')
        self.pool2 = Pool2D(pool_size=2,pool_stride=2,pool_type='max')
        self.conv3 = Conv2D(num_channels=256,num_filters=384,filter_size=3,stride=1,padding=1,act='relu')
        self.conv4 = Conv2D(num_channels=384, num_filters=384, filter_size=3, stride=1, padding=1, act='relu')
        self.conv5 = Conv2D(num_channels=384, num_filters=256, filter_size=3, stride=1, padding=1, act='relu')
        self.pool5 = Pool2D(pool_size=2,pool_stride=2,pool_type='max')

        self.fc1 = Linear(input_dim=12544,output_dim=4096,act='relu')
        self.drop_ratio1 = 0.5
        self.fc2 = Linear(input_dim=4096,output_dim=4096,act='relu')
        self.drop_ratio2 = 0.5
        self.fc3 = Linear(input_dim=4096,output_dim=num_classes)
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool5(x)
        x = fluid.layers.reshape(x, [x.shape[0], -1])
        x = self.fc1(x)
        x = fluid.layers.dropout(x,self.drop_ratio1)
        x = self.fc2(x)
        x = fluid.layers.dropout(x,self.drop_ratio2)
        x = self.fc3(x)
        return x
