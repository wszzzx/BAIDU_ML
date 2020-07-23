import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph import Conv2D,Pool2D,BatchNorm,Linear
from paddle.fluid.dygraph.base import to_variable
import math

class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(self,num_channels,num_filters,filter_size,stride=1,groups=1,act=None):
        super(ConvBNLayer,self).__init__()
        self._conv = Conv2D(num_channels=num_channels,num_filters=num_filters,filter_size=filter_size,stride=stride,padding=(filter_size -1)//2,groups=groups,act=None,bias_attr=False)
        self._batch_norm = BatchNorm(num_filters,act=act)
    def forward(self, inputs,):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y

class BottleneckBlock(fluid.dygraph.Layer):
    def __init__(self,num_channels,num_filters,stride,shortcut=True):
        super(BottleneckBlock,self).__init__()
        self.conv0 = ConvBNLayer(num_channels=num_channels,num_filters=num_filters,filter_size=1,act='relu')
        self.conv1 = ConvBNLayer(num_channels=num_filters,num_filters=num_filters,filter_size=3,stride=stride,act='relu')
        self.conv2 = ConvBNLayer(num_channels=num_filters,num_filters=num_filters*4,filter_size=1,act=None)

        if not shortcut:
            self.short = ConvBNLayer(num_channels=num_channels,num_filters=num_filters*4,filter_size=1,stride=stride)
        self.shortcut = shortcut
        self._num_channels_out = num_filters*4
    def forward(self, inputs,):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = fluid.layers.elementwise_add(x=short,y=conv2,act='relu')
        return y

class ResNet(fluid.dygraph.Layer):
    def __init__(self, layers=50, class_dim=1):
        super(ResNet, self).__init__()
        self.layers = layers
        supported_layers = [50, 101, 152]
        assert layers in supported_layers, "supported layers are {} but input layer is {}".format(supported_layers,
                                                                                                  layers)
        if layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        num_filters = [64, 128, 256, 512]
        self.conv = ConvBNLayer(num_channels=3, num_filters=64, filter_size=7, stride=2, act='relu')
        self.pool2d_max = Pool2D(pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')
        self.bottlenect_block_list = []
        num_channels = 64
        for block in range(len(depth)):
            shortcut = False
            for i in range(depth[block]):
                bottleneck_block = self.add_sublayer('bb_%d_%d' % (block, i), BottleneckBlock(num_channels=num_channels,
                                                                                           num_filters=num_filters[
                                                                                               block],
                                                     stride=2 if i == 0 and block != 0 else 1, shortcut=shortcut))
                num_channels = bottleneck_block._num_channels_out
                self.bottlenect_block_list.append(bottleneck_block)
                shortcut = True
        self.pool2d_avg = Pool2D(pool_size=7, pool_type='avg', global_pooling=True)

        stdv = 1.0 / math.sqrt(2048 * 1.0)
        self.out = Linear(input_dim=2048, output_dim=class_dim,
                          param_attr=fluid.param_attr.ParamAttr(initializer=fluid.initializer.Uniform(-stdv, stdv)))

    def forward(self, inputs):
        y = self.conv(inputs)
        y = self.pool2d_max(y)
        for bottlenect_block in self.bottlenect_block_list:
            y = bottlenect_block(y)
        y = self.pool2d_avg(y)
        y = fluid.layers.reshape(y, [y.shape[0], -1])
        y = self.out(y)
        return y



