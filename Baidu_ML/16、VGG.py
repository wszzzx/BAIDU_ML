import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv2D,Pool2D,BatchNorm,Linear
from paddle.fluid.dygraph.base import to_variable

class vgg_block(fluid.dygraph.Layer):
    def __init__(self,num_convs,in_channels,out_channels):
        super(vgg_block,self).__init__()
        self.conv_list = []
        for i in range(num_convs):
            conv_layer = self.add_sublayer('conv_'+str(i),Conv2D(num_channels=in_channels,num_filters=out_channels,filter_size=3,padding=1,act='relu'))
            self.conv_list.append(conv_layer)
            in_channels = out_channels
        self.pool = Pool2D(pool_stride=2,pool_size=2,pool_type='max')
    def forward(self, x):
        for item in self.conv_list:
            x = item(x)
        return self.pool(x)

class VGG(fluid.dygraph.Layer):
    def __init__(self,conv_arch = ((2,64),(2,128),(3,256),(3,512),(3,512))):
        super(VGG,self).__init__()
        self.vgg_blocks = []
        iter_id = 0
        in_channels = [3,64,128,256,512,512]
        for (num_convs,num_channels) in conv_arch:
            block = self.add_sublayer('block_'+str(iter_id),vgg_block(num_convs,in_channels=in_channels[iter_id],out_channels=num_channels))
            self.vgg_blocks.append(block)
            iter_id +=1
        self.fc1 = Linear(input_dim=512*7*7,output_dim=4096,act='relu')
        self.drop1_ratio = 0.5
        self.fc2 = Linear(input_dim=4096,output_dim=4096,act='relu')
        self.drop2_ratio = 0.5
        self.fc3 = Linear(input_dim=4096,output_dim=1)
    def forward(self, x):
        for item in self.vgg_blocks:
            x = item(x)
        x = fluid.layers.reshape(x,[x.shape[0],-1])
        x = fluid.layers.dropout(self.fc1(x),self.drop1_ratio)
        x = fluid.layers.dropout(self.fc2(x),self.drop2_ratio)
        x = self.fc3(x)
        return x


