import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D,Pool2D,Linear

class Inception(fluid.dygraph.Layer):
    def __init__(self,c0,c1,c2,c3,c4):
        super(Inception,self).__init__()
        self.p1_1 = Conv2D(num_channels=c0,num_filters=c1,filter_size=1,act='relu')
        self.p2_1 = Conv2D(num_channels=c0,num_filters=c2[0],filter_size=1,act='relu')
        self.p2_2 = Conv2D(num_channels=c2[0],num_filters=c2[1],filter_size=3,padding=1,act='relu')
        self.p3_1 = Conv2D(num_channels=c0,num_filters=c3[0],filter_size=1,act='relu')
        self.p3_2 = Conv2D(num_channels=c3[0],num_filters=c3[1],filter_size=5,padding=2,act='relu')
        self.p4_1 = Pool2D(pool_size=3,pool_stride=1,pool_padding=1,pool_type='max')
        self.p4_2 = Conv2D(num_channels=c0,num_filters=c4,filter_size=1,act='relu')

    def forward(self, x):
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))
        return fluid.layers.concat([p1,p2,p3,p4],axis=1)
class GoogLeNet(fluid.dygraph.Layer):
    def __init__(self):
        super(GoogLeNet,self).__init__()
        # 1
        self.conv1 = Conv2D(num_channels=3,num_filters=64,filter_size=7,padding=3,act='relu')
        self.pool1 = Pool2D(pool_size=3,pool_stride=2,pool_padding=1,pool_type='max')

        # 2
        self.conv2_1 = Conv2D(num_channels=64,num_filters=64,filter_size=1,act='relu')
        self.conv2_2 = Conv2D(num_channels=64,num_filters=192,filter_size=3,padding=1,act='relu')
        self.pool2 = Pool2D(pool_size=3,pool_stride=2,pool_padding=1,pool_type='max')


        # 3
        self.block3_1 = Inception(192, 64, (96, 128), (16, 32), 32)
        self.block3_2 = Inception(256, 128, (128, 192), (32, 96), 64)
        self.pool3 = Pool2D(pool_size=3, pool_stride=2,pool_padding=1, pool_type='max')

        # 4
        self.block4_1 = Inception(480, 192, (96, 208), (16, 48), 64)
        self.block4_2 = Inception(512, 160, (112, 224), (24, 64), 64)
        self.block4_3 = Inception(512, 128, (128, 256), (24, 64), 64)
        self.block4_4 = Inception(512, 112, (144, 288), (32, 64), 64)
        self.block4_5 = Inception(528, 256, (160, 320), (32, 128), 128)
        self.pool4 = Pool2D(pool_size=3, pool_stride=2,pool_padding=1, pool_type='max')

        # 5
        self.block5_1 = Inception(832,256,(160,320),(32,128),128)
        self.block5_2 = Inception(832,384,(192,384),(48,128),128)
        self.pool5 = Pool2D(pool_stride=1,global_pooling=True,pool_type='avg')



        self.fc = Linear(input_dim=1024,output_dim=1,act=None)

    def forward(self,x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2_2(self.conv2_1(x)))
        x = self.pool3(self.block3_2(self.block3_1(x)))
        x = self.pool4(self.block4_5(self.block4_4(self.block4_3(self.block4_2(self.block4_1(x))))))
        x = self.pool5(self.block5_2(self.block5_1(x)))
        x = fluid.layers.reshape(x,[x.shape[0],-1])
        x = self.fc(x)
        return x











