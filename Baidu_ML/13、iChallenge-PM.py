import cv2
import random
import numpy as np
import os
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Conv2D,Pool2D,Linear,BatchNorm
import math

def transform_img(img):
    img = cv2.resize(img,(224,224))
    img = np.transpose(img,(2,0,1)).astype('float32')
    img = img/255
    img = img*2 - 1
    return img

def data_loader(datadir,batch_size = 10,mode='train'):
    filenames = os.listdir(datadir)
    def reader():
        if mode == 'train':
            random.shuffle(filenames)
        batch_imgs = []
        batch_labels = []
        for name in filenames:
            filepath = os.path.join(datadir,name)
            img = cv2.imread(filepath)
            img = transform_img(img)
            if name[0]=='H' or name[0] =='N':
                label = 0
            elif name[0] == 'P':
                label = 1
            else:
                raise ('Not excepted file name')
            batch_imgs.append(img)
            batch_labels.append(label)
            if len(batch_imgs) == batch_size:
                imgs_array = np.array(batch_imgs).astype('float32')
                labels_array = np.array(batch_labels).astype('float32').reshape(-1,1)
                yield imgs_array,labels_array
                batch_imgs = []
                batch_labels = []
        if len(batch_imgs) >0:
            imgs_array = np.array(batch_imgs).astype('float32')
            labels_array = np.array(batch_labels).astype('float32').reshape(-1,1)
            yield imgs_array,labels_array
    return reader

def valid_data_loader(datadir,csvfile,batch_size=10,mode='valid'):
    filelists = open(csvfile).readlines()
    def reader():
        batch_imgs = []
        batch_labels = []
        for line in filelists[1:]:
            # print("line",line)
            line = line.strip().split(',')
            name = line[1]
            label = int(line[2])
            filepath = os.path.join(datadir,name)
            img = cv2.imread(filepath)
            img = transform_img(img)
            batch_imgs.append(img)
            batch_labels.append(label)
            if len(batch_imgs) == batch_size:
                imgs_array = np.array(batch_imgs).astype('float32')
                labels_array = np.array(batch_labels).astype('float32').reshape(-1,1)
                yield imgs_array,labels_array
                batch_imgs = []
                batch_labels = []
        if len(batch_imgs) >0:
            imgs_array = np.array(batch_imgs).astype('float32')
            labels_array = np.array(batch_labels).astype('float32').reshape(-1, 1)
            yield imgs_array, labels_array
    return reader
train_datadir = r"D:\codes\python_codes\Prepared_2020\Baidu_ML\palm\PALM-Training400\PALM-Training400"
valid_datadir = r"D:\codes\python_codes\Prepared_2020\Baidu_ML\palm\PALM-Validation400"
csv_file = r"D:\codes\python_codes\Prepared_2020\Baidu_ML\palm\PALM-Validation-GT\PM_Label_and_Fovea_Location.csv"

def train(model):
    with fluid.dygraph.guard():
        print("start training ...")
        model.train()
        epoch_num = 10
        opt = fluid.optimizer.SGDOptimizer(learning_rate=0.01,parameter_list=model.parameters())
        train_loader = data_loader((train_datadir))
        vaild_loader = valid_data_loader(valid_datadir,csv_file)
        for epoch in range(epoch_num):
            for batch_id,data in enumerate(train_loader()):
                x_data,y_data = data
                img = fluid.dygraph.to_variable(x_data)
                label = fluid.dygraph.to_variable(y_data)
                logits = model.forward(img)
                loss = fluid.layers.sigmoid_cross_entropy_with_logits(logits,label)
                avg_loss = fluid.layers.mean(loss)
                if batch_id%10 == 0:
                    print("epoch:{},batch_id:{},loss is :{}".format(epoch,batch_id,avg_loss.numpy()))
                avg_loss.backward()
                opt.minimize(avg_loss)
                model.clear_gradients()
            model.eval()
            accuracies = []
            losses = []
            for batch_id,data in enumerate(vaild_loader()):
                x_data,y_data = data
                img = fluid.dygraph.to_variable(x_data)
                label = fluid.dygraph.to_variable(y_data)
                logits = model.forward(img)
                pred = fluid.layers.sigmoid(logits)
                # print("pred",pred)
                loss = fluid.layers.sigmoid_cross_entropy_with_logits(logits,label)
                pred2 = pred*(-1) + 1
                # print("pred2", pred2)
                pred = fluid.layers.concat([pred2,pred],axis=1)
                # print("pred3", pred)
                acc = fluid.layers.accuracy(pred,fluid.layers.cast(label,dtype='int64'))
                accuracies.append(acc.numpy())
                losses.append(loss.numpy())
            print("[validation] accuracy/loss:{}/{}".format(np.mean(accuracies),np.mean(losses)))
            model.train()
        fluid.save_dygraph(model.state_dict(),'iChallenge_0708')
        # fluid.save_dygraph(opt.state_dict(),'iChallenge_0708')
def evaluation(model,params_file_path):
    with fluid.dygraph.guard():
        print('start evaluation ...')
        model_state_dict,_ = fluid.load_dygraph(params_file_path)
        model.load_dict(model_state_dict)
        model.eval()

# 定义 LeNet 网络结构
class LeNet(fluid.dygraph.Layer):
    def __init__(self, num_classes=1):
        super(LeNet, self).__init__()

        # 创建卷积和池化层块，每个卷积层使用Sigmoid激活函数，后面跟着一个2x2的池化
        self.conv1 = Conv2D(num_channels=3, num_filters=6, filter_size=5, act='relu')
        self.pool1 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
        self.conv2 = Conv2D(num_channels=6, num_filters=16, filter_size=5, act='relu')
        self.pool2 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
        # 创建第3个卷积层
        self.conv3 = Conv2D(num_channels=16, num_filters=120, filter_size=4, act='relu')
        # 创建全连接层，第一个全连接层的输出神经元个数为64， 第二个全连接层输出神经元个数为分类标签的类别数
        self.fc1 = Linear(input_dim=300000, output_dim=64, act='relu')
        self.fc2 = Linear(input_dim=64, output_dim=num_classes)
    # 网络的前向计算过程
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = fluid.layers.reshape(x, [x.shape[0], -1])
        x = self.fc1(x)
        x = self.fc2(x)
        return x

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


class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(self, num_channels, num_filters, filter_size, stride=1, groups=1, act=None):
        super(ConvBNLayer, self).__init__()
        self._conv = Conv2D(num_channels=num_channels, num_filters=num_filters, filter_size=filter_size, stride=stride,
                            padding=(filter_size - 1) // 2, groups=groups, act=None, bias_attr=False)
        self._batch_norm = BatchNorm(num_filters, act=act)

    def forward(self, inputs, ):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y


class BottleneckBlock(fluid.dygraph.Layer):
    def __init__(self, num_channels, num_filters, stride, shortcut=True):
        super(BottleneckBlock, self).__init__()
        self.conv0 = ConvBNLayer(num_channels=num_channels, num_filters=num_filters, filter_size=1, act='relu')
        self.conv1 = ConvBNLayer(num_channels=num_filters, num_filters=num_filters, filter_size=3, stride=stride,
                                 act='relu')
        self.conv2 = ConvBNLayer(num_channels=num_filters, num_filters=num_filters * 4, filter_size=1, act=None)

        if not shortcut:
            self.short = ConvBNLayer(num_channels=num_channels, num_filters=num_filters * 4, filter_size=1,
                                     stride=stride)
        self.shortcut = shortcut
        self._num_channels_out = num_filters * 4

    def forward(self, inputs, ):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = fluid.layers.elementwise_add(x=short, y=conv2, act='relu')
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

if __name__ == '__main__':
    # 创建模型
    with fluid.dygraph.guard():
        # model = AlexNet(num_classes=1)
        # model = VGG()
        # model = GoogLeNet()
        # model = ResNet()
        model = LeNet()


    train(model)



