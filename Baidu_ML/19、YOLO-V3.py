import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.image import imread
import math

import os
import xml.etree.ElementTree as ET
import cv2

def draw_rectangle(currentAxis,bbox,edgecolor='k',facecolor='y',fill=False,linestyle='-'):
    rect = patches.Rectangle((bbox[0],bbox[1]),bbox[2]-bbox[0]+1,bbox[3]-bbox[1]+1,linewidth=1,edgecolor=edgecolor,facecolor=facecolor,fill=fill,linestyle=linestyle)
    currentAxis.add_patch(rect)

            # draw_anchor_box([300., 500.], 100., [2.0], [0.5, 1.0, 2.0], img_height, img_width)
def draw_anchor_box(currentAxis,center,length,scales,ratios,img_height,img_width):
    bboxes = []
    for scale in scales:
        for ratio in ratios:
            h = length*scale*math.sqrt(ratio)
            w = length*scale/math.sqrt(ratio)
            x1 = max(center[0] - w/2,0)
            y1 = max(center[1] - h/2,0)
            x2 = min(center[0] + w/2 -1.0,img_width - 1.0)
            y2 = min(center[1] + h/2 - 1.0,img_height - 1.0)
            print(center[0],center[1],w,h)
            bboxes.append([x1,y1,x2,y2])
    for bbox in bboxes:
        draw_rectangle(currentAxis,bbox,edgecolor='b')
def box_iou_xyxy(box1,box2):
    x1min,y1min,x1max,y1max = box1[0],box1[1],box1[2],box1[3]
    s1 = (y1max - y1min + 1)*(x1max - x1min +1)
    x2min,y2min,x2max,y2max = box2[0],box2[1],box2[2],box2[3]
    s2 = (y2max - y2min +1)*(x2max - x2min +1)
    xmin = np.maximum(x1min,x2min)
    ymin = np.maximum(y1min,y2min)
    xmax = np.minimum(x1max,x2max)
    ymax = np.minimum(x2max,y2max)

    inter_h = np.maximum(ymax -ymin + 1,0)
    inter_w = np.maximum(xmax -xmin +1,0)

    intersection = inter_h * inter_w
    union = s1 + s2 - intersection

    iou = intersection / union
    return iou
def box_iou_xywh(box1, box2):
    x1min, y1min = box1[0] - box1[2]/2.0, box1[1] - box1[3]/2.0
    x1max, y1max = box1[0] + box1[2]/2.0, box1[1] + box1[3]/2.0
    s1 = box1[2] * box1[3]

    x2min, y2min = box2[0] - box2[2]/2.0, box2[1] - box2[3]/2.0
    x2max, y2max = box2[0] + box2[2]/2.0, box2[1] + box2[3]/2.0
    s2 = box2[2] * box2[3]

    xmin = np.maximum(x1min, x2min)
    ymin = np.maximum(y1min, y2min)
    xmax = np.minimum(x1max, x2max)
    ymax = np.minimum(y1max, y2max)
    inter_h = np.maximum(ymax - ymin, 0.)
    inter_w = np.maximum(xmax - xmin, 0.)
    intersection = inter_h * inter_w

    union = s1 + s2 - intersection
    iou = intersection / union
    return iou
INSECT_NAMES = ['Boerner', 'Leconte', 'Linnaeus', 'acuminatus', 'armandi', 'coleoptera', 'linnaeus']

def get_insect_names():
    insect_category2id = {}
    for i,item in enumerate(INSECT_NAMES):
        insect_category2id[item] = i
    return insect_category2id
cname2cid = get_insect_names()
def get_annotations(cname2cid,datadir):
    filenames = os.listdir(os.pardir.join(datadir,'annotations','xmls'))
    records = []
    ct = 0
    for fname in filenames:
        fid = fname.split('.')[0]
        fpath = os.path.join(datadir,'annotations','xmls',fname)
        img_file = os.path.join(datadir,'images',fid +'.jpeg')
        tree = ET.parse(fpath)
        if tree.find('id') is None:
            im_id = np.array([ct])
        else:
            im_id = np.array([int(tree.find('id').text)])
        objs = tree.findall('object')
        im_w = float(tree.find('size').find('width').text)
        im_h = float(tree.find('size').find('height').text)
        gt_bbox = np.zeros((len(objs),4),dtype= np.float32)
        gt_class = np.zeros((len(objs),),dtype=np.int32)
        is_crowd = np.zeros((len(objs),),dtype=np.int32)
        difficult = np.zeros((len(objs),),dtype=np.int32)
        for i,obj in enumerate(objs):
            cname = obj.find('name').text
            gt_class[i] = cname2cid[cname]
            _difficult = int(obj.find('difficult').text)
            x1 = float(obj.find('bndbox').find('xmin').text)
            y1 = float(obj.find('bndbox').find('ymin').text)
            x2 = float(obj.find('bndbox').find('xmax').text)
            y2 = float(obj.find('bndbox').find('ymax').text)
            gt_bbox[i] = [(x1+x2)/2.0,(y1+y2)/2.0,x2-x1+1,y2-y1+1]
            is_crowd[i] = 0
            difficult[i] = _difficult
        voc_rec = {'im_file':img_file,'im_id':im_id,'h':im_h,'w':im_w,'is_crowd':is_crowd,'gt_class':gt_class,'gt_bbox':gt_bbox,'gt_poly':[],'difficult':difficult}
        if len(objs) !=0:
            records.append(voc_rec)
        ct +=1
    return records
TRAINDIR = 'D:\codes\python_codes\Data/insects/train'
TESTDIR = 'D:\codes\python_codes\Data/insects/test'
VALIDIR = 'D:\codes\python_codes\Data/insects/val'
cname2cid = get_insect_names()
records = get_annotations(cname2cid,TESTDIR)
print(len(records))
print(records[0])

def get_bbox(gt_bbox,gt_class):
    MAX_NUM = 50
    gt_bbox2 = np.zeros((MAX_NUM,4))
    gt_class2 = np.zeros((MAX_NUM))
    for i in range(len(gt_bbox)):
        gt_bbox2[i,:] = gt_bbox[i:]
        gt_class2[i] = gt_class[i]
        if i >=MAX_NUM:
            break
    return gt_bbox2,gt_class2


#############################V3

def get_objectness_label(img,gt_boxs,gt_labels,iou_threshold=0.7,anchors=[116,90,156,198,373,326],num_classes =7,downsample=32):
    img_shape = img.shape
    batchsize = img_shape[0]
    num_anchors = len(anchors)//2
    input_h = img_shape[2]
    input_w = img_shape[3]
    num_rows = input_h // downsample
    num_cols = input_w // downsample

    label_objectness = np.zeros([batchsize,num_anchors,num_rows,num_cols])
    label_classification = np.zeros([batchsize,num_anchors,num_classes,num_rows,num_cols])
    label_location = np.zeros([batchsize,num_anchors,4,num_rows,num_cols])

    scale_location = np.ones([batchsize,num_anchors,num_rows,num_cols])

    for n in range(batchsize):
        for n_gt in range(len([n])):
            gt = gt_boxs[n][n_gt]
            gt_cls = gt_labels[n][n_gt]
            gt_center_x = gt[0]
            gt_center_y = gt[1]
            gt_width = gt[2]
            gt_height = gt[3]
            if (gt_height <1e-3) or (gt_height <1e-3):
                continue
            i = int(gt_center_y*num_rows)
            j = int(gt_center_x*num_cols)
            ious = []
            for ka in range(num_anchors):
                bbox1 = [0,0,float(gt_width),float(gt_height)]
                anchor_w = anchors[ka*2]
                anchor_h = anchors[ka*2+1]
                bbox2 = [0,0,anchor_w/float(input_w),anchor_h(input_h)]
                iou = box_iou_xywh(bbox1,bbox2)
                ious.append(iou)
            ious = np.array(ious)
            inds = np.argsort(ious)
            k = inds[-1]
            label_objectness[n,k,i,j] = 1
            c = gt_cls
            label_classification[n,k,c,i,j] = 1

            dx_label = gt_center_x * num_cols - j
            dy_label = gt_center_y * num_rows - i
            dw_label = np.log(gt_width*input_w/anchors[k*2])
            dh_label = np.log(gt_height*input_h/anchors[k*2+1])

            label_location[n,k,0,i,j] = dx_label
            label_location[n, k, 1, i, j] = dy_label
            label_location[n, k, 2, i, j] = dw_label
            label_location[n, k, 3, i, j] = dh_label

            scale_location[n,k,i,j] = 2.0 -gt_width*gt_height

        return label_objectness.astype('float32'),label_objectness.astype('float32'),label_classification.astype('float32'),scale_location.astype('float32')

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay
from paddle.fluid.dygraph.nn import Conv2D,BatchNorm
from paddle.fluid.dygraph.base import to_variable

class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(self,ch_in,ch_out,filter_size=3,stride=1,groups=1,padding=0,act='leaky',is_test=True):
        super(ConvBNLayer,self).__init__()

        self.conv = Conv2D(num_channels=ch_in,num_filters=ch_out,filter_size=filter_size,stride=stride,padding=padding,groups=groups,param_attr=ParamAttr(initializer=fluid.initializer.Normal(0,0.02)),bias_attr=False,act=None)
        self.batch_norm = BatchNorm(num_channels=ch_out,is_test=is_test,param_attr=ParamAttr(initializer=fluid.initializer.Normal(0,0.02)),bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0),regularizer=L2Decay(0)))
        self.act = act
    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.batch_norm(out)
        if self.act == 'leaky':
            out = fluid.layers.leaky_relu(x=out,alpha=0.1)
        return  out
class DownSample(fluid.dygraph.Layer):
    def __init__(self,ch_in,ch_out,filter_size=3,stride=2,padding=1,is_test=True):
        super(DownSample,self).__init__()
        self.conv_bn_layer = ConvBNLayer(ch_in=ch_in,ch_out=ch_out,filter_size=filter_size,stride=stride,padding=padding,is_test=is_test)
        self.ch_out = ch_out
    def forward(self,inputs):
        out = self.conv_bn_layer(inputs)

class BasicBlock(fluid.dygraph.Layer):
    def __init__(self,ch_in,ch_out,is_test=True):
        super(BasicBlock,self).__init__()
        self.conv1 = ConvBNLayer(ch_in=ch_in,ch_out=ch_out,filter_size=1,stride=1,padding=0,is_test=is_test)
        self.conv2 = ConvBNLayer(ch_in=ch_out,ch_out = ch_out*2,filter_size=3,stride=1,padding=1,is_test=is_test)
    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        out = fluid.layers.elementwise_add(x=inputs,y=conv2,act=None)
        return out

class LayerWarp(fluid.dygraph.Layer):
    def __init__(self,ch_in,ch_out,count,is_test=True):
        super(LayerWarp,self).__init__()

        self.basicblock0 = BasicBlock(ch_in,ch_out,is_test=is_test)
        self.res_out_list = []
        for i in range(1,count):
            res_out = self.add_sublayer("basic_block_%d" %(i),BasicBlock(ch_out*2,ch_out,is_test=is_test))
            self.res_out_list.append(res_out)
    def forward(self, inputs):
        y = self.basicblock0(inputs)
        for basic_block_i in self.res_out_list:
            y = basic_block_i(y)
        return y
DarkNet_cfg = {53:([1,2,8,8,4])}

class DarkNet53_conv_body(fluid.dygraph.Layer):
    def __init__(self,is_test=True):
        super(DarkNet53_conv_body,self).__init__()
        self.stages = DarkNet_cfg[53]
        self.stages = self.stages[0:5]

        self.conv0 = ConvBNLayer(ch_in=3,ch_out=32,filter_size=3,stride=1,padding=1,is_test=is_test)
        self.downsample0 = DownSample(ch_in=32,ch_out=32*2,is_test=is_test)

        self.draknet53_conv_block_list = []
        self.downsample_list = []
        for i,stage in enumerate(self.stages):
            conv_block = self.add_sublayer("stage_%d" %(i),LayerWarp(32*(2**(i+1)),32*(2**i),stage,is_test=is_test))
            self.draknet53_conv_block_list.append(conv_block)
        for i in range(len(self.stages) -1):
            downsample = self.add_sublayer("stage_%d_downsample" % i,DownSample(ch_in=32*(2**(i+1)),ch_out=32*(2**(i+2)),is_test=is_test))
            self.downsample_list.append(downsample)
    def forward(self,inputs):
        out = self.conv0(inputs)
        out = self.downsample0(out)
        blocks = []
        for i ,conv_block_i in enumerate(self.draknet53_conv_block_list):
            out = conv_block_i(out)
            blocks.append(out)
            if i < len(self.stages) - 1:
                out = self.downsample_list[i](out)
        return blocks[-1:-4:-1]
class YoloDetectionBlock(fluid.dygraph.Layer):
    def __init__(self,ch_in,ch_out,is_test=True):
        super(YoloDetectionBlock, self).__init__()
        assert ch_out%2 ==0,"channel {} cannot be divided by 2".format(ch_out)
        self.conv0 = ConvBNLayer(ch_in=ch_in,ch_out=ch_out,filter_size=1,stride=1,padding=0,is_test=is_test)
        self.conv1 = ConvBNLayer(ch_in=ch_out,ch_out=ch_out*2,filter_size=3,stride=1,padding=1,is_test=is_test)
        self.conv2 = ConvBNLayer(ch_in=ch_out*2,ch_out=ch_out,filter_size=1,stride=1,padding=0,is_test=is_test)
        self.conv3 = ConvBNLayer(ch_in=ch_out,ch_out=ch_out*2,filter_size=3,stride=1,padding=0,is_test=is_test)
        self.route = ConvBNLayer(ch_in = ch_out*2,ch_out=ch_out,filter_size=1,stride=1,padding=0,is_test=is_test)
        self.tip = ConvBNLayer(ch_in=ch_out,ch_out= ch_out*2,filter_size=3,stride=1,padding=1,is_test=is_test)
    def forward(self, inputs):
        out = self.conv0(inputs)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        route = self.route(out)
        tip = self.tip(route)
        return route, tip

NUM_ANCHORS = 3
NUM_CLASSES = 7
num_filters = NUM_ANCHORS * (NUM_CLASSES + 5)
with fluid.dygraph.guard():
    













