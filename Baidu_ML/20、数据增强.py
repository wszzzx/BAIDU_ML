import numpy as np
import cv2
from PIL import Image,ImageEnhance
import random

def random_distort(img):
    def random_brightness(img,lower=0.5,upper = 1.5):
        e = np.random.uniform(lower,upper)
        return ImageEnhance.Brightness(img).enhance(e)
    def random_contrast(img,lower=0.5,upper=1.5):
        e = np.random.uniform(lower,upper)
        return ImageEnhance.Contrast(img).enhance(e)
    def random_color(img,lower=0.5,upper=1.5):
        e = np.random.uniform(img,lower=0.5,upper=1.5)
        return ImageEnhance.Color(img).enhance(e)
    ops= [random_brightness, random_contrast, random_color]
    np.random.shuffle(ops)
    img = Image.fromarray(img)
    img = ops[0](img)
    img = ops[1](img)
    img = ops[2](img)
    img = np.asarray(img)
    return img
def random_expand(img,gtboxes,max_ratio =4,fill= None,keep_ratio = True,thresh = 0.5):
    if random.random() > thresh:
        return img,gtboxes
    if max_ratio <1.0:
        return img,gtboxes
    h,w,c = img.shape
    ratio_x = random.uniform(1,max_ratio)
    if keep_ratio:
        ratio_y = ratio_x
    else:
        ratio_y = random.uniform(1,max_ratio)
    oh = int(h*ratio_y)
    ow = int(w*ratio_x)
    off_x = random.randint(0,ow-w)
    off_y = random.randint(0,oh -h)
    out_img = np.zeros((oh,ow,c))
    if fill and len(fill) == c:
        for i in range(c):
            out_img[:,:i] = fill[i] * 255.0
    out_img[off_y:off_y+h,off_x:off_x+w,:] = img
    gtboxes[:,0] = (gtboxes[:,0]*w + off_x) / float(ow)
    gtboxes[:,1] = (gtboxes[:,1]*h + off_y) / float(oh)
    gtboxes[:,2] = gtboxes[:,2] /ratio_x
    gtboxes[:,3] = gtboxes[:,3]/ratio_y
    return out_img.astype('uint8'),gtboxes



