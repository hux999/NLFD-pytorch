import sys
import random
import math
import skimage

import numpy as np 
import argparse
import json
import cv2


class TransFactory():
   
    def __init__(self):
        self.trans_map = {}

    def register(self,trans_name, trans_class):
        self.trans_map[trans_name] = trans_class

    def create_preprocess(self,trans_name,*parameter):
        print(parameter)
        if parameter:
            return self.trans_map[trans_name](json.loads(*parameter))
        else:
            return self.trans_map[trans_name]()
    def trans_name():
        return self.trans_map.key()

_trans_factory = TransFactory()


class CurriculumWrapper:
    def __init__(self, trans, prob):
        self.trans = trans
        self.prob = prob
    def __call__(self, *args):
        if random.random() < self.prob:
            return self.trans(*args)
        else:
            if len(args) == 1:
                args = args[0]
            return args

class Resize:
    def __init__(self, img_size, target_size=None):
        self.img_size = img_size
        self.target_size = target_size if target_size is not None else img_size

    def __call__(self, im, mask):
        im = cv2.resize(im, self.img_size)
        mask = cv2.resize(mask, self.target_size)
        return im, mask

class ReColor: 
    def __init__(self, alpha=0.05, beta=0.3):
        self._alpha = alpha
        self._beta = beta

    def __call__(self, im, mask):
        # random amplify each channel
        t = np.random.uniform(-1, 1, 3)
        im *= (1 + t * self._alpha)
        mx = 255. * (1 + self._alpha)
        up = np.random.uniform(-1, 1)
        im = np.power(im / mx, 1. + up * self._beta)
        im = im * 255
        return im, mask
_trans_factory.register('ReColor', ReColor) 

class GaussianNoise:
    def __init__(self, mean=0, var=0.01):
        self.mean = mean
        self.var = var

    def __call__(self, im, mask):
        noise = np.random.normal(self.mean, self.var, size=im.shape)*255.0
        im = np.clip(im+noise, 0, 255.0)
        return im, mask
_trans_factory.register('GaussianNoise', GaussianNoise ) 


class GaussianNoiseLocal:
    def __init__(self, diff=20):
        self.diff = diff

    def __call__(self, im, mask):
        src_h,src_w,_ = im.shape

        center_h = random.randint(0,src_h)
        center_w = random.randint(0,src_w)
        R = random.randint(1,30)

        j, i = np.meshgrid(np.arange(src_w), np.arange(src_h))
        dis = np.sqrt((i-center_h)**2+(j-center_w)**2)
        noise_map = np.exp(-0.5*dis/R)

        rand_map = np.random.rand(src_h,src_w)
        noise_map[rand_map>noise_map]= 0

        R_change = random.randint(0,self.diff)

        im = im - noise_map.reshape(src_h, src_w, 1)*R_change  
        im = np.clip(im, 0, 255.0)
        return im, mask
_trans_factory.register('GaussianNoiseLocal', GaussianNoiseLocal ) 

class SampleVolume:
    def __init__(self, dst_shape=[96, 96, 5], pos_ratio=-1):
        self.dst_shape = dst_shape
        self.pos_ratio = pos_ratio

    def __call__(self, data, label):
        src_h,src_w,src_d,_ = data.shape
        dst_h,dst_w,dst_d = self.dst_shape
        if type(dst_d) is list:
            dst_d = random.choice(dst_d)
        if self.pos_ratio<0:
            h = random.randint(0, src_h-dst_h)
            w = random.randint(0, src_w-dst_w)
            d = random.randint(0, src_d-dst_d)
        else:
            select = label>0 if random.random() < pos else label==0
            h, w, d = np.where(select)
            select_idx = random.randint(0, len(h)-1)
            h = h[select_idx] + int(dst_h/2)
            w = w[select_idx] + int(dst_w/2)
            d = d[select_idx] + int(dst_d/2)
            h = min(max(h,0), h-dst_h+1)
            w = min(max(w,0), w-dst_w+1)
            d = min(max(d,0), w-dst_d+1)
        sub_volume = data[h:h+dst_h,w:w+dst_w,d:d+dst_d,:]
        sub_label = label[h:h+dst_h,w:w+dst_w,d:d+dst_d]
        return sub_volume,sub_label
_trans_factory.register('SampleVolume', SampleVolume )

class ScaleAndPad:
    def __init__(self, dst_size = 500, rand_pad=False):
        self.dst_size = dst_size
        self.rand_pad = rand_pad

    def __call__(self, im, mask):
        org_h,org_w,org_d,org_c = im.shape
        fx = math.floor(self.dst_size/org_w)
        fy = math.floor(self.dst_size/org_h)
        new_im = np.zeros((self.dst_size, self.dst_size,org_d,org_c), im.dtype)
        org_h = int(org_h*fy)
        org_w = int(org_w*fx)
        offset_x = 0 if self.rand_pad is False else random.randint(0, self.dst_size-org_w)
        offset_y = 0 if self.rand_pad is False else random.randint(0, self.dst_size-org_h)
        for i in range(org_c):
            new_im[offset_y:offset_y+org_h, offset_x:offset_x+org_w, :,i] = cv2.resize(im[:,:,:,i], None,fx=fx, fy=fy)
        new_mask = np.zeros((self.dst_size, self.dst_size,org_d), mask.dtype)
        new_mask[offset_y:offset_y+org_h, offset_x:offset_x+org_w,:] = cv2.resize(mask, None, fx=fx, fy=fy)
        return new_im, new_mask
_trans_factory.register('ScaleAndPad', ScaleAndPad )

class RandomJitter:
    def __init__(self, max_angle=180, max_scale=0.1):
        self._max_angle = max_angle
        self._max_scale = max_scale

    def __call__(self, im, mask):
        h,w,d,c= im.shape
        center = h/2.0,w/2.0
        angle = np.random.uniform(0, self._max_angle)
        scale = np.random.uniform(0, self._max_scale) + 1.0
        m = cv2.getRotationMatrix2D(center, angle, scale)
        new_im = np.zeros((h,w,d,c), im.dtype)
        for j in range(d):
            for i in range(c):
                    new_im[:,:,j,i] = cv2.warpAffine(im[:,:,j,i], m, (w,h))
            mask[:,:,j] = cv2.warpAffine(mask[:,:,j], m, (w,h))
        return new_im, mask
_trans_factory.register('RandomJitter', RandomJitter )

class RandomCrop:
    def __init__(self, crop_size=[96,96], rotation=False):
        self.crop_size = crop_size

    def __call__(self, im, mask):
        x = random.randint(0, im.shape[1]-self.crop_size[0])
        y = random.randint(0, im.shape[0]-self.crop_size[1])
        crop_im = im[y:y+self.crop_size[1], x:x+self.crop_size[0], :]
        crop_mask = mask[y:y+self.crop_size[1], x:x+self.crop_size[0]]
        return crop_im, crop_mask
_trans_factory.register('RandomCrop', RandomCrop)

class RandomFlip:
    def __init__(self):
        pass

    def __call__(self, im, mask):
        if random.random() > 0.5:
            im = im[:, ::-1, :]
            mask = mask[:, ::-1]
        return im, mask
_trans_factory.register('RandomFlip', RandomFlip)

class RandomRotate:
    def __init__(self, random_flip=True):
        self.random_flip = random_flip

    def __call__(self, im, mask):
        rotate = random.randint(0, 3)
        if self.random_flip and random.random() > 0.5:
            im = im[:, ::-1, :]
            mask = mask[:, ::-1]
        if rotate > 0:
            im = np.rot90(im, rotate)
            mask = np.rot90(mask, rotate)
        return im.copy(), mask.copy()
_trans_factory.register('RandomRotate', RandomRotate)

def Data2Mat(data):
    data = data.astype(np.float32)
    data *= 255.0/data.max()
    return data.astype(np.uint8)

def MakeGrid(imgs, width=8):
    h, w, c = imgs[0].shape
    height = int(len(imgs)/width) + (1 if len(imgs)%width > 0 else 0)
    ind = 0
    concat_img = np.zeros((h*height, w*width, c), np.uint8)
    for h_idx in range(height):
        for w_idx in range(width):
            if ind >= len(imgs):
                continue
            concat_img[h_idx*h:(h_idx+1)*h, w_idx*w:(w_idx+1)*w, :] =  imgs[ind]
            ind += 1
    return concat_img

def Label2Color(data):
    color_bar = [
            (0, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 0, 0),
            (255, 0, 255),
            (0, 255, 255),
            (255, 255, 0)
            ]
    max_label = np.max(data)
    R = np.zeros(data.shape, np.uint8)
    G = np.zeros(data.shape, np.uint8)
    B = np.zeros(data.shape, np.uint8)

    for label in range(1, max_label+1):
        R[data==label] = color_bar[label][0]
        G[data==label] = color_bar[label][1]
        B[data==label] = color_bar[label][2]
    img_data = cv2.merge([B, G, R])
    return img_data

def VisualizeData(data,label,is_process = False):
    imgs = {}
    concat_img = []
    for i in range(data.shape[2]):
        for j in range(data.shape[3]):
            imgs[str(j)] = Data2Mat(data[:,:,i,j])
        imgs["ot"] = label[:,:,i]
        for img_type, img_data in imgs.items():
            if img_type == "ot":
                imgs[img_type] = Label2Color(img_data)
            else:
                imgs[img_type] = cv2.merge([img_data, img_data, img_data])
        for img_type, img_data in imgs.items():
            cv2.putText(img_data, img_type, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
        concat_img.append(MakeGrid(imgs.values(), data.shape[3]+1))
    return concat_img
        
def Visualize(person_data,process,parameter):
    data = person_data['data']
    label = person_data['label']
    raw_data = data.copy()
    new_data,new_label = _trans_factory.create_preprocess(process,*parameter)(data,label)
    img = VisualizeData(raw_data,label)
    new_img = VisualizeData(new_data,new_label,True)
    for i in range(len(new_img)):
        cv2.imshow('img', img[i])
        cv2.imshow('process_img', new_img[i])
        cv2.waitKey()


def preprcess_visulize():
    process = ['ReColor', 'GaussianNoise','RandomFlip','ScaleAndPad','RandomJitter','RandomCrop','RandomRotate','SampleVolume']
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=str, help="root of image")
    parser.add_argument('process', type=str, choices=process,help="process function")
    parser.add_argument('parameter', nargs='*',type=str, help="parameter of process")
    args = parser.parse_args()

    Visualize(np.load(args.root),args.process,args.parameter)


if __name__ == '__main__':
    preprcess_visulize()
