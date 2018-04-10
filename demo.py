import sys
import os
import time

import torch
import cv2
from torch.autograd import Variable
import numpy as np
from torchvision import transforms

from nlfd import build_model


def preprocess_img(img):
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (352,352))
    img = torch.Tensor(img).permute(2,0,1)/255.0
    img = Variable(img.unsqueeze(0), volatile=True)
    return img

def postprocess(prob):
    return prob*255
    _, contours, _ = cv2.findContours((prob>0.3).astype(np.uint8),
            cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_size = 0
    max_contour = None
    for contour in contours:
        bb = cv2.boundingRect(contour)
        bb_size = (bb[2]-bb[0]+1)*(bb[3]-bb[1]+1)
        if bb_size > max_size:
            max_size = bb_size
            max_contour = contour
    prob = np.zeros(prob.shape, dtype=np.uint8)
    cv2.fillPoly(prob, [max_contour], (255,255,255))
    return prob

def demo(net, img, cuda):
    img_h, img_w, _ = img.shape
    img = preprocess_img(img)
    if cuda:
        img = img.cuda()
    prob = net(img)
    prob = prob.cpu().data[0][0].numpy()
    prob = postprocess(prob)
    prob = cv2.resize(prob, (img_w, img_h)).astype(np.uint8)
    return prob

if __name__ == '__main__':
    model_path = sys.argv[1]
    img_path = sys.argv[2]
    cuda = False

    img = cv2.imread(img_path)

    net = build_model()
    net.load_state_dict(torch.load(model_path))
    net.eval()
    if cuda:
        net = net.cuda()

    num_run = 1
    total_time = 0
    for i in range(num_run):
        start = time.time()
        mask = demo(net, img, cuda)
        eps = time.time() -start
        print('run-%d\t%.3f' % (i, eps*1000))
        if i != 0: # exclude first run
            total_time += eps 
    #print('average runtime[%s] %.3fms' % (mode, total_time/(num_run-1)*1000))

    mask = cv2.merge([mask, mask, mask])
    result = np.concatenate((img, mask), axis=1)
    cv2.imshow('result', result)
    cv2.waitKey()
