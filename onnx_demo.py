import sys
import os
import time

import onnx
import onnx_caffe2.backend
import numpy as np
import cv2


def preprocess(img):
    img = cv2.resize(img, (352, 352))
    if len(img) == 2: # gray to color
        img = cv2.merge(img, img, img)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)[np.newaxis, :, :, :]
    return img

def postprocess(img, lscore, gscore):
    img_h, img_w, _ = img.shape
    bimg = lscore + gscore[0]
    bimg = bimg.reshape(176, 176)
    bimg = cv2.resize(bimg.copy(), (img_w, img_h))
    bimg = (bimg>0).astype(np.uint8)*255
    return bimg

def demo(model, backend, img):
    # setup input
    x = preprocess(img)
    W = {model.graph.input[0].name: x}

    # forward
    lscore, gscore = backend.run(W)

    return None

    # post processing
    mask = postprocess(img, lscore, gscore)

    return mask

if __name__ == '__main__':
    model_path = sys.argv[1]
    img_path = sys.argv[2]
    mode = 'CUDA:0' # or "CPU"
    mode = "CPU"

    print('load model')
    model = onnx.load(model_path)

    print('prepared backend')
    backend = onnx_caffe2.backend.prepare(model, device=mode)

    img = cv2.imread(img_path)

    num_run = 30
    total_time = 0
    for i in range(num_run):
        start = time.time()
        mask = demo(model, backend, img)
        eps = time.time() -start
        print('run-%d\t%.3f' % (i, eps*1000))
        if i != 0: # exclude first run
            total_time += eps 
    print('average runtime[%s] %.3fms' % (mode, total_time/(num_run-1)*1000))
    
    '''
    mask = cv2.merge([mask, mask, mask])
    result = np.concatenate((img, mask), axis=1)
    cv2.imshow('result', result)
    cv2.waitKey()
    '''

    

