import os
import torch
from preprocess import *
from torch.utils import data
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pdb

class ImageData(data.Dataset):
    def __init__(self, img_root, label_root, transform):
        self.image_path = [ os.path.join(img_root, x) for x in os.listdir(img_root) ]
        self.label_path = [ os.path.join(label_root, x.split('/')[-1][:-3]+'png')
                for x in self.image_path ]
        self.transform = transform

    def __getitem__(self, item):
        image = cv2.imread(self.image_path[item])
        label = cv2.imread(self.label_path[item])
        if image is None or label is None:
            print('can not open file', self.image_path[item], self.label_path[item])
            assert(False)
        image = image[:,:,(2,1,0)]
        image = image.astype(np.float32)
        if len(label.shape) == 3:
            label = label[:,:,0]
        for trans in self.transform:
            image, label = trans(image, label)
        # to tensor
        image = torch.Tensor(image).permute(2, 0, 1)/255.0
        label = torch.Tensor(label[np.newaxis,:,:])/255.0
        label = label.round()
        return image, label

    def __len__(self):
        return len(self.image_path)

def get_loader(img_root, label_root, img_size, batch_size, mode='train', num_thread=1):
    if mode == 'train':
        transform = [
            ReColor(alpha=0.05),
            GaussianNoise(),
            RandomFlip(),
            Resize((img_size,img_size), (img_size//2, img_size//2))
            ]
        shuffle = True
    else:
        transform = []
        shuffle = False
    dataset = ImageData(img_root, label_root, transform)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size,
            shuffle=shuffle, num_workers=num_thread)
    return data_loader


if __name__ ==  '__main__':
    img_root = sys.argv[1]
    label_root = sys.argv[2]

    img_size = 352
    transform = [
            ReColor(),
            GaussianNoise(),
            RandomFlip(),
            Resize((img_size,img_size), (img_size//2, img_size//2))
        ]

    dataset = ImageData(img_root, label_root, transform)
    print('number of image %d' % len(dataset))
    for i in range(len(dataset)):
        image, label = dataset[i]
        image = (image.permute(1, 2, 0).numpy()*255).astype(np.uint8)
        image = image[:, :, [2,1,0]]
        label = (label.numpy()*255).astype(np.uint8)
        label = label[0, :, :]
        cv2.imshow('image', image)
        cv2.imshow('label', label)
        cv2.waitKey()
