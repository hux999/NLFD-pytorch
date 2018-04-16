import os
import torch
from preprocess import ReColor,RandomRotate,CurriculumWrapper,GaussianNoiseLocal,GaussianNoise,RandomFlip
from torch.utils import data
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pdb

class ImageData(data.Dataset):
    def __init__(self, img_root, label_root, transform, t_transform):
        self.image_path = list(map(lambda x: os.path.join(img_root, x), os.listdir(img_root)))
        self.label_path = list(map(lambda x: os.path.join(label_root, x.split('/')[-1][:-3] + 'png'), self.image_path))
        self.transform = transform
        self.t_transform = t_transform
        self.set_trans_prob(0.5)

    def __getitem__(self, item):
        image_cv2 = cv2.imread(self.image_path[item])
        label_cv2 = cv2.imread(self.label_path[item])
        image = image_cv2[:,:,(2,1,0)]
        label = label_cv2[:,:,(2,1,0)]
        assert(label is not None)
        for trans in self.trans_all:
            image, label = trans(image, label)
        image= Image.fromarray(image)
        label= Image.fromarray(label).convert("L")
        #pre-process
        if self.transform is not None:
            image = self.transform(image)
        if self.t_transform is not None:
            label = self.t_transform(label)
        return image, label

    def __len__(self):
        return len(self.image_path)

    def set_trans_prob(self, prob):
        self.trans_prob = prob
        self.trans_all = [ CurriculumWrapper(ReColor(alpha=0.05), prob), \
                        CurriculumWrapper(GaussianNoiseLocal(diff=20), prob), \
                        CurriculumWrapper(RandomFlip(), prob), 
                        ]

def get_loader(img_root, label_root, img_size, batch_size, mode='train', num_thread=1):
    shuffle = False
    #mean = torch.Tensor([123.68, 116.779, 103.939]).view(3, 1, 1) / 255
    if mode == 'train':
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            #transforms.Lambda(lambda x: x - mean)
        ])
        t_transform = transforms.Compose([
            transforms.Resize((img_size // 2, img_size // 2)),
            transforms.ToTensor(),
            #transforms.Lambda(lambda x: torch.round(x))  # TODO: it maybe unnecessary
        ])
        shuffle = True
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x - mean)
        ])
        t_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.round(x))  # TODO: it maybe unnecessary
        ])
    dataset = ImageData(img_root, label_root, transform, t_transform)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_thread)
    return data_loader


# if __name__ ==  '__main__':
    # ima_root = '/home/tinzhuo/ML_SalientObject/NLFD-pytorch/MSRA/images/0_0_147.jpg'
    # label_root = '/home/tinzhuo/ML_SalientObject/NLFD-pytorch/MSRA/ground_truth_mask/0_0_147.png'
    # image = cv2.imread(ima_root)
    # label = cv2.imread(label_root)

    # cv2.imshow('img', image)

    # ImageData A = ImageData()
    # A.set_trans_prob(1)
    # for trans in self.trans_all:
    #         image, label = trans(image, label)

    #     # cv2.imshow('img1', image)
    #     # cv2.waitKey(0)
    # image= Image.fromarray(image)
    # label= Image.fromarray(label).convert("L")