import os
import sys

import torch
from torch.autograd import Variable
import torchvision
import cv2
import numpy as np

from nlfd import build_model

if __name__ == '__main__':
    model_path = sys.argv[1]

    # setup model
    net = build_model()
    net.load_state_dict(torch.load(model_path))
    #net = torchvision.models.alexnet(pretrained=False)
    net.cuda() 
    net.eval()

    # Input to the model
    x = Variable(torch.randn(1, 3, 352, 352), requires_grad=True).cuda()

    # Export the model
    torch_out = torch.onnx.export(net, x, "./nlfd.proto")




