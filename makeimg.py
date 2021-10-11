import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models.net import *
from utils.utils import *
import torchvision.utils as utils
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from PIL import ImageFilter
from imageio import imwrite
from utils.dataset import *
import time
from datetime import datetime
# import torchsummary.summary as summary
# from ptflops import get_model_complexity_info

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="EDMNet:Deblurring Network Using Edge Module, Deformable Convolution and Multi-Stage Network.")
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
parser.add_argument("--test_data", type=str, default='data/GOPRO/test/blur/', help='Gopro Testset')
opt = parser.parse_args()

def normalize(data):
    return data/255.

def main():
    # Build model
    print('Loading model ...\n')
    net = edmnet()
    model = net.cuda()

    # summary(net, (3, 128, 128))
    load = torch.load(os.path.join(opt.logdir, 'edmnet_1000_0.0153.pth'))
    model.load_state_dict(load)
    model.eval()

    # load data info
    print('Loading data info ...\n')

    filenames = os.listdir(opt.test_data)
    folderlist = []
    fileslist = []
    for names in filenames:
        name = names.split('/')[0]
        folderlist.append(name)
    for names in folderlist:
        files = glob.glob(os.path.join(opt.test_data, names, '*.png'))
        files.sort()
        fileslist.extend(files)

    print('Model Test Start!')

    for f in fileslist:
        # image
        Img = Image.open(f)
        Img = normalize(np.asarray(Img, dtype=np.float32))
        Img = np.transpose(Img, (2, 0, 1))
        Img = np.expand_dims(Img, 0)
        blur_val = torch.Tensor(Img)

        filename = f.split('/blur/')[1]

        with torch.no_grad():
            blur_val = Variable(blur_val.cuda())

            out_val = model(blur_val)
            out_val = torch.clamp(out_val[0], 0., 1.)


        clean_img = utils.make_grid(out_val.data, nrow=8, normalize=True, scale_each=True)

        print(filename, 'finished!')
        result_img = torch.clamp(clean_img * 255, 0, 255)
        result_img = np.uint8(result_img.cpu())
        np_img = np.transpose(result_img, (1, 2, 0))

        result = Image.fromarray(np_img)

        result.save('data/output/edmnet/' + filename, format='PNG')



if __name__ == "__main__":
    main()
