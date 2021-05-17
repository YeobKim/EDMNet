import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import *
from utils import *
import torchvision.utils as utils
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from PIL import ImageFilter
from imageio import imwrite
from testdataset import *
import time
from datetime import datetime
# import torchsummary.summary as summary
# from ptflops import get_model_complexity_info
from efficientnet_pytorch import EfficientNet

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="Image Deblurring Works")
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
parser.add_argument("--test_data", type=str, default='val_blur_jpeg', help='test on Set12 or Set68')
opt = parser.parse_args()

def normalize(data):
    return data/255.

def main():
    toTensor = transforms.Compose([transforms.ToTensor()])
    toPILImg = transforms.ToPILImage()

    # Build model
    print('Loading model ...\n')
    net = MEDNet(channels=3)
    model = net.cuda()
    # summary(net, (3, 128, 128))
    load = torch.load(os.path.join(opt.logdir, 'MEDNet_122_0.0347'))
    model.load_state_dict(load)
    model.eval()
    # load data info
    print('Loading data info ...\n')
    dataset_val = Dataset(train=False).data
    # files_source = glob.glob(os.path.join('data/val', opt.test_data, '*.png'))
    # files_source.sort()
    # process data
    psnr_val = 0
    ssim_val = 0
    val_time = 0.

    start_time = datetime.now()
    print('Model Test Start!')
    print(start_time)

    for _, (blur_val, gt_val) in enumerate(dataset_val, 0):
        with torch.no_grad():
            gt_val, blur_val = Variable(gt_val.cuda()), Variable(blur_val.cuda())
            # Time check
            beforetime = time.time()
            out_val, edge_val = model(blur_val)
            aftertime = time.time() - beforetime
            val_time += aftertime
            # print(aftertime)
            out_val = torch.clamp(out_val, 0., 1.)
        psnr_test = batch_PSNR(out_val, gt_val, 1.)
        psnr_val += psnr_test
        ssim_test = batch_SSIM(out_val, gt_val, 1.)
        ssim_val += ssim_test
        # print(val_time)
        print("PSNR: %.4f SSIM: %.4f" %
            (psnr_test, ssim_test))

    psnr_val /= len(dataset_val)
    ssim_val /= len(dataset_val)
    val_time /= len(dataset_val)
    print("PSNR_val: %.4f SSIM_val: %.4f Time_avr: %.3f\n" % (psnr_val, ssim_val, val_time))

    end_time = datetime.now() - start_time
    print(end_time)

if __name__ == "__main__":
    main()
