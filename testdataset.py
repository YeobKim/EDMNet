import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata
from os import path
from utils import data_augmentation
from torch.utils.data import DataLoader
from PIL import Image

class DataLoad(udata.Dataset):

    def __init__(self, batch_size, patch_size=128, target_dir='data',train=True, keep_range=False):
        if train:
            split = 'train'
            split_path = 'train_crop'
            train_input = 'train/train_blur_jpeg'
            train_gt = 'train/train_sharp'
        else:
            split = 'val_submit_folder'
            split_path = split
            train_input = 'val_submit_folder/blur'
            train_gt = 'val_submit_folder/gt'


        path_blur = path.join(target_dir, train_input)
        scan_blur = self.scan_over_dirs_jpg(path_blur)
        path_sharp = path.join(target_dir, train_gt)
        scan_sharp = self.scan_over_dirs_png(path_sharp)
        scans = [(b, s) for b, s, in zip(scan_blur, scan_sharp)]

        # if train:
        #     random.shuffle(scans)
        #     scans = scans[0:16000]
        # else: #val
        #     #random.shuffle(scans)
        #     scans = scans[0:2999:20]

        print('train =',train,len(scans))

        #print(scans)
        # Shuffle the dataset
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.scans = scans
        self.train = train
        self.keep_range = keep_range

    def scan_over_dirs_jpg(self, dir):
        filenames = os.listdir(dir)
        folderlist = []
        fileslist = []
        for names in filenames:
            name = names.split('/')[0]
            folderlist.append(name)
        for foldername in folderlist:
            files = sorted(glob.glob(os.path.join(dir, foldername, '*.jpg')))
            # files.sort()
            fileslist.extend(files)

        return fileslist

    def scan_over_dirs_png(self, dir):
        filenames = os.listdir(dir)
        folderlist = []
        fileslist = []
        for names in filenames:
            name = names.split('/')[0]
            folderlist.append(name)
        for names in folderlist:
            files = glob.glob(os.path.join(dir, names, '*.png'))
            files.sort()
            fileslist.extend(files)

        return fileslist

    def __len__(self):
        return len(self.scans) // self.batch_size

    def __getitem__(self, idx):
        image, target = self.scans[idx]
        # print(self.scans[idx])

        blur = Image.open(image)
        gt = Image.open(target)
        if self.train:
            random_angle = int(random.random() * 5) * 90
            blur = blur.rotate(random_angle)
            gt = gt.rotate(random_angle)
            blur = np.asarray(blur, dtype=np.float32)
            gt = np.asarray(gt, dtype=np.float32)
            blur, gt = self.random_crop(blur, gt)
            blur, gt = self.train_preprocess(blur, gt)
        else:
            blur = np.asarray(blur, dtype=np.float32)
            gt = np.asarray(gt, dtype=np.float32)

        blur /= 255.0
        gt /= 255.0
        blur = np.transpose(blur, (2, 0, 1))
        gt = np.transpose(gt, (2, 0, 1))

        # sample = {'image': image, 'target': target}

        return blur, gt

    def random_crop(self, blur, sharp):
        h, w, _ = blur.shape

        py = random.randrange(0, h - self.patch_size + 1)
        px = random.randrange(0, w - self.patch_size + 1)
        crop_blur = blur[py:(py + self.patch_size), px:(px + self.patch_size)]
        crop_sharp = sharp[py:(py + self.patch_size), px:(px + self.patch_size)]

        return crop_blur, crop_sharp


    # def random_crop(lr_img, hr_img, hr_crop_size):
    #     lr_crop_size = hr_crop_size
    #
    #     lr_w = np.random.randint(lr_img.shape[1] - lr_crop_size + 1)
    #     lr_h = np.random.randint(lr_img.shape[0] - lr_crop_size + 1)
    #
    #     hr_w = lr_w
    #     hr_h = lr_h
    #
    #     lr_img_cropped = lr_img[lr_h:lr_h + lr_crop_size, lr_w:lr_w + lr_crop_size]
    #     hr_img_cropped = hr_img[hr_h:hr_h + hr_crop_size, hr_w:hr_w + hr_crop_size]
    #
    #     return lr_img_cropped, hr_img_cropped
    def train_preprocess(self, image, target):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            target = (target[:, ::-1, :]).copy()

        # Random gamma, brightness, color augmentation
        # do_augment = random.random()
        # if do_augment > 0.5:
        #     image = self.augment_image(image)

        return image, target

    def augment_image(self, image):
            # gamma augmentation
            gamma = random.uniform(0.9, 1.1)
            image_aug = image ** gamma

            # brightness augmentation

            brightness = random.uniform(0.9, 1.1)
            image_aug = image_aug * brightness

            # color augmentation
            colors = np.random.uniform(0.9, 1.1, size=3)
            white = np.ones((image.shape[0], image.shape[1]))
            color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
            image_aug *= color_image
            image_aug = np.clip(image_aug, 0, 1)

            return image_aug

class Dataset(object):
    def __init__(self, train=True, batchSize=4):
        if train:
            self.transformed_data = DataLoad(1, train=True,)
            self.data = DataLoader(self.transformed_data, batchSize, num_workers=4, shuffle=True)
        else:
            self.transformed_data = DataLoad(1, train=False,)
            self.data = DataLoader(self.transformed_data, 1, num_workers=4, shuffle=False)


    def __len__(self):
        return len(self.transformed_data)