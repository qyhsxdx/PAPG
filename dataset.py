import os, random
from PIL import Image
from torchvision import transforms
import numpy as np
from numpy import asarray
import torch.nn as nn
import math
import torch
import matplotlib.pyplot as plt
from itertools import permutations as P
from collections import OrderedDict
import argparse
random.seed(0)


parser = argparse.ArgumentParser(description="Load dataset!")



def train_transform():
    img = transforms.Compose([
        transforms.Resize((288, 144)),
        # transforms.RandomCrop((288, 144)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    return img

def test_transform():
    img = transforms.Compose([
        transforms.Resize((288, 144)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    return img



def val_loader_ndarray(path):
    img = np.load(path)
    return img

def val_loader_img(path):
    img = Image.open(path).convert("RGB")
    return img

class Sysu_DataLoader():
    def __init__(self, imgs_path, pose_path, idx_path, transform, loader_ndarray, loader_img):
        #index
        idx_ir = loader_ndarray(idx_path + "SYSU-MM01train_ir_resized_label.npy")
        idx_rgb = loader_ndarray(idx_path + "SYSU-MM01train_rgb_resized_label.npy")

        #imgs
        images_ir = sorted(os.listdir(os.path.join(imgs_path, "train_ir")))
        rgb_pose = sorted(os.listdir(os.path.join(pose_path, "train_rgb_pose")))
        ir_pose = sorted(os.listdir(os.path.join(pose_path, "train_ir_pose")))
        images_rgb = sorted(os.listdir(os.path.join(imgs_path, "train_rgb")))

        # paired-image
        data = []
        rgb_dict = OrderedDict()
        for i in np.unique(idx_ir):
            rgb_data = []
            for j in rgb_pose:
                if int(j.split("_")[0]) == i:
                    rgb_data.append(j)
            rgb_dict[i] = rgb_data

        ir_dict = OrderedDict()
        for i in np.unique(idx_ir):
            ir_data = []
            for j in ir_pose:
                if int(j.split("_")[0]) == i:
                    ir_data.append(j)
            ir_dict[i] = ir_data

        for i in np.unique(idx_ir):
            for ir, rgb in zip(ir_dict[i], rgb_dict[i]):
                data.append((ir, rgb))


        random.shuffle(data)
        self.data = data
        self.imgs_path_ir = os.path.join(imgs_path, "train_ir")
        self.imgs_path_rgb = os.path.join(imgs_path, "train_rgb")
        self.pose_rgb = os.path.join(pose_path, "train_rgb_pose")
        self.pose_ir = os.path.join(pose_path, "train_ir_pose")
        #self.pose_rgb = os.path.join(pose_path, "train_rgb_pose_map")
        self.loader_img = loader_img
        self.transform = transform
        #self.transform_pose = transform2


    def __getitem__(self, index):
        ir_name, rgb_name = self.data[index]

        #rgb2ir
        # ir_img = self.loader_img(os.path.join(self.imgs_path_rgb, src_path))
        # tgt_img = self.loader_img(os.path.join(self.imgs_path_ir, tgt_path))
        # pose_rgb = self.loader_img(os.path.join(self.pose_ir, tgt_path))

        #ir2rgb
        rgb_img = self.loader_img(os.path.join(self.imgs_path_rgb, rgb_name))
        ir_img = self.loader_img(os.path.join(self.imgs_path_ir, ir_name))
        pose_ir = self.loader_img(os.path.join(self.pose_ir, ir_name))
        pose_rgb = self.loader_img(os.path.join(self.pose_rgb, rgb_name))

        #pose_rgb = np.load(os.path.join(self.pose_rgb, tgt_path.split(".")[0] + ".npy"))
        # pose_rgb = torch.from_numpy(pose_rgb)
        # pose_rgb = pose_rgb.transpose(2, 0)
        # pose_rgb = pose_rgb.transpose(2, 1).unsqueeze(0)
        # conv = nn.Conv2d(in_channels=18, out_channels=3, kernel_size=1, stride=1, padding=0)
        # pose_rgb = conv(pose_rgb).squeeze(0)

        rgb_img, ir_img, pose_ir, pose_rgb = self.transform(rgb_img), self.transform(ir_img), \
                                    self.transform(pose_ir), self.transform(pose_rgb)
        return {"rgb_img": rgb_img, "ir_img": ir_img, "ir_pose": pose_ir, "rgb_pose": pose_rgb,
                "ir_name": ir_name, "rgb_name": rgb_name}

    def __len__(self):
        return len(self.data)


class RegDB_DataLoader():
    def __init__(self, imgs_path, pose_path, idx_path, transform, loader_ndarray, loader_img):
        #index
        idx_ir = loader_ndarray(idx_path + "RegDB_train_ir_label.npy")
        idx_rgb = loader_ndarray(idx_path + "RegDB_train_rgb_label.npy")

        #imgs
        images_ir = sorted(os.listdir(os.path.join(imgs_path, "train_regdb_ir")))
        rgb_pose = sorted(os.listdir(os.path.join(pose_path, "train_regdb_rgb_pose")))
        ir_pose = sorted(os.listdir(os.path.join(pose_path, "train_regdb_ir_pose")))
        images_rgb = sorted(os.listdir(os.path.join(imgs_path, "train_regdb_rgb")))

        # paired-image
        data = []
        rgb_dict = OrderedDict()
        for i in np.unique(idx_ir):
            rgb_data = []
            for j in rgb_pose:
                if int(j.split("_")[0]) == i:
                    rgb_data.append(j)
            rgb_dict[i] = rgb_data

        ir_dict = OrderedDict()
        for i in np.unique(idx_ir):
            ir_data = []
            for j in ir_pose:
                if int(j.split("_")[0]) == i:
                    ir_data.append(j)
            ir_dict[i] = ir_data

        for i in np.unique(idx_ir):
            for ir, rgb in zip(ir_dict[i], rgb_dict[i]):
                data.append((ir, rgb))


        random.shuffle(data)
        self.data = data
        self.imgs_path_ir = os.path.join(imgs_path, "train_regdb_ir")
        self.imgs_path_rgb = os.path.join(imgs_path, "train_regdb_rgb")
        self.pose_rgb = os.path.join(pose_path, "train_regdb_rgb_pose")
        self.pose_ir = os.path.join(pose_path, "train_regdb_ir_pose")
        #self.pose_rgb = os.path.join(pose_path, "train_rgb_pose_map")
        self.loader_img = loader_img
        self.transform = transform
        #self.transform_pose = transform2


    def __getitem__(self, index):
        ir_name, rgb_name = self.data[index]

        #rgb2ir
        # ir_img = self.loader_img(os.path.join(self.imgs_path_rgb, src_path))
        # tgt_img = self.loader_img(os.path.join(self.imgs_path_ir, tgt_path))
        # pose_rgb = self.loader_img(os.path.join(self.pose_ir, tgt_path))

        #ir2rgb
        rgb_img = self.loader_img(os.path.join(self.imgs_path_rgb, rgb_name))
        ir_img = self.loader_img(os.path.join(self.imgs_path_ir, ir_name))
        pose_ir = self.loader_img(os.path.join(self.pose_ir, ir_name))
        pose_rgb = self.loader_img(os.path.join(self.pose_rgb, rgb_name))

        #pose_rgb = np.load(os.path.join(self.pose_rgb, tgt_path.split(".")[0] + ".npy"))
        # pose_rgb = torch.from_numpy(pose_rgb)
        # pose_rgb = pose_rgb.transpose(2, 0)
        # pose_rgb = pose_rgb.transpose(2, 1).unsqueeze(0)
        # conv = nn.Conv2d(in_channels=18, out_channels=3, kernel_size=1, stride=1, padding=0)
        # pose_rgb = conv(pose_rgb).squeeze(0)

        rgb_img, ir_img, pose_ir, pose_rgb = self.transform(rgb_img), self.transform(ir_img), \
                                    self.transform(pose_ir), self.transform(pose_rgb)
        return {"rgb_img": rgb_img, "ir_img": ir_img, "ir_pose": pose_ir, "rgb_pose": pose_rgb,
                "ir_name": ir_name, "rgb_name": rgb_name}

    def __len__(self):
        return len(self.data)









