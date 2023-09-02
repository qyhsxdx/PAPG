import argparse
import sys
import itertools
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
from torchvision.utils import save_image
import torch.utils.data as Data
import dataset
from models import Generator

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--input_nc_pose', type=int, default=6, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument("--gpu", default="0", type=str, help="gpu device ids for CUDA_VISIBLE_DEVICES")
parser.add_argument("--cuda", action='store_true', help='use GPU computation')
parser.add_argument('--generator_A2B', type=str, default='output/netG_A2B.pth', help='A2B generator checkpoint file')
parser.add_argument('--generator_B2A', type=str, default='output/netG_B2A.pth', help='B2A generator checkpoint file')
parser.add_argument("--img_h", default=288, type=int)
parser.add_argument("--img-w", default=144, type=int)
parser.add_argument("--add_att", default="on", type=str)
parser.add_argument("--add_pose", default="on", type=str)
parser.add_argument("--dataset", default="sysu", type=str)

parser.add_argument("--imgs_path", default="make_pose", type=str, help="train dataset path")
parser.add_argument("--pose_path", default="make_pose", type=str, help="pose dataset path")
parser.add_argument("--idx_path", default="datasets/idx/", type=str, help="idx path")
parser.add_argument("--num_workers", default=8, type=int, help="number of cpu threads to use during batch generation")

opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

#Networks
if opt.add_pose == "on":
    netG_A2B = Generator(opt.input_nc_pose, opt.add_att, opt.add_pose)
    netG_B2A = Generator(opt.input_nc_pose, opt.add_att, opt.add_pose)
else:
    netG_A2B = Generator(opt.input_nc, opt.add_att, opt.add_pose)
    netG_B2A = Generator(opt.input_nc, opt.add_att, opt.add_pose)

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()

#load state dicts
netG_A2B.load_state_dict(torch.load(opt.generator_A2B))
netG_B2A.load_state_dict(torch.load(opt.generator_B2A))

#set model's test mode
netG_A2B.eval()
netG_B2A.eval()
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, 3, opt.img_h, opt.img_w)
input_B = Tensor(opt.batchSize, 3, opt.img_h, opt.img_w)
input_A_pose = Tensor(opt.batchSize, 3, opt.img_h, opt.img_w)
input_B_pose = Tensor(opt.batchSize, 3, opt.img_h, opt.img_w)

#----Dataset loader----#
if opt.dataset == "sysu":
    train_data = dataset.Sysu_DataLoader(imgs_path=opt.imgs_path, pose_path=opt.pose_path, idx_path=opt.idx_path,
                                               transform = dataset.test_transform(),
                                             loader_ndarray = dataset.val_loader_ndarray,
                                             loader_img = dataset.val_loader_img)
    train_loader = Data.DataLoader(train_data, batch_size = opt.batchSize, shuffle = False, num_workers = opt.num_workers,
                                       drop_last=True)
elif opt.dataset == "regdb":
    train_data = dataset.RegDB_DataLoader(imgs_path=opt.imgs_path, pose_path=opt.pose_path, idx_path=opt.idx_path,
                                         transform=dataset.test_transform(),
                                         loader_ndarray=dataset.val_loader_ndarray,
                                         loader_img=dataset.val_loader_img)
    train_loader = Data.DataLoader(train_data, batch_size=opt.batchSize, shuffle=False, num_workers=opt.num_workers,
                                   drop_last=True)

#----------Testing-------------#

#Create output dirs if they don't exist
if not os.path.exists("output/A"):
    os.makedirs("output/A")
if not os.path.exists("output/B"):
    os.makedirs("output/B")

for i, batch in enumerate(train_loader):
    #Set model input
    real_A = Variable(input_A.copy_(batch["rgb_img"])).cuda()  # rgb images
    real_B = Variable(input_B.copy_(batch["ir_img"])).cuda()  # ir images
    A_pose = Variable(input_A_pose.copy_(batch["rgb_pose"])).cuda()
    B_pose = Variable(input_B_pose.copy_(batch["ir_pose"])).cuda()
    ir_name = batch["ir_name"]
    rgb_name = batch["rgb_name"]

    # Generate output
    fake_B = 0.5 * (netG_A2B(real_A, A_pose).data + 1.0)
    fake_A = 0.5 * (netG_B2A(real_B, B_pose).data + 1.0)

    # Save image files
    save_image(fake_A, f'output/A/{ir_name[0].split(".")[0] + "_" }.png')
    save_image(fake_B, f'output/B/{rgb_name[0].split(".")[0] + "_" }.png')

    sys.stdout.write('\rGenerated images %04d of %04d' % (i + 1, len(train_loader)))

sys.stdout.write('\n')






















