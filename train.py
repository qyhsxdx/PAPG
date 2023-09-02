import argparse
import itertools
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
import torch.utils.data as Data
from tensorboardX import SummaryWriter

from models import Generator, Discriminator
from utils import ReplayBuffer, LambdaLR, Logger, weights_init_normal,save_images
import dataset

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="Starting epoch")
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--input_nc_pose', type=int, default=6, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument("--gpu", default="2", type=str, help="gpu device ids for CUDA_VISIBLE_DEVICES")
parser.add_argument("--cuda", action='store_true', help='use GPU computation')
parser.add_argument("--save_model_path", default="./output", type=str)
parser.add_argument("--log_path", default="./", type=str)
parser.add_argument("--save_img_path", default="./images", type=str)
parser.add_argument("--img_h", default=288, type=int)
parser.add_argument("--img-w", default=144, type=int)
parser.add_argument("--add_att", default="on", type=str)
parser.add_argument("--add_pose", default="on", type=str)
parser.add_argument("--dataset", default="sysu", type=str)

parser.add_argument("--imgs_path", default="make_pose", type=str, help="train dataset path")
parser.add_argument("--pose_path", default="make_pose", type=str, help="pose dataset path")
parser.add_argument("--idx_path", default="datasets/idx/", type=str, help="idx path")
parser.add_argument("--num_workers", default=8, type=int, help="number of cpu threads to use during batch generation")
parser.add_argument("--lambda_identity", default=5.0, type=float)
parser.add_argument("--lambda_cycle", default=10.0, type=float)

opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


summary = SummaryWriter(log_dir="%s" % (os.path.join(opt.log_path, "log")), comment="")

#----Networks----#
if opt.add_pose == "on":
    netG_A2B = Generator(opt.input_nc_pose, opt.add_att, opt.add_pose)
    netG_B2A = Generator(opt.input_nc_pose, opt.add_att, opt.add_pose)
else:
    netG_A2B = Generator(opt.input_nc, opt.add_att, opt.add_pose)
    netG_B2A = Generator(opt.input_nc, opt.add_att, opt.add_pose)
netD_A = Discriminator(opt.output_nc)
netD_B = Discriminator(opt.output_nc)

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()
    netD_A.cuda()
    netD_B.cuda()

netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)


#----losses----#
criterion_GAN = torch.nn.MSELoss().cuda()
criterion_cycle = torch.nn.L1Loss().cuda()
criterion_identity = torch.nn.L1Loss().cuda()

#----Optimizers & LR schedulers----#
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

#----Inputs & targets memory allocation----#
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, 3, opt.img_h, opt.img_w)
input_B = Tensor(opt.batchSize, 3, opt.img_h, opt.img_w)
input_A_pose = Tensor(opt.batchSize, 3, opt.img_h, opt.img_w)
input_B_pose = Tensor(opt.batchSize, 3, opt.img_h, opt.img_w)

def target_real(input):
    target_real = Variable(Tensor(input.size()).fill_(1.0), requires_grad=False)
    return target_real

def target_fake(input):
    target_fake = Variable(Tensor(input.size()).fill_(0.0), requires_grad = False)
    return target_fake


fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

#----Dataset loader----#
if opt.dataset == "sysu":
    train_data = dataset.Sysu_DataLoader(imgs_path=opt.imgs_path, pose_path=opt.pose_path, idx_path=opt.idx_path,
                                               transform = dataset.train_transform(),
                                             loader_ndarray = dataset.val_loader_ndarray,
                                             loader_img = dataset.val_loader_img)
    train_loader = Data.DataLoader(train_data, batch_size = opt.batchSize, shuffle = False, num_workers = opt.num_workers,
                                       drop_last=True)
elif opt.dataset == "regdb":
    train_data = dataset.RegDB_DataLoader(imgs_path=opt.imgs_path, pose_path=opt.pose_path, idx_path=opt.idx_path,
                                         transform=dataset.train_transform(),
                                         loader_ndarray=dataset.val_loader_ndarray,
                                         loader_img=dataset.val_loader_img)
    train_loader = Data.DataLoader(train_data, batch_size=opt.batchSize, shuffle=False, num_workers=opt.num_workers,
                                   drop_last=True)

    #----Loss plot----#
#logger = Logger(opt.n_epochs, len(train_loader))


##################################################
#----Training----#
count = 0
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(train_loader):
        #set model input
        real_A = Variable(input_A.copy_(batch["rgb_img"])).cuda()  #rgb images
        real_B = Variable(input_B.copy_(batch["ir_img"])).cuda()  #ir images
        A_pose = Variable(input_A_pose.copy_(batch["rgb_pose"])).cuda()
        B_pose = Variable(input_B_pose.copy_(batch["ir_pose"])).cuda()

        #-----------Generators A2B and B2A-------------------#
        optimizer_G.zero_grad()

        #Identity loss
        same_B = netG_A2B(real_B, B_pose)
        loss_identity_B = criterion_identity(same_B, real_B) * opt.lambda_identity

        same_A = netG_B2A(real_A, A_pose)
        loss_identity_A = criterion_identity(same_A, real_A) * opt.lambda_identity

        #GAN loss
        fake_B = netG_A2B(real_A, A_pose)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real(pred_fake))

        fake_A = netG_B2A(real_B, B_pose)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real(pred_fake))

        #Cycle loss
        recovered_A = netG_B2A(fake_B, A_pose)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * opt.lambda_cycle

        recovered_B = netG_A2B(fake_A, B_pose)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * opt.lambda_cycle

        #Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()

        optimizer_G.step()

        #----------------------Discriminator A-----------------#
        optimizer_D_A.zero_grad()

        #real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real(pred_real))

        #fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake(pred_fake))

        #total loss
        loss_D_A = (loss_D_real + loss_D_fake) * 0.5
        loss_D_A.backward()
        optimizer_D_A.step()


        #---------------Discriminator B-------------------#
        optimizer_D_B.zero_grad()

        #real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real(pred_real))

        #fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake(pred_fake))

        #total loss
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5
        loss_D_B.backward()

        optimizer_D_B.step()

        # --------Update log and display loss info.------#
        count += 1
        summary.add_scalar("G_loss", loss_G.item(), count)
        summary.add_scalar("loss_G_identity", (loss_identity_A.item() + loss_identity_B.item()), count)
        summary.add_scalar("loss_G_GAN", (loss_GAN_A2B.item() + loss_GAN_B2A.item()), count)
        summary.add_scalar("loss_G_cycle", (loss_cycle_ABA.item() + loss_cycle_BAB.item()), count)
        summary.add_scalar("loss_D", (loss_D_A.item() + loss_D_B.item()), count)

        #-------------------------------------------------------------#
        print("Epoch:{}/{} | Step:{}/{} | lr:{:.6f} | loss_G:{:.6f} | loss_G_identity:{:.6f} "
              "| loss_G_GAN:{:.6f} | loss_G_cycle:{:.6f} | loss_D:{:.6f}"
            .format(epoch, opt.n_epochs, i + 1, len(train_loader), optimizer_G.param_groups[0]["lr"],
                    loss_G.item(), (loss_identity_A.item() + loss_identity_B.item()),
                    (loss_GAN_A2B.item() + loss_GAN_B2A.item()),
                    (loss_cycle_ABA.item() + loss_cycle_BAB.item()), (loss_D_A.item() + loss_D_B.item())))


        # Progress report (http://localhost:8097)
        # logger.log({'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B),
        #             'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
        #             'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)},
        #            images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    save_images(opt.save_img_path, epoch, real_A, fake_B, real_B, fake_A, summary)

    # Save models checkpoints
    print("Saving model-------------------------------")
    if not os.path.exists(opt.save_model_path):
        os.mkdir(opt.save_model_path)
    torch.save(netG_A2B.state_dict(), f'output/netG_A2B_{epoch}.pth')
    torch.save(netG_B2A.state_dict(), f'output/netG_B2A_{epoch}.pth')
    torch.save(netD_A.state_dict(), f'output/netD_A_{epoch}.pth')
    torch.save(netD_B.state_dict(), f'output/netD_B_{epoch}.pth')

    print("Finished----------------------------------")
summary.close()



















