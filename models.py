import torch.nn as nn
import torch.nn.functional as F
import torch


class Non_local(nn.Module):
    def __init__(self, in_channels, reduc_ratio = 2):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = in_channels // reduc_ratio

        self.g = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                      padding=0)
        )

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels)
        )

        nn.init.constant_(self.W[1].weight, 0.)
        nn.init.constant_(self.W[1].bias, 0.)


        self.theta = nn.Conv2d(in_channels = self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels = self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """

        :param x: (b, c, t, h, w)
        :return:
        """
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        #f_div_C = torch.nn.functional.softmax(f, dim = 1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])

        W_y = self.W(y)

        z = W_y + x
        return z

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, kernel_size=3),
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, kernel_size=3),
                      nn.InstanceNorm2d(in_features)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)
    

class ResidualBlock_Non_local(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock_Non_local, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, kernel_size=3),
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, kernel_size=3),
                      nn.InstanceNorm2d(in_features)]
        self.conv_block = nn.Sequential(*conv_block)
        self.non_local_att = Non_local(in_features)

    def forward(self, x):
        return self.non_local_att(x + self.conv_block(x))


class Non_local_attention(nn.Module):
    def __init__(self, in_features):
        super(Non_local_attention, self).__init__()

        self.non_local_att = Non_local(in_features)

    def forward(self, x):
        return self.non_local_att(x)


class Generator(nn.Module):
    def __init__(self, input_nc, add_att = "on" , add_pose = "on", n_residual_blocks = 9):
        super(Generator, self).__init__()

        self.add_att = add_att
        self.add_pose = add_pose
        #initial convolution block
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, 64, kernel_size=7),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(inplace=True)]

        #downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2

        #Attention blocks
        for _ in range(n_residual_blocks):
            # in_features 256
            if self.add_att == "on":
                #model += [Non_local_attention(in_features)]
                model += [ResidualBlock_Non_local(in_features)]
            else:
                model += [ResidualBlock(in_features)]

        #upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2

        #output layer
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, 3, kernel_size=7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input, pose):
        if self.add_pose == "on":
            x = torch.cat((input, pose), dim=1)
        else:
            x = input
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        model = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(256, 512, 4, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

















