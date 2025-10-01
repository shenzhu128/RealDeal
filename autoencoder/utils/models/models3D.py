import torch
import torch.nn as nn
import torch.nn.functional as F

class Downsample3d(nn.Module):
    def __init__(self, in_channels, out_channels, nonlinearity):
        super(Downsample3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nonlinearity = nonlinearity

        self.conv = torch.nn.Conv3d(in_channels, out_channels, kernel_size = 3, padding = 1)

    def forward(self, x):
        y = torch.nn.functional.interpolate(x, scale_factor = 0.5, mode = "trilinear")
        y = self.nonlinearity(self.conv(y))
        return(y)

class Upsample3d(nn.Module):
    def __init__(self, in_channels, out_channels, nonlinearity):
        super(Upsample3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nonlinearity = nonlinearity

        self.conv = torch.nn.Conv3d(in_channels, out_channels, kernel_size = 3, padding = 1)

    def forward(self, x):
        y = torch.nn.functional.interpolate(x, scale_factor = 2.0, mode = "nearest")
        y = self.nonlinearity(self.conv(y))
        return(y)

class ResNetBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, nonlinearity, use_batchnorm = True):
        super(ResNetBlock3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nonlinearity = nonlinearity

        self.conv1 = torch.nn.Conv3d(in_channels, out_channels, kernel_size = 3, padding = 1)
        self.conv2 = torch.nn.Conv3d(out_channels, out_channels, kernel_size = 3, padding = 1)
        if(use_batchnorm):
            self.bn1 = torch.nn.BatchNorm3d(out_channels)
            self.bn2 = torch.nn.BatchNorm3d(out_channels)
        else:
            self.bn1 = torch.nn.Identity()
            self.bn2 = torch.nn.Identity()

        if(in_channels != out_channels):
            self.skip = torch.nn.Conv3d(in_channels, out_channels, kernel_size = 1)
        else:
            self.skip = torch.nn.Identity()

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.nonlinearity(y)

        y = self.conv2(x)
        y = self.bn2(y)

        y = y + self.skip(x)
        y = self.nonlinearity(y)

        return(y)

class ResNetAutoencoder3d(nn.Module):
    def __init__(self, num_channels, nonlinearity, use_batchnorm = True):
        super(ResNetAutoencoder3d, self).__init__()
        self.num_channels = num_channels
        self.nonlinearity = nonlinearity
        self.use_batchnorm = use_batchnorm

        self.in_conv = torch.nn.Conv3d(1, num_channels, kernel_size = 3, padding = 1)
        self.eresblock1 = ResNetBlock3d(num_channels, num_channels, nonlinearity, use_batchnorm = False)
        #self.down1 = torch.nn.Conv3d(num_channels, num_channels, kernel_size = 3, stride = 2, padding = 1)
        self.down1 = Downsample3d(num_channels, 2*num_channels, nonlinearity)
        self.eresblock2 = ResNetBlock3d(2*num_channels, 2*num_channels, nonlinearity, use_batchnorm = False)
        #self.down2 = torch.nn.Conv3d(num_channels, num_channels, kernel_size = 3, stride = 2, padding = 1)
        self.down2 = Downsample3d(2*num_channels, num_channels, nonlinearity)
        self.eresblock3 = ResNetBlock3d(num_channels, num_channels, nonlinearity, use_batchnorm = False)
        self.echan_conv1 = torch.nn.Conv3d(num_channels, 32, kernel_size = 1)
        self.echan_conv2 = torch.nn.Conv3d(32, 4, kernel_size = 1)

        if(use_batchnorm):
            self.ebn1 = torch.nn.BatchNorm3d(num_channels)
            self.ebn2 = torch.nn.BatchNorm3d(2*num_channels)
            self.ebn3 = torch.nn.BatchNorm3d(num_channels)
            self.ebn4 = torch.nn.BatchNorm3d(32)

            self.dbn1 = torch.nn.BatchNorm3d(num_channels)
            self.dbn2 = torch.nn.BatchNorm3d(2*num_channels)
            self.dbn3 = torch.nn.BatchNorm3d(num_channels)
            self.dbn4 = torch.nn.BatchNorm3d(1)
        else:
            self.ebn1 = torch.nn.Identity()
            self.ebn2 = torch.nn.Identity()
            self.ebn3 = torch.nn.Identity()
            self.ebn4 = torch.nn.Identity()

            self.dbn1 = torch.nn.Identity()
            self.dbn2 = torch.nn.Identity()
            self.dbn3 = torch.nn.Identity()
            self.dbn4 = torch.nn.Identity()

        self.dchan_conv1 = torch.nn.Conv3d(4, 32, kernel_size = 1)
        self.dchan_conv2 = torch.nn.Conv3d(32, num_channels, kernel_size = 1)
        self.dresblock1 = ResNetBlock3d(num_channels, num_channels, nonlinearity, use_batchnorm = False)
        #self.upconv1 = torch.nn.Conv3d(num_channels, num_channels, kernel_size = 3, padding = 1)
        self.up1 = Upsample3d(num_channels, 2*num_channels, nonlinearity)
        self.dresblock2 = ResNetBlock3d(2*num_channels, 2*num_channels, nonlinearity, use_batchnorm = False)
        #self.upconv2 = torch.nn.Conv3d(num_channels, num_channels, kernel_size = 3, padding = 1)
        self.up2 = Upsample3d(2*num_channels, num_channels, nonlinearity)
        self.dresblock3 = ResNetBlock3d(num_channels, num_channels, nonlinearity, use_batchnorm = False)
        self.out_conv = torch.nn.Conv3d(num_channels, 1, kernel_size = 3, padding = 1)

    def encoder(self, x):
        z = self.nonlinearity(self.ebn1(self.in_conv(x)))
        z = self.eresblock1(z)
        z = self.ebn2(self.down1(z))
        z = self.eresblock2(z)
        z = self.ebn3(self.down2(z))
        z = self.eresblock3(z)
        z = self.nonlinearity(self.ebn4(self.echan_conv1(z)))
        z = torch.sigmoid(self.echan_conv2(z))
        #z = z + ((z * 32.0).round() / 32.0).detach() - z.detach()
        #z = self.echan_conv3(z)
        #z = 2.0 * torch.sigmoid(self.echan_conv2(z)) - 1.0
        #z = 0.5 * (torch.clamp(self.echan_conv(z), -1.0, 1.0) + 1.0)
        #z = 2.0 * (z + ((z * 32.0).round() / 32.0).detach() - z.detach()) - 1.0
        return(z)

    def decoder(self, z):
        #y = self.dchan_conv0(z)
        y = self.nonlinearity(self.dchan_conv1(z))
        y = self.nonlinearity(self.dbn1(self.dchan_conv2(y)))
        y = self.dresblock1(y)
        #y = F.interpolate(y, scale_factor = 2.0, mode = "nearest")
        #y = self.upconv1(y)
        y = self.dbn2(self.up1(y))
        y = self.dresblock2(y)
        #y = F.interpolate(y, scale_factor = 2.0, mode = "nearest")
        #y = self.upconv2(y)
        y = self.dbn3(self.up2(y))
        y = self.dresblock3(y)
        y = self.out_conv(y)
        y = torch.sigmoid(y) * 2.0 - 1.0
        return(y)

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return(y)
