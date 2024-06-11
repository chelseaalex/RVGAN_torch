import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.optim import Adam
import numpy as np
from src.losses import *
#from losses import *

class ReflectionPadding2D(nn.Module):
    def __init__(self, padding=(1, 1)):
        super(ReflectionPadding2D, self).__init__()
        if isinstance(padding, int):
            padding = (padding, padding)
        self.padding = padding

    def forward(self, x):
        w_pad, h_pad = self.padding
        return F.pad(x, (w_pad, w_pad, h_pad, h_pad), mode='reflect')

    
class NovelResidualBlock(nn.Module):
    def __init__(self, filters):
        super(NovelResidualBlock, self).__init__()
        self.reflection_pad1 = ReflectionPadding2D((1, 1))
        self.separable_conv1 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=0, groups=1)
        self.batch_norm1 = nn.BatchNorm2d(filters)
        self.leaky_relu1 = nn.LeakyReLU(0.2)

        self.reflection_pad2 = ReflectionPadding2D((1, 1))
        self.separable_conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=0, groups=1)
        self.batch_norm2 = nn.BatchNorm2d(filters)
        self.leaky_relu2 = nn.LeakyReLU(0.2)

        self.reflection_pad3 = ReflectionPadding2D((1, 1))
        self.separable_conv3 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=0, groups=1)
        self.batch_norm3 = nn.BatchNorm2d(filters)
        self.leaky_relu3 = nn.LeakyReLU(0.2)

    def forward(self, x):
        x1 = self.reflection_pad1(x)
        x1 = self.separable_conv1(x1)
        x1 = self.batch_norm1(x1)
        x1 = self.leaky_relu1(x1)

        x2 = self.reflection_pad2(x1)
        x2 = self.separable_conv2(x2)
        x2 = self.batch_norm2(x2)
        x2 = self.leaky_relu2(x2)

        x3 = self.reflection_pad3(x)
        x3 = self.separable_conv3(x3)
        x3 = self.batch_norm3(x3)
        x3 = self.leaky_relu3(x3)

        x3 = F.interpolate(x3, size=(x2.size(2), x2.size(3)), mode='nearest')

        return x + x2 + x3
    


class DiscResBlock(nn.Module):
    def __init__(self, filters):
        super(DiscResBlock, self).__init__()
        self.reflection_pad1 = ReflectionPadding2D((1, 1))
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=2, stride=1, dilation=2, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(filters)
        self.leaky_relu1 = nn.LeakyReLU(0.2)

        self.reflection_pad2 = ReflectionPadding2D((1, 1))
        self.separable_conv = nn.Conv2d(filters, filters, kernel_size=2, stride=1, dilation=2, padding=0, groups=filters)
        self.batch_norm2 = nn.BatchNorm2d(filters)
        self.leaky_relu2 = nn.LeakyReLU(0.2)

    def forward(self, x):
        x1 = self.reflection_pad1(x)
        x1 = self.conv1(x1)
        x1 = self.batch_norm1(x1)
        x1 = self.leaky_relu1(x1)

        x2 = self.reflection_pad2(x)
        x2 = self.separable_conv(x2)
        x2 = self.batch_norm2(x2)
        x2 = self.leaky_relu2(x2)

        x_add = x1 + x2
        out = x + x_add

        return out


class SFA(nn.Module):
    def __init__(self, filters, i):
        super(SFA, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(filters)
        self.leaky_relu1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(filters)
        self.leaky_relu2 = nn.LeakyReLU(0.2)

    def forward(self, x):
        x_input = x
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.leaky_relu1(x)
        x = x + x_input

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.leaky_relu2(x)
        x = x + x_input

        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.leaky_relu(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        x = self.leaky_relu(x)
        return x


class CoarseGenerator(nn.Module):
    def __init__(self, img_shape=(3, 64, 64), mask_shape=(1, 64, 64), ncf=64, n_downsampling=2, n_blocks=9, n_channels=1):
        super(CoarseGenerator, self).__init__()
        self.reflection_pad = ReflectionPadding2D((3, 3))
        self.conv1 = nn.Conv2d(img_shape[0] + mask_shape[0], ncf, kernel_size=7, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(ncf)
        self.leaky_relu1 = nn.LeakyReLU(0.2)


        self.encoder_blocks = nn.ModuleList()
        self.conv1x1_blocks = nn.ModuleList()
        for i in range(n_downsampling):
            self.encoder_blocks.append(EncoderBlock(ncf * 2**i, ncf * 2**(i + 1)))
            self.conv1x1_blocks.append(nn.Conv2d(ncf * 2**(i + 1), ncf * 2**(i + 1), kernel_size=1, stride=1, padding=0))

        self.residual_blocks = nn.ModuleList()
        for i in range(n_blocks):
            self.residual_blocks.append(NovelResidualBlock(ncf * 2**n_downsampling))

        self.decoder_blocks = nn.ModuleList()
        self.sfa_blocks = nn.ModuleList()
        for i in range(n_downsampling):
            self.decoder_blocks.append(DecoderBlock(ncf * 2**(n_downsampling - i), ncf * 2**(n_downsampling - i - 1)))
            self.sfa_blocks.append(SFA(ncf * 2**(n_downsampling - i - 1), i))

        self.final_reflection_pad = ReflectionPadding2D((3, 3))
        self.final_conv = nn.Conv2d(ncf, n_channels, kernel_size=7, stride=1, padding=0)
        self.tanh = nn.Tanh()

    def forward(self, x, mask):
        x = torch.cat([x, mask], dim=1)
        x = self.reflection_pad(x)
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.leaky_relu1(x)

        skips = []
        for encoder, conv1x1 in zip(self.encoder_blocks, self.conv1x1_blocks):
            x = encoder(x)
            skips.append(conv1x1(x))

        for residual in self.residual_blocks:
            x = residual(x)

        for i, (decoder, sfa) in enumerate(zip(self.decoder_blocks, self.sfa_blocks)):
            x = x + skips[-(i + 1)]
            x = decoder(x)
            x = sfa(x)
        feature_out=x
        x = self.final_reflection_pad(x)
        x = self.final_conv(x)
        x = self.tanh(x)
        return x,feature_out


class FineGenerator(nn.Module):
    def __init__(self, x_coarse_shape=(64, 64, 64), input_shape=(3, 128, 128), mask_shape=(1, 128, 128), nff=64, n_blocks=3, n_coarse_gen=1, n_channels=1):
        super(FineGenerator, self).__init__()
        self.n_coarse_gen = n_coarse_gen

        self.reflection_pad = ReflectionPadding2D((3, 3))
        self.conv1 = nn.Conv2d(input_shape[0] + mask_shape[0], nff, kernel_size=7, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(nff)
        self.leaky_relu1 = nn.LeakyReLU(0.2)

        self.encoder_blocks = nn.ModuleList()
        self.conv1x1_blocks = nn.ModuleList()
        for i in range(n_coarse_gen):
            self.encoder_blocks.append(EncoderBlock(nff * 2**i, nff * 2**(i + 1)))
            self.conv1x1_blocks.append(nn.Conv2d(nff * 2**(i + 1), nff * 2**(i + 1), kernel_size=1, stride=1, padding=0))

        self.residual_blocks = nn.ModuleList()
        for i in range(n_blocks):
            self.residual_blocks.append(NovelResidualBlock(nff * 2**(n_coarse_gen)))

        self.decoder_blocks = nn.ModuleList()
        self.sfa_blocks = nn.ModuleList()
        for i in range(n_coarse_gen):
            self.decoder_blocks.append(DecoderBlock(nff * 2**(n_coarse_gen - i), nff * 2**(n_coarse_gen - i - 1)))
            self.sfa_blocks.append(SFA(nff * 2**(n_coarse_gen - i - 1), i))

        self.final_reflection_pad = ReflectionPadding2D((3, 3))
        self.final_conv = nn.Conv2d(nff, n_channels, kernel_size=7, stride=1, padding=0)
        self.tanh = nn.Tanh()

        self.adjust_channels = nn.Conv2d(x_coarse_shape[0], x_coarse_shape[0] * 2, kernel_size=1, stride=1, padding=0)

    def forward(self, x, mask, x_coarse):
        x = torch.cat([x, mask], dim=1)
        x = self.reflection_pad(x)
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.leaky_relu1(x)

        skips = []
        for encoder, conv1x1 in zip(self.encoder_blocks, self.conv1x1_blocks):
            x = encoder(x)
            skips.append(conv1x1(x))

        x_coarse = self.adjust_channels(x_coarse)
        x = x + x_coarse

        for residual in self.residual_blocks:
            x = residual(x)

        for i, (decoder, sfa) in enumerate(zip(self.decoder_blocks, self.sfa_blocks)):
            x = x + skips[-(i + 1)]
            x = decoder(x)
            x = sfa(x)

        x = self.final_reflection_pad(x)
        x = self.final_conv(x)
        x = self.tanh(x)

        return x


class DiscriminatorAE(nn.Module):
    def __init__(self, input_shape_fundus=(3, 128,128), input_shape_label=(1,128,128), ndf=32, n_layers=3, activation='tanh'):
        super(DiscriminatorAE, self).__init__()
        self.activation = activation

        in_channels = input_shape_fundus[0] + input_shape_label[0]
        
        self.encoder_block1 = EncoderBlock(in_channels, ndf)
        self.res_block1 = DiscResBlock(ndf)
        
        self.encoder_block2 = EncoderBlock(ndf, ndf)
        self.res_block2 = DiscResBlock(ndf)
        
        self.encoder_block3 = EncoderBlock(ndf, ndf)
        self.res_block3 = DiscResBlock(ndf)
        
        self.decoder_block1 = DecoderBlock(ndf, ndf)
        self.decoder_block2 = DecoderBlock(ndf, ndf)
        self.decoder_block3 = DecoderBlock(ndf, ndf)
        
        self.final_conv = nn.Conv2d(ndf, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x_fundus, x_label):
        x = torch.cat((x_fundus, x_label), dim=1)
        
        x_down1 = self.encoder_block1(x)
        x_down1_res = self.res_block1(x_down1)
        
        x_down2 = self.encoder_block2(x_down1_res)
        x_down2_res = self.res_block2(x_down2)
        
        x_down3 = self.encoder_block3(x_down2_res)
        x_down3_res = self.res_block3(x_down3)
        
        x_up1 = self.decoder_block1(x_down3_res)
        
        x_up2 = self.decoder_block2(x_up1)
        
        x_up3 = self.decoder_block3(x_up2)
        
        x = self.final_conv(x_up3)
        
        # Adjust the final convolution padding to get the correct size
        if self.activation == 'tanh':
            x = torch.tanh(x)
        else:
            x = torch.sigmoid(x)

        return x



class RVGAN(nn.Module):
    def __init__(self, g_model_fine, g_model_coarse, d_model1, d_model2, inner_weight):
        super(RVGAN, self).__init__()
        self.g_model_fine = g_model_fine
        self.g_model_coarse = g_model_coarse
        self.d_model1 = d_model1
        self.d_model2 = d_model2
        self.inner_weight = inner_weight

    def forward(self, in_fine, in_coarse, in_x_coarse, in_fine_mask, in_coarse_mask, label_fine, label_coarse, sample_weight):
        #print(in_coarse.shape, in_coarse_mask.shape)
        gen_out_coarse,_ = self.g_model_coarse(in_coarse, in_coarse_mask)
        #print(gen_out_coarse.shape)
        gen_out_fine = self.g_model_fine(in_fine, in_fine_mask, in_x_coarse)
        #print(gen_out_fine.shape)

        #print(in_fine.shape, gen_out_fine.shape)
        dis_out_1_fake = self.d_model1(in_fine, gen_out_fine)
        #print(in_coarse.shape, gen_out_coarse.shape)
        dis_out_2_fake = self.d_model2(in_coarse, gen_out_coarse)
        #print(dis_out_1_fake.shape)
        #print(dis_out_2_fake.shape)



        fm1 = weighted_feature_matching_loss(dis_out_1_fake, in_fine, label_fine, self.d_model1, self.inner_weight, sample_weight)
        fm2 = weighted_feature_matching_loss(dis_out_2_fake, in_coarse, label_coarse, self.d_model2, self.inner_weight, sample_weight)

        return dis_out_1_fake[0], dis_out_2_fake[0], gen_out_fine,gen_out_coarse, gen_out_coarse, gen_out_fine, gen_out_coarse, gen_out_fine, fm1, fm2


def create_rvgan(g_model_fine, g_model_coarse, d_model1, d_model2, inner_weight):
    model = RVGAN(g_model_fine, g_model_coarse, d_model1, d_model2, inner_weight)
    return model

if __name__ == "__main__":
    # 初始化生成器和判别器
    g_model_coarse = CoarseGenerator()
    g_model_fine = FineGenerator()
    d_model1 = DiscriminatorAE()
    d_model2 = DiscriminatorAE()

    # 创建 RVGAN 模型
    rvgan = create_rvgan(g_model_fine, g_model_coarse, d_model1, d_model2, inner_weight=0.5)

    # 定义优化器
    optimizer = Adam(rvgan.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # 定义损失函数
    criterion = nn.MSELoss()
    hinge_loss = nn.HingeEmbeddingLoss()
    batch_size = 4
    device = torch.device('cpu')
    sample_weight = 1.0  # 这里可以根据实际情况定义sample_weight
    in_size = 128
    # 训练示例
    in_fine = torch.randn((batch_size, 3, in_size, in_size)).to(device)
    in_coarse = torch.randn((batch_size, 3, in_size//2, in_size//2)).to(device)
    in_x_coarse = torch.randn((batch_size, 64, in_size//2, in_size//2)).to(device)
    in_fine_mask = torch.randn((batch_size, 1, in_size, in_size)).to(device)
    in_coarse_mask = torch.randn((batch_size, 1, in_size//2, in_size//2)).to(device)
    label_fine = torch.randn((batch_size, 1, in_size, in_size)).to(device)
    label_coarse = torch.randn((batch_size, 1, in_size//2, in_size//2)).to(device)

    outputs = rvgan(in_fine, in_coarse, in_x_coarse, in_fine_mask, in_coarse_mask, label_fine, label_coarse, sample_weight)

    # 计算损失并更新模型参数
    loss = criterion(outputs[0], label_fine) + criterion(outputs[1], label_coarse) + hinge_loss(outputs[2], label_fine) + hinge_loss(outputs[3], label_coarse) + outputs[8] + outputs[9]
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
