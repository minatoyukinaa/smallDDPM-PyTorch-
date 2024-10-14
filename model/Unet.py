import math

import torch
import torch.nn.functional as F
from torch import nn
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
class ResnetBlock(nn.Module):
    def __init__(self,in_channels,out_channels,time_embed_dim):
        super().__init__()
        if time_embed_dim is not None:
            self.mlp = nn.Sequential(
                nn.Mish(),
                nn.Linear(time_embed_dim, out_channels)
            )
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Mish()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.Mish()
        )
        self.res_conv = nn.Conv2d(in_channels, out_channels,kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x,time_embed,add):
        h = self.block1(x)
        if time_embed is not None:
            # 在第二个卷积层加入time_embed
            # 不能直接相加，要经过mlp,把time_emdbed 变为
            # [batch,output_channel]的形状,然后再view 变成[batch,output_channel,1,1]
            # 然后进行广播加法
            h += self.mlp(time_embed)[:,:,None,None]
        if add is not None:
            # 如果是Unet右边的模块，可能会传递左边的值过来,需要相加
            h += add
        h = self.block2(h)
        # 如果in_channels != out_channels, 就要加一个kernel_size=1的卷积
        return h + self.res_conv(x)
class Downsample(nn.Module):
    def __init__(self,channels):
        super().__init__()
        # 该卷积会让高宽减半
        self.conv = nn.Conv2d(channels, channels, kernel_size=3,stride=2,padding=1)
    def forward(self,x):
        return self.conv(x)
class Upsample(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels,kernel_size=4, stride=2,padding=1)
    def forward(self,x):
        return self.conv(x)
class Unet(nn.Module):
    # image_channel:图像输入的通道,RGB就是3
    # n_chaneel:初始卷积的通道大小
    # ch_mults: 在Unet中，每一层的通道会逐渐增加，这个数组表示第i层增加的倍数
    # is_attn: 是否使用注意力机制
    # n_blocks: 在连续的同一层中，使用多少个 Residual_Blocks
    def __init__(self,image_channels=3,n_channels=64,ch_multiplier=[1,2,1,2],use_attn=False
                 ):
        super().__init__()
        padding = 1
        self.use_attn = use_attn
        self.image_proj = nn.Conv2d(image_channels,n_channels,kernel_size=3,padding=padding)

        self.time_emb = SinusoidalPosEmb(dim=n_channels)
        self.mlp = nn.Sequential(
            nn.Linear(n_channels, n_channels*4),
            nn.Mish(),
            nn.Linear(n_channels*4,n_channels)
        )
        # 一共有多深
        num_resolutions = len(ch_multiplier)
        # 接下来进入三个模块的编写 Unet = Downs + middle + Up
        # 1. 编写Downs:  down模块 = Residual_block * 2 + Downsample
        self.downs = nn.ModuleList([])
        out_channels = in_channels = n_channels
        for i in range(num_resolutions):
            # 一共有 num_resolutions层down模块，
            # 注意，最后一层down模块没有Downsample模块
            out_channels = in_channels * ch_multiplier[i]
            is_last =i == num_resolutions-1
            self.downs.append(nn.ModuleList([
                ResnetBlock(in_channels=in_channels, out_channels=out_channels, time_embed_dim=n_channels),
                ResnetBlock(in_channels=out_channels, out_channels=out_channels, time_embed_dim=n_channels),
                Downsample(out_channels) if not is_last else nn.Identity()
            ]))
            # 下一层的in_channel = 本层的out_channel
            in_channels = out_channels
        self.mid1 = ResnetBlock(in_channels=in_channels, out_channels=in_channels, time_embed_dim=n_channels)
        self.mid2 = ResnetBlock(in_channels=in_channels, out_channels=in_channels,time_embed_dim=n_channels)
        if use_attn:
            # 占位置填坑
            self.attn = nn.Identity()
        else:
            self.attn = nn.Identity()
        self.ups = nn.ModuleList([])
        out_channels = out_channels
        in_channels = 2 * out_channels
        for i in reversed(range(num_resolutions)):
            is_last = i == 0
            out_channels = out_channels // ch_multiplier[i]
            self.ups.append(nn.ModuleList([
                ResnetBlock(in_channels=in_channels, out_channels=out_channels, time_embed_dim=n_channels),
                ResnetBlock(in_channels=out_channels, out_channels=out_channels, time_embed_dim=n_channels),
                Upsample(out_channels) if not is_last else nn.Identity()
            ]))
            in_channels = 2 * out_channels
            print(ch_multiplier[i])
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,kernel_size=3, padding=1),
            nn.Mish(),
            nn.Conv2d(in_channels=out_channels,out_channels=image_channels,kernel_size=1)
        )
        self.apply_weight_norm()
    def apply_weight_norm(self):
        def _apply_weight_norm(m):
            if isinstance(m,torch.nn.Conv2d):
                nn.utils.weight_norm(m)
        self.apply(_apply_weight_norm)
    def forward(self,x,t):
        # 回忆一下Unet的作用，接受一个加噪图像x,时间t,预测对应的噪声eps
        # 做time_embedding,需要输入一个维度，表示把整数t映射到dim维度的向量
        # 不要忘了，这里的 x,t 格式都是 [batch,data]的形式，前面第一个维度是batch
        t = self.mlp(self.time_emb(t))
        x = self.image_proj(x)

        # downs部分的forward
        h = []
        for i,(res1,res2,downsample) in enumerate(self.downs):
            x = res1(x,t,None)
            x = res2(x,t,None)
            h.append(x)
            x = downsample(x)
        # middle部分的forward
        x = self.mid1(x,t,None)
        if self.use_attn:
            x = self.attn(x)
        x = self.mid2(x,t,None)
#1. 64
#2. 128
#3. 128
#4. 256
        for i,(resnet1,resnet2,upsample) in enumerate(self.ups):
            x1 = h.pop()
            # print(f'Resnet {i}: x1.shape = {x1.shape},x.shape={x.shape}')
            x = torch.cat((x,x1),dim=1)
            x = resnet1(x,t,None)
            x = resnet2(x,t,None)
            x = upsample(x)
        return self.final_conv(x)


