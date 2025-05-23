"""
Modified from ConvNeXt official repo: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
"""

# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from functools import partial


class PartialConv2d(nn.Module):
    r""" 
    Conduct convolution on partial channels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, 
                 conv_ratio=1.0,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, **kwargs,
                 ):
        super().__init__()
        in_chs = int(in_channels * conv_ratio)
        out_chs = int(out_channels * conv_ratio)
        gps = int(groups * conv_ratio) or 1 # groups should be at least 1
        self.conv = nn.Conv2d(in_chs, out_chs, 
                              kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, 
                              groups=gps, bias=bias,
                              **kwargs,
                              )
        self.split_indices = (in_channels - in_chs, in_chs)

    def forward(self, x):
        identity, conv = torch.split(x, self.split_indices, dim=1)
        return torch.cat(
            (identity, self.conv(conv)),
            dim=1,
        )


class MSCABlock(nn.Module):
    r""" ConvNeXt Block with MSCA attention"""
    def __init__(self, dim, kernel_sizes=[3, 11, 11],
                drop_path=0., layer_scale_init_value=1e-6,
                conv_fn=nn.Conv2d,
                ):
        super().__init__()

        # 四分支配置 (3x3, 1x11, 11x1, identity)
        self.conv_branches = nn.ModuleList([
            conv_fn(dim//4, dim//4, kernel_size=(3,3), padding=1, groups=dim//4)
            if i==0 else
            conv_fn(dim//4, dim//4, kernel_size=(1,11), padding=(0,5), groups=dim//4)
            if i==1 else
            conv_fn(dim//4, dim//4, kernel_size=(11,1), padding=(5,0), groups=dim//4)
            for i in range(3)
        ])
        self.identity = nn.Identity()

        # 参数验证
        assert len(kernel_sizes) == 3, "需要3个卷积核配置 [3x3, 1x11, 11x1]"
        assert kernel_sizes[0] == 3 and kernel_sizes[1] == 11 and kernel_sizes[2] == 11, \
            "卷积核尺寸需要为 [3, 11, 11]"

        # 空间注意力机制
        self.attn = nn.Sequential(
            nn.Conv2d(dim, dim//8, 1),
            nn.LayerNorm([dim//8, 1, 1]),
            nn.GELU(),
            nn.Conv2d(dim//8, dim, 1),
            nn.Sigmoid()
        )

        # 通道融合卷积
        self.channel_fusion = nn.Conv2d(dim*2, dim, 1)  # 包含原始输入和特征处理流

        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim))) \
            if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        identity = x
        
        # 分割输入到四个分支
        x_split = torch.chunk(x, 4, dim=1)
        
        # 前三路卷积处理
        conv_outs = [
            branch(x_split[i]) for i, branch in enumerate(self.conv_branches)
        ]
        # 第四路恒等映射
        conv_outs.append(self.identity(x_split[3]))
        
        # 拼接特征
        fused = torch.cat(conv_outs, dim=1)
        
        # 第一次残差连接
        fused = identity + fused
        
        # 空间注意力
        attn_map = self.attn(fused.mean(dim=1, keepdim=True))
        attended = fused * attn_map
        
        # 通道融合
        fused_features = torch.cat([identity, attended], dim=1)
        fused_features = self.channel_fusion(fused_features)
        
        # MLP处理
        x = fused_features.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        
        # 最终残差连接
        return fused_features + self.drop_path(x)


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, kernel_size=7,
                drop_path=0., layer_scale_init_value=1e-6,
                conv_fn=nn.Conv2d,
                ):
        super().__init__()
        self.dwconv = conv_fn(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 kernel_sizes=7, conv_fns=nn.Conv2d,
                 **kwargs,
                 ):
        super().__init__()

        num_stages = len(depths)
        self.num_stages = num_stages

        if not isinstance(kernel_sizes, (list, tuple)):
            kernel_sizes = [kernel_sizes] * num_stages
        if not isinstance(conv_fns, (list, tuple)):
            conv_fns = [conv_fns] * num_stages

        self.num_classes = num_classes
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(self.num_stages - 1):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(self.num_stages):
            stage = nn.Sequential(
                *[MSCABlock(dim=dims[i], drop_path=dp_rates[cur + j],
                kernel_sizes=[3,5,7],
                layer_scale_init_value=layer_scale_init_value,
                conv_fn=conv_fns[i],
                ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(self.num_stages):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.0', 'classifier': 'head.fc',
        **kwargs
    }


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",

    # add by this InceptionNeXt repo
    "convnext_tiny_k5_1k": "https://github.com/sail-sg/inceptionnext/releases/download/model/convnext_tiny_k5_1k_224_ema.pth",
    "convnext_tiny_k3_1k": "https://github.com/sail-sg/inceptionnext/releases/download/model/convnext_tiny_k3_1k_224_ema.pth",
    "convnext_tiny_k3_par1_2_1k": "https://github.com/sail-sg/inceptionnext/releases/download/model/convnext_tiny_k3_par1_2_1k_224_ema.pth",
    "convnext_tiny_k3_par3_8_1k": "https://github.com/sail-sg/inceptionnext/releases/download/model/convnext_tiny_k3_par3_8_1k_224_ema.pth",
    "convnext_tiny_k3_par1_4_1k": "https://github.com/sail-sg/inceptionnext/releases/download/model/convnext_tiny_k3_par1_4_1k_224_ema.pth",
    "convnext_tiny_k3_par1_8_1k": "https://github.com/sail-sg/inceptionnext/releases/download/model/convnext_tiny_k3_par1_8_1k_224_ema.pth",
    "convnext_tiny_k3_par1_16_1k": "https://github.com/sail-sg/inceptionnext/releases/download/model/convnext_tiny_k3_par1_16_1k_224_ema.pth",

}


@register_model
def convnext_tiny(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_tiny_k5(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                     kernel_sizes=5, 
                     **kwargs)
    assert not in_22k, "22k pre-trained model not available"
    model.default_cfg = _cfg()
    if pretrained:
        url = model_urls['convnext_tiny_k5_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model

@register_model
def convnext_tiny_k3(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                     kernel_sizes=3,
                     **kwargs)
    assert not in_22k, "22k pre-trained model not available"
    model.default_cfg = _cfg()
    if pretrained:
        url = model_urls['convnext_tiny_k3_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model

@register_model
def convnext_tiny_k3_par1_2(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                     kernel_sizes=3,
                     conv_fns=partial(PartialConv2d, conv_ratio=0.5),
                     **kwargs)
    assert not in_22k, "22k pre-trained model not available"
    model.default_cfg = _cfg()
    if pretrained:
        url = model_urls['convnext_tiny_k3_par1_2_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model

@register_model
def convnext_tiny_k3_par3_8(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                     kernel_sizes=3,
                     conv_fns=partial(PartialConv2d, conv_ratio=3/8),
                     **kwargs)
    assert not in_22k, "22k pre-trained model not available"
    model.default_cfg = _cfg()
    if pretrained:
        url = model_urls['convnext_tiny_k3_par3_8_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model

@register_model
def convnext_tiny_k3_par1_4(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                     kernel_sizes=3,
                     conv_fns=partial(PartialConv2d, conv_ratio=0.25),
                     **kwargs)
    assert not in_22k, "22k pre-trained model not available"
    model.default_cfg = _cfg()
    if pretrained:
        url = model_urls['convnext_tiny_k3_par1_4_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model

@register_model
def convnext_tiny_k3_par1_8(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                     kernel_sizes=3,
                     conv_fns=partial(PartialConv2d, conv_ratio=0.125),
                     **kwargs)
    assert not in_22k, "22k pre-trained model not available"
    model.default_cfg = _cfg()
    if pretrained:
        url = model_urls['convnext_tiny_k3_par1_8_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model

@register_model
def convnext_tiny_k3_par1_16(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                     kernel_sizes=3,
                     conv_fns=partial(PartialConv2d, conv_ratio=1/16),
                     **kwargs)
    assert not in_22k, "22k pre-trained model not available"
    model.default_cfg = _cfg()
    if pretrained:
        url = model_urls['convnext_tiny_k3_par1_16_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model

@register_model
def convnext_small(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_base(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_large(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_xlarge(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        assert in_22k, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
        url = model_urls['convnext_xlarge_22k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model