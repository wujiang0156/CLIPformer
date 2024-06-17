import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from mmengine.model import BaseModule
from timm.models.layers import DropPath

from mmseg.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead


def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp* W_sp, C)
    return img_perm

def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img

class Mlp(BaseModule):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LePEAttention(BaseModule):
    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        """Not supported now, since we have cls_tokens now.....
        """
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.idx = idx
        if idx == -1:
            H_sp, W_sp = self.resolution, self.resolution
        elif idx == 0:
            H_sp, W_sp = self.resolution, self.split_size
        elif idx == 1:
            W_sp, H_sp = self.resolution, self.split_size
        else:
            print ("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp

        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim)        
        self.attn_drop = nn.Dropout(attn_drop)

    def im2cswin(self, x):
        B, C, H, W = x.shape
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp* self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def get_rpe(self, x, func):
        B, C, H, W = x.shape
        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp) ### B', C, H', W'

        rpe = func(x) ### B', C, H', W'
        rpe = rpe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()

        x = x.reshape(-1, self.num_heads, C // self.num_heads, self.H_sp* self.W_sp).permute(0, 1, 3, 2).contiguous()
        return x, rpe

    def forward(self, temp):
        B, _, C, H, W = temp.shape
        idx = self.idx
        if idx == -1:
            H_sp, W_sp = H, W
        elif idx == 0:
            H_sp, W_sp = H, self.split_size
        elif idx == 1:
            H_sp, W_sp = self.split_size, W
        else:
            print ("ERROR MODE in forward", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp

        ### padding for split window
        H_pad = (self.H_sp - H % self.H_sp) % self.H_sp
        W_pad = (self.W_sp - W % self.W_sp) % self.W_sp
        top_pad = H_pad//2
        down_pad = H_pad - top_pad
        left_pad = W_pad//2
        right_pad = W_pad - left_pad
        H_ = H + H_pad
        W_ = W + W_pad

        qkv = F.pad(temp, (left_pad, right_pad, top_pad, down_pad)) ### B,3,C,H',W'
        qkv = qkv.permute(1, 0, 2, 3, 4)  ### 3,B,C,H',W'
        q,k,v = qkv[0], qkv[1], qkv[2]
        
        q = self.im2cswin(q) # B head N C
        k = self.im2cswin(k) # B head N C
        v, rpe = self.get_rpe(v, self.get_v)

        ### Local attention
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N        
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v) + rpe
        # x = (attn @ v)
        x = x.transpose(1, 2).reshape(-1, self.H_sp* self.W_sp, C)  # B head N N @ B head N C

        ### Window2Img
        x = windows2img(x, self.H_sp, self.W_sp, H_, W_) # B H_ W_ C
        x = x[:, top_pad:H+top_pad, left_pad:W+left_pad, :]
        x = x.reshape(B, -1, C)

        return x

class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(num_channels=out_ch, num_groups=32),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class LowRankBilinearAttention(nn.Module):
    def __init__(self, d_model=128, d_k=64, d_v=256):
        super(LowRankBilinearAttention, self).__init__()
        self.WQ = nn.Linear(d_model, d_k)
        self.WK = nn.Linear(d_model, d_k)
        self.WV = nn.Linear(d_model, d_v)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K):
        V = K.clone()
        q = self.WQ(Q)
        k = self.WK(K)
        v = self.WV(V)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        attn_probs = self.softmax(attn_scores)
        output = torch.matmul(attn_probs, v)
        return output

class CrossShapedGLAttention(BaseModule):
    def __init__(self, dim, patches_resolution, num_heads, split_size=7, 
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., 
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm, norm_cfg=dict(type='LN')):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = patches_resolution
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.norm1 = norm_layer(dim)        
        self.projattn = nn.Linear(dim, dim)
        # self.proj_drop = nn.Dropout(drop)
                
        self.attns = nn.ModuleList([
            LePEAttention(
                dim//2, resolution=self.patches_resolution, idx = i,
                split_size=split_size, num_heads=num_heads//2, dim_out=dim//2,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop)
            for i in range(2)])
            
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                       out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

        atten_mask_matrix = None
        self.register_buffer("atten_mask_matrix", atten_mask_matrix)
        self.H = None
        self.W = None
                   
        self.proj = nn.Linear(256, 128)
        self.proj_drop = nn.Dropout(drop)
        self.low_rank_bilinear = LowRankBilinearAttention()

    def forward(self, x, text):
        x_text = self.low_rank_bilinear(x, text)
        x = self.proj_drop(self.proj(x_text))+x

        B, L, C = x.shape
        H = self.H
        W = self.W
        assert L == H * W, "flatten img_tokens has wrong size"
        
        img = self.norm1(x)  ###X : BLC

        temp = self.qkv(img).reshape(B, H, W, 3, C).permute(0, 3, 4, 1, 2) #B,3,C,H,W        
        if True:
            x1 = self.attns[0](temp[:,:,:C//2,:,:]) #0~k-1
            x2 = self.attns[1](temp[:,:,C//2:,:,:]) #k-1~C
            attened_x = torch.cat([x1,x2], dim=2)

        attened_x = attened_x + img # BLC
        attened_x = self.projattn(attened_x)
        x = x + self.drop_path(attened_x)

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class CSwinDecoder(BaseModule):    
    def __init__(self, dim, patches_resolution, num_heads, split_size=7, 
                 embed_dim=256, depth=1, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                 drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
                 drop_path_rate=0.1):
       
        super().__init__()

        self.depth = depth    
       
        norm_cfg = dict(type='BN', requires_grad=True)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, np.sum(3))]  # stochastic depth decay rule

        self.blk = CrossShapedGLAttention(dim=dim, patches_resolution=patches_resolution, 
                num_heads=num_heads, split_size=split_size, mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                act_layer=act_layer, norm_layer=norm_layer, norm_cfg=norm_cfg, drop_path=dpr[2])
        
        self.up = up_conv(embed_dim, embed_dim // 2)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(embed_dim // 2, embed_dim // 2, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

    
    def forward(self, x, text):      
        B, C, H, W = x.shape
        x = self.up(x)  # B , C//2, 2H, 2W
        x = x.reshape(B, C // 2, H * W * 4)   ## BCL
        x = x.permute(0, 2, 1)  ## BLC

        self.blk.H = H * 2   ## *2
        self.blk.W = W * 2   ## *2 
        x = self.blk(x, text) ##BLC
        
        x = x.permute(0, 2, 1) #BCL
        x = x.reshape(B, C // 2, H * 2, W * 2)   # B, C//4 2H, 2W
       
        x = self.conv_relu(x)

        return x


class CSwin_Decoder(BaseModule):
    def __init__(self, dim, patches_resolution, channels,
                 num_heads=4, split_size=7, embed_dim=256, depth=1):
        super().__init__()

        self.up = CSwinDecoder(dim, patches_resolution, num_heads, split_size, embed_dim, depth)
        
        self.conv_relu = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

    def forward(self, x1, x2, text):
        x1 = self.up(x1, text)
        x2 = self.conv2(x2)
        x1 = torch.cat((x2, x1), dim=1)
        out = self.conv_relu(x1)

        return out


@MODELS.register_module()
class CSwinTextDecode(BaseDecodeHead):
    def __init__(self, dim, patches_resolution, num_heads=8, split_size=7, 
                 embed_dim=256, text_dim=512, depth=1, **kwargs): 
        super().__init__(**kwargs)
        
        num_inputs = len(self.in_channels)
        assert num_inputs == len(self.in_index) 
        
        self.layer1 = CSwin_Decoder(dim=128, patches_resolution=patches_resolution,
                 channels=self.channels, num_heads=2, split_size=1, embed_dim=embed_dim, depth=depth)
        self.layer2 = CSwin_Decoder(dim=128, patches_resolution=patches_resolution,
                 channels=self.channels, num_heads=num_heads, split_size=split_size, embed_dim=embed_dim, depth=depth)
        self.layer3 = CSwin_Decoder(dim=128, patches_resolution=patches_resolution,
                 channels=self.channels, num_heads=num_heads, split_size=split_size, embed_dim=embed_dim, depth=depth)
        self.layer4 = CSwin_Decoder(dim=128, patches_resolution=patches_resolution*2,
                 channels=self.channels, num_heads=num_heads, split_size=split_size, embed_dim=embed_dim, depth=depth)
        
        self.text_dim = text_dim
        self.linearA = nn.Sequential(nn.Linear(self.text_dim, 128), nn.LayerNorm(128))
        self.linearB = nn.Sequential(nn.Linear(self.text_dim, 128), nn.LayerNorm(128))

    def forward(self, inputs, textA, textB):
        e1, e2, e3, e4, e5 = inputs
        text = torch.cat([self.linearA(textA), self.linearB(textB)], dim=1)
        d1 = self.layer1(e4, e3, text)
        d2 = self.layer2(d1, e2, text)
        d3 = self.layer3(d2, e1, text)
        d4 = self.layer4(d3, e5, text)

        output = self.cls_seg(d4)
        return output