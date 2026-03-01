import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from timm.layers import create_act_layer, get_act_layer, make_divisible
from timm.layers import ConvMlp
from timm.layers import LayerNorm2d

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5
        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        a = x
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        B, N, C = x.shape

        q = self.q_linear(x).reshape(B, N, C)
        k = self.k_linear(x).reshape(B, N, C)
        v = self.v_linear(x).reshape(B, N, C)

        H, W = a.shape[1], a.shape[2]
        central_pixel_index = (H // 2) * W + (W // 2)
        central_pixel = q[:, central_pixel_index, :].unsqueeze(1)
        q = central_pixel.expand(B, N, C)

        pos_indices = torch.arange(N).reshape(H, W)
        center_x, center_y = H // 2, W // 2
        distances = torch.sqrt((pos_indices // W - center_x) ** 2 + (pos_indices % W - center_y) ** 2)
        weights = torch.exp(1 - distances).to(device)

        q = q * weights.view(1, N, 1).expand(B, N, C)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).reshape(B, N, C)
        x = self.proj(x)
        x = x.reshape(a.shape[0], a.shape[1], a.shape[2], a.shape[3])
        return x


class Attention1(nn.Module):
    def __init__(self, dim, max_relative_position=14, num_heads=1, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(dropout)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * max_relative_position - 1), num_heads))
        self.max_relative_position = max_relative_position


    def forward(self, x):

        B, N, C = x.shape
        q = self.q_linear(x).reshape(B, N, C)
        k = self.k_linear(x).reshape(B, N, C)
        v = self.v_linear(x).reshape(B, N, C)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_dropout(out)
        return out


class GatedAttention_spectral(nn.Module):
    def __init__(self, dim, expension_ratio=8/3, num_heads=8, max_relative_position=14,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, drop_path=0.):
        super().__init__()
        self.norm = norm_layer(dim)
        hidden = int(expension_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.act = act_layer()
        self.split_indices = (hidden, hidden)
        self.attn = Attention1(hidden, max_relative_position)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        shortcut = x
        x = x.reshape(shortcut.shape[0], shortcut.shape[3], shortcut.shape[1] * shortcut.shape[2])
        x = self.norm(x)
        g, i = torch.split(self.fc1(x), self.split_indices, dim=-1)
        i = self.attn(i)
        x = self.fc2(self.act(g) * i)
        x = x.reshape(shortcut.shape[0], shortcut.shape[1], shortcut.shape[2], shortcut.shape[3])
        return (x + shortcut).permute(0, 3, 1, 2)

class GlobalContext(nn.Module):
    def __init__(self, channels, use_attn=True, fuse_add=False, fuse_scale=True, init_last_zero=False,
                 rd_ratio=1. / 8, rd_channels=None, rd_divisor=1, act_layer=nn.ReLU, gate_layer='sigmoid'):
        super(GlobalContext, self).__init__()
        act_layer = get_act_layer(act_layer)
        self.conv_attn = nn.Conv2d(channels, 1, kernel_size=1, bias=True) if use_attn else None
        if rd_channels is None:
            rd_channels = make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.)  # 计算 rd_channels
        if fuse_add:
            self.mlp_add = ConvMlp(channels, rd_channels, act_layer=act_layer, norm_layer=LayerNorm2d)
        else:
            self.mlp_add = None
        if fuse_scale:
            self.mlp_scale = ConvMlp(channels, rd_channels, act_layer=act_layer, norm_layer=LayerNorm2d)
        else:
            self.mlp_scale = None
        self.gate = create_act_layer(gate_layer)
        self.init_last_zero = init_last_zero
        self.reset_parameters()
    def reset_parameters(self):
        if self.conv_attn is not None:
            nn.init.kaiming_normal_(self.conv_attn.weight, mode='fan_in', nonlinearity='relu')
        if self.mlp_add is not None:
            nn.init.zeros_(self.mlp_add.fc2.weight)
    def forward(self, x):
        B, C, H, W = x.shape
        if self.conv_attn is not None:
            attn = self.conv_attn(x).reshape(B, 1, H * W)
            attn = F.softmax(attn, dim=-1).unsqueeze(3)
            context = x.reshape(B, C, H * W).unsqueeze(1) @ attn
            context = context.view(B, C, 1, 1)
        else:
            context = x.mean(dim=(2, 3), keepdim=True)
        if self.mlp_scale is not None:
            x = self.mlp_scale(context)
            x = x * self.gate(x)
        if self.mlp_add is not None:
            mlp_x = self.mlp_add(context)
            x = x + mlp_x
        return x

class SpatialExchange(nn.Module):
    def __init__(self, p=1 / 2):
        super().__init__()
        assert p >= 0 and p <= 1
        self.p = int(1 / p)

    def forward(self, x1, x2):
        N, c, h, w = x1.shape
        exchange_mask = torch.arange(w) % self.p == 0

        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[..., ~exchange_mask] = x1[..., ~exchange_mask]
        out_x2[..., ~exchange_mask] = x2[..., ~exchange_mask]
        out_x1[..., exchange_mask] = x2[..., exchange_mask]
        out_x2[..., exchange_mask] = x1[..., exchange_mask]

        return out_x1, out_x2

class ChannelExchange(nn.Module):

    def __init__(self, p=1 / 2):
        super().__init__()
        assert p >= 0 and p <= 1
        self.p = int(1 / p)

    def forward(self, x1, x2):
        N, c, h, w = x1.shape

        exchange_map = torch.arange(c) % self.p == 0
        exchange_mask = exchange_map.unsqueeze(0).expand((N, -1))

        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[~exchange_mask, ...] = x1[~exchange_mask, ...]
        out_x2[~exchange_mask, ...] = x2[~exchange_mask, ...]
        out_x1[exchange_mask, ...] = x2[exchange_mask, ...]
        out_x2[exchange_mask, ...] = x1[exchange_mask, ...]

        return out_x1, out_x2

class GatedAttention_spatial(nn.Module):
    def __init__(self, dim, expension_ratio=8/3, num_heads=8, max_relative_position=14,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, drop_path=0.):
        super().__init__()
        self.norm = norm_layer(dim)
        hidden = int(expension_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.act = act_layer()
        self.split_indices = (hidden, hidden)
        self.attn = Attention(hidden, max_relative_position)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        shortcut = x
        x = self.norm(x)
        g, i = torch.split(self.fc1(x), self.split_indices, dim=-1)
        i = self.attn(i)
        x = self.fc2(self.act(g) * i)
        return (x + shortcut).permute(0, 3, 1, 2)

class GatedModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        hidden_layer = 64
        self.CNN1 = nn.Conv2d(input_dim, hidden_layer, kernel_size=3, stride=1, padding=1)
        self.block1 = GatedAttention_spatial(hidden_layer)
        self.layer1 = GatedAttention_spectral(49)
        self.CNN2 = nn.Conv2d(hidden_layer, hidden_layer*2, kernel_size=3, stride=1, padding=0)
        self.block2 = GatedAttention_spatial(hidden_layer*2)
        self.layer2 = GatedAttention_spectral(25)
        self.CNN3 = nn.Conv2d(hidden_layer*2, hidden_layer*4, kernel_size=3, stride=1, padding=0)
        self.block3 = GatedAttention_spatial(hidden_layer * 4)
        self.layer3 = GatedAttention_spectral(9)
        self.CNN4 = nn.Conv2d(hidden_layer * 4, hidden_layer * 8, kernel_size=3, stride=1, padding=0)
        self.block4 = GatedAttention_spatial(hidden_layer * 8)
        self.layer4 = GatedAttention_spectral(1)
        self.Enhance4 = nn.Sequential(
            nn.Conv2d(hidden_layer * 8, hidden_layer * 8, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(hidden_layer * 8),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Sequential(
            nn.Linear(3072, 128, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2, bias=True),
        )
        self.fuse7 = GlobalContext(hidden_layer * 8)
        self.fuse8 = GlobalContext(hidden_layer * 8)
        self.spatial_exchange = SpatialExchange(p=0.5)
        self.channel_exchange = ChannelExchange(p=0.5)

    def forward(self, x1, x2):
        x1 = self.CNN1(x1)
        x2 = self.CNN1(x2)

        x1,x2 = self.spatial_exchange(x1,x2)
        x1,x2 = self.channel_exchange(x1,x2)
        spe1 = self.block1(x1)
        a1 = spe1
        spe2 = self.block1(x2)
        a2 = spe2
        spa1 = self.layer1(x1)
        b1 = spa1
        spa2 = self.layer1(x2)
        b2 = spa2

        a1 = self.CNN2(a1)
        a2 = self.CNN2(a2)
        b1 = self.CNN2(b1)
        b2 = self.CNN2(b2)
        a1, a2 = self.spatial_exchange(a1, a2)
        b1, b2 = self.channel_exchange(b1, b2)
        spe1 = self.block2(a1)
        a1 = spe1
        spe2 = self.block2(a2)
        a2 = spe2
        spa1 = self.layer2(b1)
        b1 = spa1
        spa2 = self.layer2(b2)
        b2 = spa2

        a1 = self.CNN3(a1)
        a2 = self.CNN3(a2)
        b1 = self.CNN3(b1)
        b2 = self.CNN3(b2)
        a1, a2 = self.spatial_exchange(a1, a2)
        b1, b2 = self.channel_exchange(b1, b2)

        spe1 = self.block3(a1)
        a1 = spe1
        spe2 = self.block3(a2)
        a2 = spe2
        spa1 = self.layer3(b1)
        b1 = spa1
        spa2 = self.layer3(b2)
        b2 = spa2

        a1 = self.CNN4(a1)
        a2 = self.CNN4(a2)
        b1 = self.CNN4(b1)
        b2 = self.CNN4(b2)
        a1, a2 = self.spatial_exchange(a1, a2)
        b1, b2 = self.channel_exchange(b1, b2)

        spe1 = self.block4(a1)
        spe2 = self.block4(a2)
        spa1 = self.layer4(b1)
        spa2 = self.layer4(b2)

        fe1111 = self.fuse7(spe1).squeeze(-1).squeeze(-1)
        fe2222 = self.fuse7(spe2).squeeze(-1).squeeze(-1)
        fa1111 = self.fuse8(spa1).squeeze(-1).squeeze(-1)
        fa2222 = self.fuse8(spa2).squeeze(-1).squeeze(-1)
        y = torch.cat([fe1111 - fe2222, fa1111 - fa2222, fe1111, fe2222, fa1111, fa2222], dim=1)

        X = self.fc(y)
        return X