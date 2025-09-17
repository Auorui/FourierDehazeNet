import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SKFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(SKFusion, self).__init__()

        self.height = height
        d = max(int(dim / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape
        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats * attn, dim=1)
        return out

# Fourier Transform Feature Decomposition
class FTFD(nn.Module):
    def __init__(self, low_range=.2):
        super(FTFD, self).__init__()
        self.low_range = low_range

    def frequency_filter(self, low_dim, high_dim, shape, device, mode='band'):
        rows, cols = shape
        y, x = torch.meshgrid(torch.arange(rows, device=device), torch.arange(cols, device=device), indexing='ij')
        distance = torch.sqrt((y - rows // 2) ** 2 + (x - cols // 2) ** 2)
        mask = torch.zeros((rows, cols), dtype=torch.float32, device=device)
        if mode == 'low':
            mask[distance <= low_dim] = 1
        elif mode == 'high':
            mask[distance >= high_dim] = 1
        elif mode == 'band':
            mask[(distance > low_dim) & (distance < high_dim)] = 1
        return mask.unsqueeze(0).unsqueeze(0)

    def FFBD(self, target, low_range=.2):
        # Frequency filtering band decomposition
        device = target.device
        B, C, H, W = target.shape
        max_radius = np.sqrt((H // 2) ** 2 + (W // 2) ** 2)
        low_dim = max_radius * low_range
        high_dim = max_radius * (1 - low_range)
        fft_tensor = torch.fft.fftshift(torch.fft.fft2(target, dim=(-2, -1)), dim=(-2, -1))
        low_pass_filter = self.frequency_filter(low_dim, None, (H, W), mode='low', device=device)
        high_pass_filter = self.frequency_filter(None, high_dim, (H, W), mode='high', device=device)
        mid_pass_filter = self.frequency_filter(low_dim, high_dim, (H, W), mode='band', device=device)
        low_freq_fft = fft_tensor * low_pass_filter
        high_freq_fft = fft_tensor * high_pass_filter
        mid_freq_fft = fft_tensor * mid_pass_filter
        return low_freq_fft, mid_freq_fft, high_freq_fft

    def _ifft(self, fft_freq):
        freq_tensor = torch.fft.ifft2(torch.fft.ifftshift(fft_freq, dim=(-2, -1)), dim=(-2, -1)).real
        return freq_tensor

    def forward(self, x):
        low_freq_fft, mid_freq_fft, high_freq_fft = self.FFBD(x, self.low_range)
        low_freq_tensor = self._ifft(low_freq_fft)
        high_freq_tensor = self._ifft(high_freq_fft)
        mid_freq_tensor = self._ifft(mid_freq_fft)
        return low_freq_tensor, mid_freq_tensor, high_freq_tensor

class FrequencyAttention(nn.Module):
    def __init__(self):
        super(FrequencyAttention, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, freq_feat, x, mode='mid'):
        proj_feat = F.relu(freq_feat)
        proj_x = F.relu(x)
        # cosine similarity matrix
        similarity = F.cosine_similarity(proj_feat, proj_x, dim=1)
        attn_weight = self.sigmoid(similarity).unsqueeze(1)
        if mode in ['low', 'L']:
            return x * (1 - attn_weight)
        elif mode in ['high', 'H']:
            return x * (1 - attn_weight)
        else:
            return x * attn_weight

# Frequency multi-scale fusion block
class FMSFB(nn.Module):
    def __init__(self, dim, low_range=.2, kd_cfg=None):
        super(FMSFB, self).__init__()
        self.ftfd = FTFD(low_range)
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        # kernel_size + (kernel_size - 1)(dilation - 1)
        self.low_conv, self.mid_conv, self.high_conv = [
            nn.Conv2d(
                dim, dim,
                kernel_size=k,
                padding=(k * d - d) // 2,
                groups=dim,
                dilation=d,
                padding_mode='reflect'
            ) for (k, d) in kd_cfg
        ]
        # Channel Attention
        self.low_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        # Space Attention
        self.mid_attn = nn.Sequential(
            nn.Conv2d(dim, dim, 5, padding=2, padding_mode='reflect'),
            nn.Sigmoid()
        )
        # Pixel Attention
        self.high_attn = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(dim // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.mlp = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 4, 1),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, 1)
        )
        self.mlp2 = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 4, 1),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, 1)
        )
        self.fattn = FrequencyAttention()

    def forward(self, x):
        identity = x
        x = self.norm1(x)
        lf, mf, hf = self.ftfd(x)
        low_feat = self.low_conv(lf)
        mid_feat = self.mid_conv(mf)
        high_feat = self.high_conv(hf)
        # band fusion
        freq_band = torch.cat([low_feat, mid_feat, high_feat], dim=1)
        freq_band = self.mlp(freq_band)
        x = identity + freq_band

        identity = x
        x = self.norm2(x)
        """
            dynamic frequency extraction
                        ||
                        ||
                        VV
            in-band independent convolution
        Low-freq, mid-freq, and high-freq scale information fusion
                        ||
                        ||
                        VV
             hybrid attention mechanism
        Low-freq:  channel attention × frequency attention → dual suppression of fog concentration
        Mid-freq:  spatial attention × frequency attention → structural enhancement
        High-freq: pixel attention × frequency attention → detail retention
        """
        low_feat2 = self.fattn(lf, x, mode='low') * self.low_attn(x)
        mid_feat2 = self.fattn(mf, x, mode='band') * self.mid_attn(x)
        high_feat2 = self.fattn(hf, x, mode='high') * self.high_attn(x)
        freq_band2 = torch.cat([low_feat2, mid_feat2, high_feat2], dim=1)
        freq_band2 = self.mlp2(freq_band2)
        x = identity + freq_band2
        return x

class BasicLayer(nn.Module):
    """堆叠 L 次 Transformer Basic Block"""
    def __init__(self, dim, depth, kd_cfg=((7, 3), (5, 3), (3, 3))):
        super(BasicLayer, self).__init__()
        max_low_range = 0.1 if depth <= 4 else 0.12
        self.layers = nn.ModuleList([
            FMSFB(
                dim=dim,
                low_range=max(0.24 - (max_low_range / depth + 1e-4) * i, 0.12),
                kd_cfg=kd_cfg
            )
            for i in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_channels=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = patch_size

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect')

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_channels=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_channels = out_channels
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_channels * patch_size ** 2, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class FourierDehazeNet(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=4,
                 embed_dims=(24, 48, 96, 48, 24),
                 depths=(2, 2, 4, 2, 2)
            ):
        super(FourierDehazeNet, self).__init__()
        self.patch_size = 4
        self.patch_embed = PatchEmbed(
            patch_size=1, in_channels=in_channels, embed_dim=embed_dims[0], kernel_size=3)

        self.layer1 = BasicLayer(dim=embed_dims[0], depth=depths[0], kd_cfg=((7, 3), (5, 3), (3, 3)))

        self.patch_merge1 = PatchEmbed(
            patch_size=2, in_channels=embed_dims[0], embed_dim=embed_dims[1], kernel_size=3)

        self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)

        self.layer2 = BasicLayer(dim=embed_dims[1], depth=depths[1], kd_cfg=((7, 3), (5, 3), (3, 3)))

        self.patch_merge2 = PatchEmbed(
            patch_size=2, in_channels=embed_dims[1], embed_dim=embed_dims[2], kernel_size=3)

        self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)

        self.layer3 = BasicLayer(dim=embed_dims[2], depth=depths[2], kd_cfg=((7, 3), (5, 3), (3, 3)))

        self.patch_split1 = PatchUnEmbed(
            patch_size=2, out_channels=embed_dims[3], embed_dim=embed_dims[2])

        assert embed_dims[1] == embed_dims[3]
        self.fusion1 = SKFusion(embed_dims[3])

        self.layer4 = BasicLayer(dim=embed_dims[3], depth=depths[3], kd_cfg=((7, 3), (5, 3), (3, 3)))

        self.patch_split2 = PatchUnEmbed(
            patch_size=2, out_channels=embed_dims[4], embed_dim=embed_dims[3])

        assert embed_dims[0] == embed_dims[4]
        self.fusion2 = SKFusion(embed_dims[4])

        self.layer5 = BasicLayer(dim=embed_dims[4], depth=depths[4], kd_cfg=((7, 3), (5, 3), (3, 3)))

        self.patch_unembed = PatchUnEmbed(
            patch_size=1, out_channels=out_channels, embed_dim=embed_dims[4], kernel_size=3)

    def check_image_size(self, x):
        # NOTE: for I2I test
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.layer1(x)
        skip1 = x

        x = self.patch_merge1(x)
        x = self.layer2(x)
        skip2 = x

        x = self.patch_merge2(x)
        x = self.layer3(x)
        x = self.patch_split1(x)

        x = self.fusion1([x, self.skip2(skip2)]) + x
        x = self.layer4(x)
        x = self.patch_split2(x)

        x = self.fusion2([x, self.skip1(skip1)]) + x
        x = self.layer5(x)
        x = self.patch_unembed(x)
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        feat = self.forward_features(x)
        K, B = torch.split(feat, (1, 3), dim=1)
        x = K * x - B + x
        x = x[:, :, :H, :W]
        return x


if __name__ == "__main__":
    from pyzjr.nn import summary_1, model_complexity_info, summary_2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FourierDehazeNet()
    model = model.to(device)
    inputs = torch.randn(1, 3, 256, 256).to(device)
    output = model(inputs)

    # summary_1(model, input_size=(3, 256, 256))
    summary_2(model, input_size=(3, 256, 256))
    model_complexity_info(model, input_size=(3, 256, 256))
    """
    # MACs: 27.559 GMac
    # FLOPs: 55.118 GFLOPs
    # Params: 3.011 M
    """

