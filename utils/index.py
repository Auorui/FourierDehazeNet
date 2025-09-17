import torch
import torch.nn.functional as F
from utils.pytorch_ssim2 import ssim

def dehaze_index(output, target, ssim_index=True):
    output = output * 0.5 + 0.5
    target = target * 0.5 + 0.5
    mse = F.mse_loss(output, target)
    psnr = 10 * torch.log10(1 / mse).item()
    if ssim_index:
        _, _, H, W = output.size()
        down_ratio = max(1, round(min(H, W) / 256))
        ssim_val = ssim(F.adaptive_avg_pool2d(output, (int(H / down_ratio), int(W / down_ratio))),
                        F.adaptive_avg_pool2d(target, (int(H / down_ratio), int(W / down_ratio))),
                        data_range=1, size_average=False).item()
        return psnr, ssim_val
    else:
        return psnr