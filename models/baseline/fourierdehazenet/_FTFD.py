"""
The principle of FTFDV2 is consistent with the principle of FTFD in FourierDehzeNet,as the paper
had already been completed at that time, so this part of the work was not further studied.

The image is transformed from the spatial domain to the frequency domain using Fourier Transform,
and then the frequency domain signal is decomposed using various selectable filters, including
Butterworth, ideal, exponential, and Gaussian low-pass filters. Finally, the image is separated
into three components: low-frequency, mid-frequency, and high-frequency.

Key features
    1. Multi-filter support: Provides four different types of filters, each with distinct frequency response characteristics.
    2. Learnable parameters: By introducing a learnable alpha parameter, adaptive mixing of low-frequency signals was achieved.
    3. Band decomposition: accurately decomposes image information into three frequency bands for subsequent targeted processing.

If you want to research this area, consider how to perform feature extraction after obtaining
low, medium, and high-frequency components, how to better integrate them, and how to handle
attention mechanisms. This is also the idea of our paper on the specialized processing of different
frequency components.
"""
from PIL import Image, ImageEnhance
import torch
import torch.nn as nn
import numpy as np


# Fourier Transform Feature Decomposition
class FTFDV2(nn.Module):
    def __init__(self, low_range=0.25, alpha=0.5):
        super(FTFDV2, self).__init__()
        self.low_range = low_range
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def butterworth_lowpass(self, shape, cutoff, device, n=2):
        """Butterworth low-pass filter (smoother transition)"""
        rows, cols = shape
        center_row, center_col = rows // 2, cols // 2
        y, x = torch.meshgrid(
            torch.arange(rows, device=device),
            torch.arange(cols, device=device),
            indexing='ij'
        )
        distance = torch.sqrt((y - center_row) ** 2 + (x - center_col) ** 2)
        return 1 / (1 + (distance / cutoff) ** (2 * n))

    def ideal_lowpass(self, shape, cutoff, device):
        """Ideal low-pass filter"""
        rows, cols = shape
        center_row, center_col = rows // 2, cols // 2
        y, x = torch.meshgrid(
            torch.arange(rows, device=device),
            torch.arange(cols, device=device),
            indexing='ij'
        )
        distance = torch.sqrt((y - center_row) ** 2 + (x - center_col) ** 2)
        return (distance <= cutoff).float()

    def exponential_lowpass(self, shape, cutoff, device, gamma=1.0):
        """Low-pass filter with adjustable attenuation rate"""
        rows, cols = shape
        center_row, center_col = rows // 2, cols // 2
        y, x = torch.meshgrid(
            torch.arange(rows, device=device),
            torch.arange(cols, device=device),
            indexing='ij'
        )
        distance = torch.sqrt((y - center_row) ** 2 + (x - center_col) ** 2)
        return torch.exp(-(distance / cutoff) ** gamma)

    def gaussian_lowpass(self, shape, cutoff, device):
        """Gaussian low-pass filter, smoother transition"""
        rows, cols = shape
        center_row, center_col = rows // 2, cols // 2
        y, x = torch.meshgrid(
            torch.arange(rows, device=device),
            torch.arange(cols, device=device),
            indexing='ij'
        )
        distance = torch.sqrt((y - center_row) ** 2 + (x - center_col) ** 2)
        weights = torch.exp(-(distance ** 2) / (2 * (cutoff ** 2)))
        return weights.unsqueeze(0).unsqueeze(0)

    def FFBD(self, target):
        device = target.device
        B, C, H, W = target.shape
        max_radius = np.sqrt((H // 2) ** 2 + (W // 2) ** 2)

        # can be replaced by any of the above filters
        low_cutoff = max_radius * self.low_range
        low_pass = self.exponential_lowpass((H, W), low_cutoff, device)

        # Bandpass filtering retains edges.
        high_cutoff = max_radius * (1 - self.low_range)
        band_pass = 1 - self.exponential_lowpass((H, W), high_cutoff, device)

        fft_tensor = torch.fft.fftshift(torch.fft.fft2(target, dim=(-2, -1)), dim=(-2, -1))

        # Weighted mix of the original signal and the low-frequency signal.
        low_freq = fft_tensor * low_pass
        blended_low = self.alpha * low_freq + (1 - self.alpha) * fft_tensor

        high_freq = fft_tensor * (1 - low_pass - band_pass)
        mid_freq = fft_tensor * band_pass

        return blended_low, mid_freq, high_freq

    def _ifft(self, fft_freq):
        freq_tensor = torch.fft.ifft2(torch.fft.ifftshift(fft_freq, dim=(-2, -1)), dim=(-2, -1)).real
        return freq_tensor

    def forward(self, x):
        low, mid, high = self.FFBD(x)
        return self._ifft(low), self._ifft(mid), self._ifft(high)


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


def write_tensor(tensor, filename):
    tensor = tensor.squeeze(0)
    tensor = tensor.permute(1, 2, 0).detach().cpu().numpy()
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    tensor = (tensor * 255).astype(np.uint8)
    image = Image.fromarray(tensor)
    image.save(filename)

def overlay_on_original(original, overlay, alpha=0.5):
    overlay = overlay.resize(original.size)
    overlay = ImageEnhance.Contrast(overlay).enhance(1.5)
    overlay = ImageEnhance.Brightness(overlay).enhance(1.2)
    return Image.blend(original, overlay, alpha)

if __name__ == "__main__":
    import pyzjr
    import matplotlib.pyplot as plt
    pyzjr.matplotlib_patch()
    file = r"airplane_58.png"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input = pyzjr.read_tensor(file, (512, 512)).to(device)

    block = FTFD(0.24)
    low_freq_tensor, mid_freq_tensor, high_freq_tensor = block(input)
    print("low_freq_tensor.shape:", low_freq_tensor.shape)
    print("mid_freq_tensor.shape:", mid_freq_tensor.shape)
    print("high_freq_tensor.shape:", high_freq_tensor.shape)
    # 保存高低频图像
    write_tensor(low_freq_tensor, "low_freq_image.png")
    write_tensor(mid_freq_tensor, "mid_freq_image.png")
    write_tensor(high_freq_tensor, "high_freq_image.png")

    # 加载原图
    original_image = Image.open(file).convert("RGB")  # 确保原图为 RGB 格式

    # 加载高低频图像
    low_freq_image = Image.open("low_freq_image.png").convert("RGB")
    mid_freq_image = Image.open("mid_freq_image.png").convert("RGB")
    high_freq_image = Image.open("high_freq_image.png").convert("RGB")

    # 叠加高低频图像到原图
    alpha = 0.5  # 叠加透明度
    low_overlay = overlay_on_original(original_image, low_freq_image, alpha)
    mid_overlay = overlay_on_original(original_image, mid_freq_image, alpha)
    high_overlay = overlay_on_original(original_image, high_freq_image, alpha)

    low_overlay.save("low_overlay.png")
    mid_overlay.save("mid_overlay.png")
    high_overlay.save("high_overlay.png")

    plt.figure()  # 设置画布大小

    # 显示原图
    plt.subplot(1, 4, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis("off")

    # 显示低频叠加
    plt.subplot(1, 4, 2)
    plt.imshow(low_overlay)
    plt.title("Low Frequency Overlay")
    plt.axis("off")

    # 显示中频叠加
    plt.subplot(1, 4, 3)
    plt.imshow(mid_overlay)
    plt.title("Mid Frequency Overlay")
    plt.axis("off")

    # 显示高频叠加
    plt.subplot(1, 4, 4)
    plt.imshow(high_overlay)
    plt.title("High Frequency Overlay")
    plt.axis("off")

    # 保存图像
    plt.tight_layout()  # 调整子图间距
    plt.savefig("overlay_2x2.png")  # 保存为文件
    plt.show()