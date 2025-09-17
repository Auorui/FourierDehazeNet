import os
import argparse
import pyzjr
import torch
from torch.utils.data import DataLoader
from pyzjr.nn import AverageMeter, release_gpu_memory
import numpy as np
from pyzjr.visualize.printf import redirect_console
from models.networks import get_dehaze_networks
from utils import dehaze_index, RESIDEDatasetTest

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='FFANet', type=str, help='train net name')
parser.add_argument('--data_dir', default=r'E:\PythonProject\DehazeProject\data\RSHD\thick\test', type=str, help='path to dataset')
parser.add_argument('--resume_training', default=r'E:\PythonProject\DehazeProject\model_pth\RSHD\thick\FFANet.pth', type=str, help='path to models saving')
parser.add_argument('--result_dir', default='./deresults', type=str, help='path to results saving')
parser.add_argument('--input_shape', default=256, type=int, help='target shape')
args = parser.parse_args()

def write_tensor(tensor, filename):
    from PIL import Image
    tensor = tensor.detach().cpu().squeeze(0)
    tensor = tensor.permute(1, 2, 0).numpy()
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    tensor = (tensor * 255).astype(np.uint8)
    image = Image.fromarray(tensor)
    image.save(filename)

def test(test_loader, network, result_dir, only_index=False):
    PSNR = AverageMeter()
    SSIM = AverageMeter()
    MSE = AverageMeter()
    torch.cuda.empty_cache()
    result_dir = os.path.join(result_dir, args.model)
    network.eval()
    time_str = pyzjr.timestr()
    save_img_path = os.path.join(result_dir, f'{time_str}/img')
    os.makedirs(save_img_path, exist_ok=True)
    f_result_path = os.path.join(result_dir, f'{time_str}/results.log')
    redirect_console(f_result_path)
    for idx, batch in enumerate(test_loader):
        input, target, filename = batch[0].cuda(), batch[1].cuda(), batch[2][0]

        with torch.no_grad():
            output = network(input).clamp_(-1, 1)
            psnr_val, ssim_val = dehaze_index(output, target)

        PSNR.update(psnr_val)
        SSIM.update(ssim_val)

        print('Test: [{0}]\t'
              'PSNR: {psnr.val:.05f} ({psnr.avg:.05f})\t'
              'SSIM: {ssim.val:.05f} ({ssim.avg:.05f})\t'
              'filename: {filename}'
              .format(idx + 1, psnr=PSNR, ssim=SSIM, filename=filename))
        # print(f"{filename} {PSNR.val:.05f}")

        if not only_index:
            write_tensor(output, os.path.join(save_img_path, f'{args.model}_{filename}'))
            print(f"第{idx+1}张 保存至{os.path.join(save_img_path, f'{args.model}_{filename}')}")


if __name__ == '__main__':
    network = get_dehaze_networks(args.model)
    cuda = network.cuda()
    network.load_state_dict(torch.load(args.resume_training, map_location=torch.device('cuda:0')))
    test_dataset = RESIDEDatasetTest(args.data_dir, args.input_shape)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             num_workers=2,
                             pin_memory=False)
    test(test_loader, network, result_dir=args.result_dir, only_index=True)