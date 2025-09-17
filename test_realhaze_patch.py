import argparse
import torch
import os
from torch.utils.data import DataLoader
from pyzjr import AverageMeter, release_gpu_memory, timestr
import numpy as np
from models.networks import get_dehaze_networks
from utils import dehaze_index

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='MixDehazeNet_s', type=str, help='train net name')
parser.add_argument('--data_dir', default=r'E:\PythonProject\DehazeProject\data\RealHaze115', type=str, help='path to dataset')
parser.add_argument('--resume_training', default=r'E:\PythonProject\DehazeProject\model_pth\real_haze\MixDehazeNet_s.pth', type=str, help='path to models saving')
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


def test_real_haze_115_index(val_loader, model, device=torch.device('cuda:0')):
    model.eval()
    total_PSNR = AverageMeter()
    total_SSIM = AverageMeter()
    # 每5个图像属于一个子数据集
    dataset_names = ['O-Haze', 'NH-Haze', 'NH-Haze2']
    dataset_metrics = {
        name: {
            'psnr_sum': 0.0,
            'ssim_sum': 0.0,
            'count': 0
        } for name in dataset_names
    }
    for k, (hazy, gt, patch_nums, filename, _) in enumerate(val_loader):
        hazy_patches = hazy[0]
        gt_patches = gt[0]
        patch_num = patch_nums[0]
        name = filename[0]
        print(f"[{k + 1}]Test image name: {name}, 总切片数为: {patch_num}")
        psnr_sum = 0.0
        ssim_sum = 0.0
        # 确定当前图像属于哪个子数据集
        dataset_index = k // 5
        if dataset_index < len(dataset_names):
            dataset_name = dataset_names[dataset_index]
        else:
            dataset_name = 'Unknown'
        for i, (h, g) in enumerate(zip(hazy_patches, gt_patches)):
            h, g = h.to(device), g.to(device)
            outputs = model(h).clamp_(-1, 1)
            psnr, ssim_val = dehaze_index(outputs, g)
            psnr_sum += psnr
            ssim_sum += ssim_val
        avg_psnr = psnr_sum / patch_num
        avg_ssim = ssim_sum / patch_num
        total_PSNR.update(avg_psnr)
        total_SSIM.update(avg_ssim)
        print(f"    ==> 图像: {name} 平均 PSNR: {total_PSNR.val:.2f}({total_PSNR.avg:.4f}), 平均 SSIM: {total_SSIM.val:.3f}({total_SSIM.avg:.4f}), 数据集: {dataset_name}")
        if dataset_name in dataset_metrics:
            dataset_metrics[dataset_name]['psnr_sum'] += avg_psnr
            dataset_metrics[dataset_name]['ssim_sum'] += avg_ssim
            dataset_metrics[dataset_name]['count'] += 1

    print("\n==== 各数据集的平均指标 ====")
    for dataset_name in dataset_names:
        metrics = dataset_metrics[dataset_name]
        if metrics['count'] > 0:
            avg_psnr = metrics['psnr_sum'] / metrics['count']
            avg_ssim = metrics['ssim_sum'] / metrics['count']
            print(f"[{dataset_name}] 平均 PSNR: {avg_psnr:.4f}, 平均 SSIM: {avg_ssim:.4f}")
        else:
            print(f"[{dataset_name}] 无图像数据")
    print(f"[RealHaze115] 平均 PSNR: {total_PSNR.avg:.4f}, 平均 SSIM: {total_SSIM.avg:.4f}")


def test_real_haze_115_image(val_loader, model, device=torch.device('cuda:0'), patch_size=256):
    model.eval()
    output_dir = f"{args.result_dir}/{timestr()}"
    os.makedirs(output_dir, exist_ok=True)
    for k, (hazy, gt, patch_nums, filename, original_shape) in enumerate(val_loader):
        hazy_patches = hazy[0]
        gt_patches = gt[0]
        patch_num = patch_nums[0]
        name = filename[0]
        original_high, original_wide = original_shape[0]
        print(f"[{k + 1}]Test image name: {name}, 总切片数为: {patch_num}")
        patch_outputs = []
        for i, (h, g) in enumerate(zip(hazy_patches, gt_patches)):
            h = h.to(device)
            with torch.no_grad():
                out = model(h).clamp_(-1, 1)  # out: [1, C, patch_size, patch_size]
            patch_outputs.append(out.cpu())  # 放回 CPU，方便拼接

        # 拼接成整图
        patch_rows = original_high // patch_size
        patch_cols = original_wide // patch_size
        C = patch_outputs[0].shape[1]
        rows = []

        for i in range(patch_rows):
            row_patches = patch_outputs[i * patch_cols:(i + 1) * patch_cols]
            row_cat = torch.cat([p[0] for p in row_patches], dim=2)  # concat along width
            rows.append(row_cat)

        full_image = torch.cat(rows, dim=1)  # concat along height → [C, H, W]
        # 去掉 padding，还原为原始大小
        full_image = full_image[:, :original_high, :original_wide]
        save_path = f"{output_dir}/restored_{name}"
        print(save_path)
        write_tensor(full_image.unsqueeze(0), save_path)

if __name__ == '__main__':
    # O-Haze, NH-Haze, NH-Haze2
    from utils import RealHaze115Dataset, val_collate_fn
    network = get_dehaze_networks(args.model)
    network = network.cuda()
    network.load_state_dict(torch.load(args.resume_training, map_location=torch.device('cuda:0')))
    test_dataset=RealHaze115Dataset(
        root_dir=args.data_dir,
        target_shape=args.input_shape,
        is_train=False,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=1,
                             pin_memory=False, collate_fn=val_collate_fn)
    # test_real_haze_115_index(test_loader, network)
    test_real_haze_115_image(test_loader, network)