from pyzjr import BaseDataset, SearchFileName, to_2tuple
import pyzjr
import os
import cv2
import torch
import time
import random
import numpy as np

class RealHaze115Dataset(BaseDataset):
    def __init__(
        self,
        root_dir,
        target_shape,
        is_train=True,
        repeat=1,
    ):
        super(RealHaze115Dataset, self).__init__()
        self.mode = is_train
        self.target_shape = to_2tuple(target_shape)
        self.repeat = repeat

        data_dir = os.path.join(root_dir, 'train' if is_train else 'test')
        self.gt = os.path.join(data_dir, 'GT')
        self.hazy = os.path.join(data_dir, 'hazy')

        self.image_name_list = SearchFileName(self.gt, '.png')

    def __len__(self):
        if self.mode:
            return len(self.image_name_list) * self.repeat
        else:
            return len(self.image_name_list)

    def crop_nonoverlap(self, img):
        h, w, _ = img.shape
        ph, pw = self.target_shape
        patches = []
        for i in range(0, h, ph):
            for j in range(0, w, pw):
                patch = img[i:i+ph, j:j+pw, :]
                if patch.shape[0] == ph and patch.shape[1] == pw:  # 避免不完整patch
                    patches.append(self.hwc2chw(patch))
        return patches

    def __getitem__(self, item):
        self.disable_cv2_multithreading()
        img_idx = item % len(self.image_name_list)
        img_name = self.image_name_list[img_idx]
        source_img = self.read_image(os.path.join(self.hazy, img_name)) * 2 - 1
        target_img = self.read_image(os.path.join(self.gt, img_name)) * 2 - 1
        self.original_height, self.original_width = source_img.shape[:2]

        if self.mode:
            seed = int(time.time() * 1e6) % (2 ** 32 - 1) + item
            random.seed(seed)
            np.random.seed(seed)
            [source_img, target_img] = self.auguments([source_img, target_img], target_shape=self.target_shape)
            return self.hwc2chw(source_img), self.hwc2chw(target_img)
        else:
            # 归一化到 [-1, 1]
            hazy_patches = self.crop_nonoverlap(source_img)
            gt_patches = self.crop_nonoverlap(target_img)
            patch_count = len(hazy_patches)
        return hazy_patches, gt_patches, patch_count, os.path.basename(img_name), (self.original_height, self.original_width)


def val_collate_fn(batch):
    all_hazy_patches = []
    all_gt_patches = []
    all_patch_nums = []
    all_filenames = []
    all_sizes = []
    for hazy_list, gt_list, patch_count, filename, (h, w) in batch:
        hazy_tensor_list = []
        gt_tensor_list = []

        for hazy_np, gt_np in zip(hazy_list, gt_list):
            hazy_tensor = torch.from_numpy(hazy_np).float().unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
            gt_tensor = torch.from_numpy(gt_np).float().unsqueeze(0)
            hazy_tensor_list.append(hazy_tensor)
            gt_tensor_list.append(gt_tensor)
        all_hazy_patches.append(hazy_tensor_list)
        all_gt_patches.append(gt_tensor_list)
        all_patch_nums.append(patch_count)
        all_filenames.append(filename)
        all_sizes.append((h, w))  # 新增

    return all_hazy_patches, all_gt_patches, all_patch_nums, all_filenames, all_sizes

class RESIZEDatasetTest(BaseDataset):
    def __init__(
        self,
        root_dir,
        target_shape,
    ):
        super(RESIZEDatasetTest, self).__init__()
        self.target_shape = to_2tuple(target_shape)
        self.gt = os.path.join(root_dir, 'GT')
        self.hazy = os.path.join(root_dir, 'hazy')
        self.image_name_list = SearchFileName(self.gt, '.png')

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, item):
        self.disable_cv2_multithreading()
        img_name = self.image_name_list[item]
        source_img = self.read_image(os.path.join(self.hazy, img_name)) * 2 - 1
        target_img = self.read_image(os.path.join(self.gt, img_name)) * 2 - 1
        self.original_height, self.original_width = source_img.shape[:2]
        source_img = cv2.resize(source_img, self.target_shape, interpolation=cv2.INTER_NEAREST)
        target_img = cv2.resize(target_img, self.target_shape, interpolation=cv2.INTER_NEAREST)
        return self.hwc2chw(source_img), self.hwc2chw(target_img), img_name, (self.original_height, self.original_width)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from pyzjr.data.loaders import TrainDataloader,EvalDataloader
    # pyzjr.SeedEvery(11)
    data_path=r'E:\PythonProject\DehazeProject\data\RealHaze115'
    val_dataset=RealHaze115Dataset(
        root_dir=data_path,
        target_shape=256,
        is_train=False,
    )
    val_loader=EvalDataloader(val_dataset, batch_size=1,
                              num_workers=1, collate_fn=val_collate_fn)
    train_dataset=RealHaze115Dataset(
        root_dir=data_path,
        target_shape=256,
        is_train=True,
        repeat=4
    )
    train_loader=TrainDataloader(train_dataset, batch_size=1, num_workers=1)
    # for i, (hazy, gt, name) in enumerate(train_loader):
    #     print(i + 1, hazy.shape, gt.shape, name)
    #     # pyzjr.display(torch.stack([hazy[0], gt[0]]))
    #     save_paths = f"./test/{name[0].split('.')[0]}"
    #     os.makedirs(save_paths, exist_ok=True)
    #     pyzjr.imwrite(f"./test/{name[0].split('.')[0]}/{i+1}_{name[0]}", torch.stack([hazy[0], gt[0]]))
    for hazy, gt, patch_nums, filename, original_shape in val_loader:
        hazy_patches = hazy[0]
        gt_patches = gt[0]
        patch_num = patch_nums[0]
        name = filename[0]
        original_shape = original_shape[0]

        print(f"图像名: {name}, 切片数: {patch_num}")
        for i, (h, g) in enumerate(zip(hazy_patches, gt_patches)):
            print(f"第 {i + 1} 个切片 shape: {h.shape}")
            pyzjr.display(torch.stack([h[0], g[0]]))