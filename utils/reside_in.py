"""
The dehazing dataset specific format is as follows:
    - RESIDE-IN
        - train
            - GT
            - hazy
        - test
            - GT
            - hazy
Use ITS as the training set and SOTS indoor as the test set.
"""
import os
import cv2
import numpy as np
from pyzjr.data.utils import to_2tuple
from pyzjr.data import BaseDataset, list_dirs
from pyzjr.data.utils import SearchFileName
from pyzjr.visualize.io import StackedImagesV1
from pyzjr.utils.dim_utils import chw2hwc

class RESIDEDataset(BaseDataset):
    def __init__(
        self,
        root_dir,
        target_shape,
        is_train=True,
    ):
        super(RESIDEDataset, self).__init__()
        self.mode=is_train
        self.target_shape=to_2tuple(target_shape)
        data_dir = os.path.join(root_dir, 'train' if is_train else 'test')
        self.gt = os.path.join(data_dir, 'GT')
        self.hazy = os.path.join(data_dir, 'hazy')
        self.image_name_list = SearchFileName(self.gt, '.png')

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, item):
        self.disable_cv2_multithreading()
        img_name = self.image_name_list[item]
        # normalize to [-1, 1]
        source_img = self.read_image(os.path.join(self.hazy, img_name)) * 2 - 1
        target_img = self.read_image(os.path.join(self.gt, img_name)) * 2 - 1
        if self.mode:
            [source_img, target_img] = self.auguments([source_img, target_img], target_shape=self.target_shape)
        else:
            [source_img, target_img] = self.align([source_img, target_img], self.target_shape)
        return self.hwc2chw(source_img), self.hwc2chw(target_img)


class RESIDEDatasetTest(BaseDataset):
    def __init__(
        self,
        root_dir,
        target_shape,
    ):
        super(RESIDEDatasetTest, self).__init__()
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
        [source_img, target_img] = self.align([source_img, target_img], self.target_shape)
        return self.hwc2chw(source_img), self.hwc2chw(target_img), img_name

def show_image_from_dataloader(dataset,end_k=27):
    loader=DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )
    for i,(hazy,gt) in enumerate(loader):
        gt_img=gt[0].numpy()
        hazy_img=hazy[0].numpy()
        gt_img=chw2hwc(gt_img)
        hazy_img=chw2hwc(hazy_img)
        gt_img=((gt_img + 1)*127.5)[:,:,::-1].astype(np.uint8)
        hazy_img=((hazy_img + 1)*127.5)[:,:,::-1].astype(np.uint8)
        stack_img=StackedImagesV1(1,[[gt_img,hazy_img]])
        cv2.imshow('GT & Hazy',stack_img)
        k=cv2.waitKey(0)
        if k == end_k:
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from pyzjr.data.loaders import TrainDataloader,EvalDataloader

    data_path=r'E:\PythonProject\DehazeProject\data\O_HAZY'
    train_dataset=RESIDEDataset(
        root_dir=data_path,
        target_shape=256,
    )
    train_loader=TrainDataloader(train_dataset, batch_size=4)

    # 验证阶段，可以用类似的代码从验证集读取
    val_dataset=RESIDEDataset(
        root_dir=data_path,
        target_shape=256,
        is_train=False,
    )
    val_loader=EvalDataloader(val_dataset, batch_size=4)




    # 训练阶段，使用训练集
    # for i, (gt, hazy) in enumerate(train_loader):
    # 	print(f"Training - Step {i}, GT Shape: {gt.shape}, Hazy Shape: {hazy.shape}")
    # show_image_from_dataloader(train_dataset)

    # for j, (gt, hazy) in enumerate(val_loader) :
    # 	print(f"Validation - Step {j}, GT Shape: {gt.shape}, Hazy Shape: {hazy.shape}")
