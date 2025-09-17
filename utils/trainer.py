import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from pyzjr.nn import release_gpu_memory, get_lr, AverageMeter, LossHistory, get_optimizer
from utils.index import dehaze_index

class DeHazeTrainEpoch(object):
    def __init__(self,
                 model,
                 total_epoch,
                 optimizer,
                 lr_scheduler,
                 loss_function,
                 use_amp=False,
                 device=torch.device("cuda:0")):
        super(DeHazeTrainEpoch, self).__init__()
        self.device = device
        self.model = model.to(device)
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.total_epoch = total_epoch
        release_gpu_memory()
        self.scaler = None
        if use_amp:
            self.scaler = GradScaler()

    def train_one_epoch(self, train_loader, epoch):
        self.model.train()
        losses = AverageMeter()
        psnr_meter = AverageMeter()

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{self.total_epoch}', postfix=dict,
                  mininterval=0.3) as pbar:
            for batch in train_loader:
                source_img, target_img = batch[0].to(self.device).float(), \
                                         batch[1].to(self.device).float()
                with autocast(enabled=self.scaler is not None):
                    outputs = self.model(source_img)
                self.optimizer.zero_grad()
                if self.scaler is not None:
                    loss = torch.nan_to_num(self.loss_function(outputs, target_img, source_img))
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss = self.loss_function(outputs, target_img, source_img)
                    loss.backward()
                    self.optimizer.step()
                self.lr_scheduler.step()
                psnr = dehaze_index(outputs, target_img, ssim_index=False)
                psnr_meter.update(psnr, source_img.size(0))
                losses.update(loss.item())
                pbar.set_postfix(**{'train_loss': losses.avg,
                                    'psnr': psnr_meter.avg,
                                    'lr': get_lr(self.optimizer)})
                pbar.update(1)
        return losses.avg

    def evaluate(self, val_loader, epoch):
        self.model.eval()
        PSNR = AverageMeter()
        SSIM = AverageMeter()
        with tqdm(total=len(val_loader), desc=f'Epoch {epoch}/{self.total_epoch}', postfix=dict,
                  mininterval=0.3) as pbar:
            for batch in val_loader:
                source_img, target_img = batch[0].to(self.device), batch[1].to(self.device)
                with torch.no_grad():
                    outputs = self.model(source_img).clamp_(-1, 1)
                    loss = self.loss_function(outputs, target_img, source_img)
                    psnr, ssim_val = dehaze_index(outputs, target_img)
                PSNR.update(psnr, source_img.size(0))
                SSIM.update(ssim_val)
                pbar.set_postfix(**{'psnr': PSNR.avg})
                pbar.update(1)
            print(f"{epoch} - psnr: {PSNR.avg}, ssim: {SSIM.avg}")
        return loss.item(), PSNR.avg


class DeHazeTrainEpoch2(object):
    def __init__(self,
                 model,
                 total_epoch,
                 optimizer,
                 lr_scheduler,
                 loss_function,
                 use_amp=False,
                 device=torch.device("cuda:0")):
        super(DeHazeTrainEpoch2, self).__init__()
        self.device = device
        self.model = model.to(device)
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.total_epoch = total_epoch
        release_gpu_memory()
        self.scaler = None
        if use_amp:
            self.scaler = GradScaler()

    def train_one_epoch(self, train_loader, epoch):
        self.model.train()
        losses = AverageMeter()
        psnr_meter = AverageMeter()

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{self.total_epoch}', postfix=dict,
                  mininterval=0.3) as pbar:
            for batch in train_loader:
                source_img, target_img = batch[0].to(self.device).float(), \
                                         batch[1].to(self.device).float()

                self.optimizer.zero_grad()
                with autocast(enabled=self.scaler is not None):
                    outputs = self.model(source_img)
                if self.scaler is not None:
                    loss = torch.nan_to_num(self.loss_function(outputs, target_img, source_img))
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss = self.loss_function(outputs, target_img, source_img)
                    loss.backward()
                    self.optimizer.step()
                self.lr_scheduler.step()
                psnr = dehaze_index(outputs, target_img, ssim_index=False)
                psnr_meter.update(psnr, source_img.size(0))
                losses.update(loss.item())
                pbar.set_postfix(**{'train_loss': losses.avg,
                                    'psnr': psnr_meter.avg,
                                    'lr': get_lr(self.optimizer)})
                pbar.update(1)
        return losses.avg

    def evaluate(self, val_loader):
        self.model.eval()
        total_PSNR = AverageMeter()
        total_SSIM = AverageMeter()
        # per_image_metrics = []
        # Every 5 images belong to a sub-dataset.
        dataset_names = ['O-Haze', 'NH-Haze', 'NH-Haze2']
        dataset_metrics = {
            name: {
                'psnr_sum': 0.0,
                'ssim_sum': 0.0,
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
                h, g = h.to(self.device), g.to(self.device)
                outputs = self.model(h).clamp_(-1, 1)

                psnr, ssim_val = dehaze_index(outputs, g)
                psnr_sum += psnr
                ssim_sum += ssim_val

            avg_psnr = psnr_sum / patch_num
            avg_ssim = ssim_sum / patch_num
            # per_image_metrics.append((name, avg_psnr, avg_ssim))
            total_PSNR.update(avg_psnr)
            total_SSIM.update(avg_ssim)
            print(
                f"    ==> 图像: {name} 平均 PSNR: {total_PSNR.val:.2f}({total_PSNR.avg:.4f}), 平均 SSIM: {total_SSIM.val:.3f}({total_SSIM.avg:.4f}), 数据集: {dataset_name}")
            if dataset_name in dataset_metrics:
                dataset_metrics[dataset_name]['psnr_sum'] += avg_psnr
                dataset_metrics[dataset_name]['ssim_sum'] += avg_ssim

        print("\n==== 各数据集的平均指标 ====")
        for dataset_name in dataset_names:
            metrics = dataset_metrics[dataset_name]
            avg_psnr = metrics['psnr_sum'] / 5
            avg_ssim = metrics['ssim_sum'] / 5
            print(f"[{dataset_name}] 平均 PSNR: {avg_psnr:.5f}, 平均 SSIM: {avg_ssim:.5f}")

        print(f"[RealHaze115] 平均 PSNR: {total_PSNR.avg:.5f}, 平均 SSIM: {total_SSIM.avg:.5f}")
        return total_PSNR.avg, total_SSIM.avg
