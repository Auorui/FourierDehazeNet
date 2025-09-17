import os
import argparse
import torch
from models import get_dehaze_networks
import pyzjr
from pyzjr.nn import release_gpu_memory, get_lr, AverageMeter, LossHistory, get_optimizer
from pyzjr.visualize.printf import redirect_console
from pyzjr.data import loss_weights_dirs, TrainDataloader, EvalDataloader
from utils import dehaze_criterion, GradualWarmupScheduler, RESIDEDataset, DeHazeTrainEpoch

def parse_args(known=False):
    parser = argparse.ArgumentParser(description='Dehazy')
    # 选择哪种网络
    parser.add_argument('--model', type=str, default='SFHformer_s', help='train net name')
    # 加载预训练权重路径，默认None
    parser.add_argument('--resume_training', type=str,
                        default=None,
                        help="resume training from last checkpoint")
    # 数据集路径
    parser.add_argument('--dataset_path', type=str,
                        default=r'E:\PythonProject\DehazeProject\data\SateHaze1K\Haze1k_thin',
                        help='dataset path')
    # 随机种子数: 11, 42, 3407, 114514, 256
    parser.add_argument('--seed', type=int, default=11, help='Random seed number')
    # 训练轮次epochs次数，默认为500轮
    parser.add_argument('--epochs', type=int, default=500, help='Training rounds')
    # 图片大小
    parser.add_argument('--input_shape', default=[256, 256], help='input image shape')
    # batch_size 批量大小 2 4 8,爆内存就改为1试试
    # 如果还有问题请看此篇： https://blog.csdn.net/m0_62919535/article/details/132725967
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    # 日志文件存放路径
    parser.add_argument('--log_dir',  type=str, default=r'./logs', help='log file path')
    # 初始学习率
    parser.add_argument('--lr', default=2e-4, help='Initial learning rate')
    # 用于优化器的动量参数，控制梯度更新的方向和速度。
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for optimizer')
    # 用于优化器的权重衰减参数，用于抑制权重的过度增长，防止过拟合。
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay for optimizer')
    # 优化器选择，可选adam、adamw、sgd
    parser.add_argument('--optimizer', type=str, default="adamw", help='Optimizer selection, optional adam and sgd')
    # 混合精度训练，要求torch版本大于等于1.17.1
    parser.add_argument('--amp', type=bool, default=False, help='Mixed precision training')

    return parser.parse_known_args()[0] if known else parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    pyzjr.SeedEvery(args.seed)
    loss_log_dir, save_model_dir, timelog_dir = loss_weights_dirs(args.log_dir)
    redirect_console(os.path.join(timelog_dir, 'out.log'))
    pyzjr.show_config(args=args)
    network = get_dehaze_networks(args.model)

    if torch.cuda.device_count() > 1:
        network = torch.nn.DataParallel(network).to(f'cuda:{0}')
    elif torch.cuda.is_available():
        network = network.to(f'cuda:{0}')
    else:
        network = network.to('cpu')

    if args.resume_training is not None:
        # load checkpoint
        print(f"权重 {args.resume_training} 加载到 {args.model}")
        # state_dict = torch.load(args.resume_training)
        # from collections import OrderedDict
        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     name = k[7:]
        #     new_state_dict[name] = v
        network.load_state_dict(torch.load(args.resume_training, map_location=torch.device('cuda:0')))
    else:
        print(f"无权重训练 {args.model}")

    loss_history = LossHistory(loss_log_dir, network, input_shape=args.input_shape)

    # criterion = nn.L1Loss()
    criterion = dehaze_criterion()
    optimizer = get_optimizer(
        network,
        optimizer_type=args.optimizer,
        init_lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    train_dataset = RESIDEDataset(args.dataset_path, target_shape=args.input_shape,
                                 is_train=True)
    val_dataset = RESIDEDataset(args.dataset_path, target_shape=args.input_shape,
                                 is_train=False)
    
    train_loader = TrainDataloader(train_dataset, batch_size=args.batch_size, num_workers=1)
    val_loader = EvalDataloader(val_dataset, batch_size=1, num_workers=1)

    warmup_epochs = 3
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)

    Defogging = DeHazeTrainEpoch(
        network, args.epochs, optimizer, lr_scheduler, criterion, use_amp=args.amp
    )
    for epoch in range(args.epochs):
        epoch = epoch + 1
        train_loss = Defogging.train_one_epoch(train_loader, epoch)
        val_loss, psnr = Defogging.evaluate(val_loader, epoch)

        loss_history.append_loss(epoch, train_loss, val_loss)

        print('Epoch:' + str(epoch) + '/' + str(args.epochs))
        print('Total Loss: %.5f || Val Loss: %.5f ' % (train_loss, val_loss))

        pyzjr.SaveModelPth(
            network,
            save_dir=save_model_dir,
            metric=psnr,
            epoch=epoch,
            total_epochs=args.epochs,
            save_period=250
        )

    loss_history.close()