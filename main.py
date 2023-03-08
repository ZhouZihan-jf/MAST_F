import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
from torch.utils.tensorboard import SummaryWriter
import functional.feeder.dataset.YouTubeVOSTrain as Y
import functional.feeder.dataset.YTVOSTrainLoader as YL
import functional.feeder.dataset.FlowOrganization as O
import matplotlib.pyplot as plt
from models.mast import MAST
from tools.score import AverageMeter
import matplotlib
import logger

matplotlib.use('Agg')

parser = argparse.ArgumentParser(description='MAST')

# Data options
parser.add_argument('--basepath', default='/dataset/zzh/youtube',
                    help='Data path for Kinetics')
parser.add_argument('--datapath', default='/dataset/zzh/youtube/JPEGImages/',
                    help='Data path for Kinetics')
parser.add_argument('--validpath', default='',
                    help='Data path for Davis')
parser.add_argument('--res', default='',
                    help='图片像素')
parser.add_argument('--csvpath', default='functional/feeder/dataset/ytvos_t.csv',
                    help='Path for csv file')
parser.add_argument('--savepath', type=str, default='results/train',
                    help='Path for checkpoints and logs')
parser.add_argument('--resume', type=str, default=None,
                    help='Checkpoint file to resume')

# Training options
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--bsize', type=int, default=6,
                    help='batch size for training (default: 12)')
parser.add_argument('--worker', type=int, default=3,
                    help='number of dataloader threads')

args = parser.parse_args()


def main():
    args.training = True

    if not os.path.isdir(args.savepath):
        os.makedirs(args.savepath)
    log = logger.setup_logger(args.savepath + '/training.log')
    writer = SummaryWriter(args.savepath + '/runs/')

    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))

    TrainData = Y.dataloader(args.csvpath)
    FlowData = O.YoutubeOrganization(args)
    TrainImgAndFlwLoader = torch.utils.data.DataLoader(
        # YL.myImageFloder(args.datapath, TrainData, True),
        YL.myImageAndFlowFloder(args.datapath, TrainData, FlowData, True),
        batch_size=12, shuffle=True, num_workers=args.worker, drop_last=True
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = "2,1,0"  # 设定gpu编号。此处主卡0对应2号
    model = MAST(args).cuda()

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    start_full_time = time.time()

    device_ids = [0]  # 主卡0号索引对应实际的2号卡
    model = nn.DataParallel(model, device_ids=device_ids).cuda()

    for epoch in range(args.epochs):
        log.info('This is {}-th epoch'.format(epoch))
        train(TrainImgAndFlwLoader, model, optimizer, log, writer, epoch)
        # TrainImgLoader <torch.utils.data.dataloader.DataLoader>
        '''
        TrainData = Y.dataloader(args.csvpath, epoch)
        FlowData = O.YoutubeOrganization(args)
        # TrainData <list>
        TrainImgLoader = torch.utils.data.DataLoader(
            YL.myImageAndFlowFloder(args.datapath, TrainData, FlowData, True),
            batch_size=args.bsize, shuffle=True, num_workers=args.worker, drop_last=True
        )
        '''

    log.info('full training time = {:.2f} Hours'.format((time.time() - start_full_time) / 3600))


iteration = 0


def train(dataloader, model, optimizer, log, writer, epoch):
    global iteration
    _loss = AverageMeter()
    n_b = len(dataloader)
    b_s = time.perf_counter()

    for b_i, (images_lab, images_rgb_, flows, images_quantized) in enumerate(dataloader):
        # print(f"img:{np.array(images_lab[0]).shape},flow:{np.array(flows[0]).shape}")
        model.train()
        adjust_lr(optimizer, epoch, b_i, n_b)
        # 帧对[batch_size,3,256,256, batch_size,3,256,256], 相当于复制了images_lab
        images_lab_gt = [lab.clone().cuda() for lab in images_lab]
        # 帧对[batch_size,3,256,256, batch_size,3,256,256]
        images_lab = [r.cuda() for r in images_lab]
        images_rgb_ = [r.cuda() for r in images_rgb_]
        flows = [r.cuda() for r in flows]

        # 选择lab中a或者b通道进行置0，然后整体乘1.5
        # ch为图片的通道坐标(1或者2)
        _, ch = model.module.dropout2d_lab(images_lab)
        # _, ch = model.module.dropout2d_lab_fill_flow(images_lab, flows)

        sum_loss, err_maps = compute_lphoto(model, images_lab, images_lab_gt, flows, ch)

        sum_loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        _loss.update(sum_loss.item())

        iteration = iteration + 1
        writer.add_scalar("Training loss", sum_loss.item(), iteration)

        info = 'Loss = {:.3f}({:.3f})'.format(_loss.val, _loss.avg)
        b_t = time.perf_counter() - b_s
        b_s = time.perf_counter()

        for param_group in optimizer.param_groups:
            lr_now = param_group['lr']
        log.info('Epoch{} [{}/{}] {} T={:.2f}  LR={:.6f}'.format(
            epoch, b_i, n_b, info, b_t, lr_now))

        if (b_i * args.bsize) % 2000 < args.bsize:
            b = 0
            fig = plt.figure(figsize=(16, 3))
            # Input
            plt.subplot(151)
            image_rgb0 = images_rgb_[0][b].cpu().permute(1, 2, 0)
            plt.imshow(image_rgb0)
            plt.title('Frame t')

            plt.subplot(152)
            image_rgb1 = images_rgb_[1][b].cpu().permute(1, 2, 0)
            plt.imshow(image_rgb1)
            plt.title('Frame t+1')

            plt.subplot(153)
            plt.imshow(torch.abs(image_rgb1 - image_rgb0))
            plt.title('Frame difference ')

            # Error map
            plt.subplot(154)
            err_map = err_maps[b]
            plt.imshow(err_map.cpu(), cmap='jet')
            plt.colorbar()
            plt.title('Error map')

            writer.add_figure('ErrorMap', fig, iteration)

        n_iter = b_i + n_b * epoch

    log.info("Saving checkpoint.")
    savefilename = args.savepath + f'/checkpoint_epoch_{epoch}.pth'
    torch.save({
        'epoch': epoch,
        'state_dict': model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, savefilename)


def compute_lphoto(model, image_lab, images_rgb_, flows, ch):
    # compute_lphoto(model, images_lab, images_lab_gt, ch)
    # image_lab是drop后的图片   images_lab_gt是drop前的
    b, c, h, w = image_lab[0].size()

    ref_x = []
    for lab in image_lab[:-1]:
        for flow in flows[:-1]:
            lab = torch.cat([lab, flow], dim=1)
            ref_x.append(lab)
            # print(f"lab:{lab.cpu().size()}")

    # ref_x = [lab for lab in image_lab[:-1]]  # [im1, im2, im3]
    ref_y = [rgb[:, ch] for rgb in images_rgb_[:-1]]  # [y1, y2, y3]
    # tar_x = image_lab[-1]  # im4
    tar_x = torch.cat([image_lab[-1], flows[-1]], dim=1)
    tar_y = images_rgb_[-1][:, ch]  # y4
    # print(f"ref_y:{ref_y[0].cpu().size()},tar_x:{tar_x.cpu().size()},tar_y:{tar_y.cpu().size()}")

    # 两张图drop的通道相同
    # 前一张图片：ref_x是drop后的图片(3,256,256)，ref_y是drop掉的那个通道(1,256,256)
    # 后一张图片：tar_x是drop后的图片，tar_y是drop掉的那个通道
    outputs = model(ref_x, ref_y, tar_x, [0, 2], 4)  # only train with pairwise data 仅使用成对数据训练
    # 前一张图片drop后的图片，drop掉的通道，后一张图片drop后的图片
    # outputs.shape=batch_size,1,64,64

    # outputs_back = model(tar_x[0], tar_y[0], [ref_x], [0,2], 4)

    outputs = F.interpolate(outputs, (h, w), mode='bilinear')

    loss = F.smooth_l1_loss(outputs * 20, tar_y * 20, reduction='mean')

    err_maps = torch.abs(outputs - tar_y).sum(1).detach()

    return loss, err_maps


def adjust_lr(optimizer, epoch, batch, n_b):
    iteration = (batch + epoch * n_b) * args.bsize

    if iteration <= 400000:
        lr = args.lr
    elif iteration <= 600000:
        lr = args.lr * 0.5
    elif iteration <= 800000:
        lr = args.lr * 0.25
    elif iteration <= 1000000:
        lr = args.lr * 0.125
    else:
        lr = args.lr * 0.0625

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
