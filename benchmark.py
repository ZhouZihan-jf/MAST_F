import argparse
import os, time
import logger
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn
import numpy as np
import functional.feeder.dataset.Davis2017 as D
import functional.feeder.dataset.DavisLoaderLab as DL
from functional.utils.f_boundary import db_eval_boundary
from functional.utils.jaccard import db_eval_iou
from models.mast import MAST
from functional.utils.io import imwrite_indexed
from tools.score import AverageMeter

parser = argparse.ArgumentParser(description='MAST')

# Data options
parser.add_argument('--ref', type=int, default=0)

parser.add_argument('--datapath', type=str,
                    default='/dataset/zzh/DAVIS-2017-Unsupervised-trainval-Full-Resolution/DAVIS',
                    help='Data path for Davis')
parser.add_argument('--savepath', type=str, default='results/test',
                    help='Path for checkpoints and logs')
parser.add_argument('--resume', type=str, default='/home/zzh337/proj/MAST_F/results/train/checkpoint_epoch_9.pth',
                    help='Checkpoint file to resume')

args = parser.parse_args()

vns = ['bike-packing', 'blackswan', 'bmx-trees', 'breakdance', 'camel', 'car-roundabout', 'car-shadow', 'cows', 'dog',
       'drift-chicane',
       'drift-straight', 'goat', 'gold-fish', 'horsejump-high', 'judo', 'kite-surf', 'lab-coat', 'libby', 'loading',
       'mbike-trick',
       'motocross-jump', 'paragliding-launch', 'parkour', 'pigs', 'scooter-black', 'shooting', 'soapbox']


def main():
    args.training = False

    if not os.path.isdir(args.savepath):
        os.makedirs(args.savepath)
    log = logger.setup_logger(args.savepath + '/benchmark.log')
    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))

    # 返回两个list，分别为annotation_all, jpeg_all
    # [[文件名,...]] list中套list，里面一层的list是单个视频的文件名
    TrainData = D.dataloaderUn(args.datapath)
    TrainImgLoader = torch.utils.data.DataLoader(
        DL.myImageFloder(TrainData[0], TrainData[1], False),
        batch_size=1, shuffle=False, num_workers=0, drop_last=False
    )

    # 一段视频一段视频地加载
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,1,0"
    model = MAST(args).cuda()
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'])

    device_ids = [0, 1]
    model = nn.DataParallel(model, device_ids=device_ids).cuda()
    test(model, TrainImgLoader, log)


def test(model, dataloader, log):
    model.eval()

    torch.backends.cudnn.benchmark = True

    Fs = AverageMeter()
    Js = AverageMeter()

    n_b = len(dataloader)

    log.info("Start testing.")
    for b_i, (images_rgb, annotations) in enumerate(dataloader):
        # if b_i != 22:
        #     continue
        videoname = vns[b_i]
        print("processing video: ", videoname)
        images_rgb = [r.cuda() for r in images_rgb]
        annotations = [q.cuda() for q in annotations]

        N = len(images_rgb)
        outputs = [annotations[0].contiguous()]

        for i in range(N - 1):
            mem_gap = 2
            ref_index = [0] + list(filter(lambda x: x > 0, range(i, i - mem_gap * 3, -mem_gap)))[::-1]

            rgb_0 = [images_rgb[ind] for ind in ref_index]
            rgb_1 = images_rgb[i + 1]

            anno_0 = [outputs[ind] for ind in ref_index]
            anno_1 = annotations[i + 1]

            _, _, h, w = anno_0[0].size()

            max_class = anno_1.max()

            with torch.no_grad():
                _output = model(rgb_0, anno_0, rgb_1, ref_index, i + 1)
                _output = F.interpolate(_output, (h, w), mode='bilinear')

                output = torch.argmax(_output, 1, keepdim=True).float()
                # print("output.dtype",output.dtype)
                # print(output.shape)
                outputs.append(output)

            js, fs = [], []

            for classid in range(1, max_class + 1):
                obj_true = (anno_1 == classid).cpu().numpy()[0, 0]
                obj_pred = (output == classid).cpu().numpy()[0, 0]

                f = db_eval_boundary(obj_true, obj_pred)
                j = db_eval_iou(obj_true, obj_pred)

                fs.append(f)
                js.append(j)
                Fs.update(f)
                Js.update(j)

            ###
            folder = os.path.join(args.savepath, 'benchmark')
            if not os.path.exists(folder): os.mkdir(folder)

            output_folder = os.path.join(folder, D.catnames[b_i].strip())

            if not os.path.exists(output_folder):
                os.mkdir(output_folder)

            pad = ((0, 0), (0, 0))
            if i == 0:
                # output first mask
                output_file = os.path.join(output_folder, '%s.png' % str(0).zfill(5))
                out_img = anno_0[0][0, 0].cpu().numpy().astype(np.uint8)

                out_img = np.pad(out_img, pad, 'edge').astype(np.uint8)
                imwrite_indexed(output_file, out_img)

            output_file = os.path.join(output_folder, '%s.png' % str(i + 1).zfill(5))
            out_img = output[0, 0].cpu().numpy().astype(np.uint8)
            out_img = np.pad(out_img, pad, 'edge').astype(np.uint8)
            imwrite_indexed(output_file, out_img)

        info = '\t'.join(['Js: ({:.3f}). Fs: ({:.3f}).'
                         .format(Js.avg, Fs.avg)])

        log.info('[{}/{}] {}'.format(b_i, n_b, info))


if __name__ == '__main__':
    main()
