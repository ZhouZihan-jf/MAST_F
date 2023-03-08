import functional.feeder.dataset.YouTubeVOSTrain as Y
import functional.feeder.dataset.YTVOSTrainLoader as YL

import torch
import numpy as np

if __name__ == '__main__':
    TrainData = Y.dataloader('functional/feeder/dataset/ytvos.csv', 20)
    # TrainData <list>
    TrainImgLoader = torch.utils.data.DataLoader(
        YL.myImageFloder('/dataset/dusen/youtube2018/train_all_frames/JPEGImages/', TrainData, True),
        batch_size=1, shuffle=True, num_workers=12, drop_last=True
    )
    for b_i, (images_lab, images_rgb_, images_quantized) in enumerate(TrainImgLoader):
        if b_i == 1:
            break
        print('开始测试')
        #images_lab_gt = [lab.clone().cuda() for lab in images_lab]
        #images_lab = [r.cuda() for r in images_lab]
        #images_rgb_ = [r.cuda() for r in images_rgb_]
        drop_ch_num = 1
        drop_ch_ind = np.random.choice(np.arange(1,3), 1, replace=False)
        print(images_lab[0])
        for a in images_lab:
            for dropout_ch in drop_ch_ind:
                a[:, dropout_ch] = 0
            a *= (3 / (3 - drop_ch_num))
        print(images_lab[0])
        