import os, sys
import os.path
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def dataloader(csv_path="ytvos.csv", E=0):
    # 列表 元素：文件名,起始帧(第一张图片名),数量
    filenames = open(csv_path).readlines()

    # 将三个变量拆分，得到三个列表
    frame_all = [filename.split(',')[0].strip() for filename in filenames]
    startframe = [int(filename.split(',')[1].strip()) for filename in filenames]
    nframes = [int(filename.split(',')[2].strip()) for filename in filenames]

    all_index = np.arange(len(nframes))
    np.random.shuffle(all_index)
    # 随机化

    refs_train = []

    for index in all_index:
        ref_num = 2

        # 以概率0.4，0.4，0.2随机从原始数据【2，5，8】里抽取数
        frame_interval = np.random.choice([2, 5, 8], p=[0.4, 0.4, 0.2])

        # compute frame index (ensures length(image set) >= random_interval)
        refs_images = []

        n_frames = nframes[index]
        start_frame = startframe[index]
        frame_indices = np.arange(start_frame, start_frame + n_frames, frame_interval)  # start from startframe
        # 相当于对原视频进行了采样，采样频率为2,5,8(p=0.4,0.4,0.2)
        total_batch, batch_mod = divmod(len(frame_indices), ref_num)
        # divmod(7,2)=(3,1) divmod(3,2)=(1,1)

        # 如果为奇数，去掉最后一个
        if batch_mod > 0:
            frame_indices = frame_indices[:-batch_mod]
        # 分成total_batch份（这里是两帧两帧分开）
        frame_indices_batches = np.split(frame_indices, total_batch)

        for batches in frame_indices_batches:
            # ref_images = [os.path.join(frame_all[index], '{:05d}.jpg'.format(frame))
            #               for frame in [max(start_frame,batches[0]-30)]+ list(batches)]
            ref_images = [os.path.join(frame_all[index], '{:05d}.jpg'.format(frame))
                          for frame in list(batches)]
            refs_images.append(ref_images)

        refs_train.extend(refs_images)
    # list：[['d87069ba86/00000.jpg', 'd87069ba86/00005.jpg'], ...]
    # 元素是帧对，样例为上
    return refs_train


if __name__ == '__main__':
    x = dataloader()
    print(len(x))
