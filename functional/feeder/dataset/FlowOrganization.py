import itertools
import os
import glob as gb

import numpy as np

pairs = [1, 2, -1, -2]
img_dir = ''
gt_dir = ''


# 对youtube数据集的光流文件的组织
def YoutubeOrganization(args):
    pair_list = [p for p in itertools.combinations(pairs, 2)]
    folders = [os.path.basename(x) for x in gb.glob(os.path.join(args.basepath, 'Flows_gap1/{}/*'.format(args.res)))]
    flow_dir = {}
    for pair in pair_list:
        p1, p2 = pair
        flowpairs = []
        for f in folders:
            path1 = os.path.join(args.basepath, 'Flows_gap{}/{}/{}'.format(p1, args.res, f))
            path2 = os.path.join(args.basepath, 'Flows_gap{}/{}/{}'.format(p2, args.res, f))

            flows1 = [os.path.basename(x) for x in gb.glob(os.path.join(path1, '*'))]
            flows2 = [os.path.basename(x) for x in gb.glob(os.path.join(path2, '*'))]

            intersect = list(set(flows1).intersection(flows2))
            intersect.sort()

            flowpair = np.array([[os.path.join(path1, i), os.path.join(path2, i)] for i in intersect])
            flowpairs += [flowpair]
        flow_dir['gap_{}_{}'.format(p1, p2)] = flowpairs

    img_dir = args.datapath

    data_dir = [flow_dir, img_dir, gt_dir]
    return data_dir
