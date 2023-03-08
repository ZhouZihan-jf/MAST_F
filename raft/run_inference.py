import os
import glob as gb
import logging

logging.basicConfig(filename="inference.log", filemode="w", format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                    datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)

data_path = "/dataset/zzh/youtube"
gap = [2, 5]
reverse = [0]
rgbpath = data_path + '/JPEGImages'  # path to the dataset
folder = gb.glob(os.path.join(rgbpath, '*'))

for r in reverse:
    for g in gap:
        for f in folder:
            logging.info('===> Running {}, gap {}'.format(f, g))
            print('===> Running {}, gap {}'.format(f, g))
            mode = 'raft-things.pth'  # model
            if r == 1:
                raw_outroot = data_path + '/Flows_gap-{}/'.format(g)  # where to raw flow
                # outroot = data_path + '/FlowImages_gap-{}/'.format(g)  # where to save the image flow
            elif r == 0:
                raw_outroot = data_path + '/Flows_gap{}/'.format(g)  # where to raw flow
                # outroot = data_path + '/FlowImages_gap{}/'.format(g)  # where to save the image flow
            os.system("python predict.py "
                      "--gap {} --model {} --path {} --reverse {} --raw_outroot {}"
                      .format(g, mode, f, r, raw_outroot))
