import numpy as np
import h5py
import os
import sys
import traceback
import os.path as osp
from synthgen import *
from common import *
import wget
import tarfile
import cv2 as cv
import time

# 定义一些配置
INSTANCE_PER_IMAGE = 1  # no. of times to use the same image
SECS_PER_IMG = 10  # max time per image in seconds
NUM_IMG = -1  # no. of images to use for generation (-1 to use all available):

# path to the data-file, containing image, depth and segmentation:
DATA_PATH = 'data'
DB_FNAME = osp.join(DATA_PATH, 'dset.h5')
# url of the data (google-drive public file):
DATA_URL = 'http://www.robots.ox.ac.uk/~ankush/data.tar.gz'
OUT_FILE = 'results/SynthText_viz.h5'


def get_data():
    """
    Download the image,depth and segmentation data:
    Returns, the h5 database.
    """
    if not osp.exists(DB_FNAME):
        try:
            colorprint(Color.BLUE, '\tdownloading data (56 M) from: ' +
                       DATA_URL, bold=True)
            
            sys.stdout.flush()
            out_fname = 'data.tar.gz'
            wget.download(DATA_URL, out=out_fname)
            tar = tarfile.open(out_fname)
            tar.extractall()
            tar.close()
            os.remove(out_fname)
            colorprint(Color.BLUE, '\n\tdata saved at:' + DB_FNAME, bold=True)
            sys.stdout.flush()
        except:
            print(colorize(
                Color.RED, 'Data not found and have problems downloading.', bold=True))
            sys.stdout.flush()
            sys.exit(-1)
    # open the h5 file and return:
    return h5py.File(DB_FNAME, 'r')


def add_res_to_db(imgname, res, db):
    """
  Add the synthetically generated text image instance
  and other metadata to the dataset.
  """
    ninstance = len(res)
    for i in range(ninstance):
        print(colorize(Color.GREEN, 'added into the db %s ' % res[i]['txt']))

        dname = "%s_%d" % (imgname, i)
        db['data'].create_dataset(dname, data=res[i]['img'])
        db['data'][dname].attrs['charBB'] = res[i]['charBB']
        db['data'][dname].attrs['wordBB'] = res[i]['wordBB']
        print('type of res[i][\'txt\'] ', type(res[i]['txt']))

        # db['data'][dname].attrs['txt'] = res[i]['txt']
        db['data'][dname].attrs.create(
            'txt', res[i]['txt'], dtype=h5py.special_dtype(vlen=str))
        print('type of db ', type(db['data'][dname].attrs['txt']))
        print(colorize(Color.GREEN, 'successfully added'))
        print(res[i]['txt'])
        print(res[i]['img'].shape)
        print('charBB', res[i]['charBB'].shape)
        print('charBB', res[i]['charBB'])
        print('wordBB', res[i]['wordBB'].shape)
        print('wordBB', res[i]['wordBB'])
        '''
        img = Image.fromarray(res[i]['img'])
        hsv_img=np.array(rgb2hsv(img))
        print 'hsv_img_shape',hsv_img.shape
        print 'hsv_img',hsv_img
        H=hsv_img[:,:,2]
        print 'H_channel',H.shape,H
        #img = Image.fromarray(db['data'][dname][:])
        '''


def rgb2hsv(image):
    return image.convert('HSV')


def rgb2gray(image):
    rgb = np.array(image)

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def main(viz=False):
    # open databases:
    print(colorize(Color.BLUE, 'getting data..', bold=True))
    db = get_data()
    print(colorize(Color.BLUE, '\t-> done', bold=True))

    # open the output h5 file:
    out_db = h5py.File(OUT_FILE, 'w')
    out_db.create_group('/data')
    print(colorize(Color.GREEN, 'Storing the output in: ' + OUT_FILE, bold=True))

    # get the names of the image files in the dataset:
    imnames = sorted(db['image'].keys())
    N = len(imnames)
    global NUM_IMG
    if NUM_IMG < 0:
        NUM_IMG = N
    start_idx, end_idx = 0, min(NUM_IMG, N)

    RV3 = RendererV3(DATA_PATH, max_time=SECS_PER_IMG)

    for i in range(start_idx, end_idx):
        t1 = time.time()
        imname = imnames[i]
        print(imname)
        try:

            # get the image:
            img = Image.fromarray(db['image'][imname][:])

            img_resize = img.resize(db['depth'][imname].shape[0:3][1:3])
            depth = db['depth'][imname][:].T

            hsv_img = np.array(rgb2hsv(img_resize))
            # print 'hsv_img',hsv_img
            H = hsv_img[:, :, 2]
            H = H.T
            H = H.astype('float32')




            gray = np.array(rgb2gray(img_resize))
            depth = (np.max(depth) / np.max(gray)) * gray.astype('float32')

            kernel = np.ones((5, 5), np.float32) / 25
            gray = cv2.filter2D(gray, -1, kernel)

            seg = db['seg'][imname][:].astype('float32')
            area = db['seg'][imname].attrs['area']
            label = db['seg'][imname].attrs['label']

            print('seg info', seg.shape, area.shape, label.shape)
            sz = depth.shape[:2][::-1]
            img = np.array(img.resize(sz, Image.ANTIALIAS))
            seg = np.array(Image.fromarray(seg).resize(sz, Image.NEAREST))

            print(colorize(Color.RED, '%d of %d' %
                           (i, end_idx - 1), bold=True))
            res = RV3.render_text(img, depth, seg, area, label,
                                  ninstance=INSTANCE_PER_IMAGE, viz=viz)
            t2 = time.time()

            for ct in range(10):

                if len(res) > 0:
                    # non-empty : successful in placing text:
                    add_res_to_db(imname, res, out_db)
                    break
                else:
                    res = RV3.render_text(img, depth, seg, area, label,
                                          ninstance=INSTANCE_PER_IMAGE, viz=viz)
            # visualize the output:
            print('time consume in each pic', t2 - t1)
            if viz:
                if 'q' in input(colorize(Color.RED, 'continue? (enter to continue, q to exit): ', True)):
                    break
        except:
            traceback.print_exc()
            print(colorize(Color.GREEN, '>>>> CONTINUING....', bold=True))
            continue
    db.close()
    out_db.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Genereate Synthetic Scene-Text Images')
    parser.add_argument('--viz', action='store_true', dest='viz', default=True,
                        help='flag for turning on visualizations')
    args = parser.parse_args()
    main(args.viz)
