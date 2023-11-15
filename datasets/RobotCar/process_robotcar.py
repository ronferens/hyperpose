import os
import os.path as osp
from PIL import Image
from datasets.RobotCar.RobotCarDataset import RobotCar
from torch.utils.data import DataLoader
from util import utils

opt = {'nThreads': 8,
       'cropsize': 256,
       'dataset': 'RobotCar',
       'scene': 'loop',
       'dataset_path': '/media/dev/data/datasets/robotcar',
       'labels_path': '/home/dev/git/multi-scene-pose-transformer/datasets/RobotCar',
       'val': False,
       'batchsize': 64,
       }

# Creating the data loader
base_dir = osp.join(opt.get('labels_path'), opt.get('scene'))
if opt.get('val'):
    print('processing VAL data using {:d} cores'.format(opt.get('nThreads')))

    split_filename = osp.join(base_dir, 'test_split.txt')
    split_dir = 'test'
    transform = utils.test_transforms.get('robotcar')
else:
    print('processing TRAIN data using {:d} cores'.format(opt.get('nThreads')))

    split_filename = osp.join(base_dir, 'train_split.txt')
    split_dir = 'train'
    transform = utils.train_transforms.get('robotcar')

dset = RobotCar(scene=opt.get('scene'), dataset_path=opt.get('dataset_path'), labels_path=opt.get('labels_path'),
                train=not opt.get('val'), transform=transform, undistort=True)
loader = DataLoader(dset, batch_size=opt.get('batchsize'), num_workers=opt.get('nThreads'))

# Gather information about output filenames
with open(split_filename, 'r') as f:
    seqs = [l.rstrip() for l in f if not l.startswith('#')]

im_filenames = []
for seq in seqs:
    seq_dir = osp.join(opt.get('dataset_path'), opt.get('scene'), split_dir, seq)
    process_imgs_dir = osp.join(seq_dir, 'stereo', 'centre_processed')
    if not osp.exists(process_imgs_dir):
        os.makedirs(process_imgs_dir)

    ts_filename = osp.join(seq_dir, 'stereo.timestamps')
    with open(ts_filename, 'r') as f:
        ts = [l.rstrip().split(' ')[0] for l in f]
    im_filenames.extend([osp.join(process_imgs_dir, '{:s}.png'.format(t)) for t in ts])
assert len(dset) == len(im_filenames)

# Main processing loop
for batch_idx, (imgs, _) in enumerate(loader):
    for idx, im in enumerate(imgs):
        im_filename = im_filenames[batch_idx * opt.get('batchsize') + idx]
        im = Image.fromarray(im.numpy())
        try:
            im.save(im_filename)
        except IOError:
            print('IOError while saving {:s}'.format(im_filename))

    if batch_idx % 10 == 0:
        print('Processed {:d} / {:d}'.format(batch_idx * opt.get('batchsize'), len(dset)))
