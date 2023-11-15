import os
import os.path as osp
from PIL import Image
from datasets.RobotCar.RobotCarDataset import RobotCar
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import pandas as pd

opt = {'nThreads': 8,
       'cropsize': 256,
       'dataset': 'RobotCar',
       'scenes': ['loop'],
       'dataset_path': '/media/dev/data/datasets/robotcar',
       'labels_path': '/home/dev/git/multi-scene-pose-transformer/datasets/RobotCar',
       'batchsize': 64,
       'output_csv_filename': 'abs_robotcar_pose_loop_train.csv',
       'data_to_process': ['train']
       }

for scene in opt.get('scenes'):
    for data_type in opt.get('data_to_process'):

        # Creating the data loader
        base_dir = osp.join(opt.get('labels_path'), scene)
        print(f'processing {data_type} data using {opt.get("nThreads")} cores')
        split_filename = osp.join(base_dir, f'{data_type}_split.txt')

        if data_type == 'test':
            is_train_data_type = False
        else:
            is_train_data_type = True

        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize(opt.get('cropsize')),
                                        transforms.Lambda(lambda x: np.asarray(x))])
        dset = RobotCar(scene=scene,
                        dataset_path=opt.get('dataset_path'),
                        labels_path=opt.get('labels_path'),
                        train=is_train_data_type,
                        real=True,
                        transform=transform, undistort=True)
        dataloader = DataLoader(dset, batch_size=opt.get('batchsize'), num_workers=opt.get('nThreads'))

        # Gather information about output filenames
        with open(split_filename, 'r') as f:
            seqs = [l.rstrip() for l in f if not l.startswith('#')]
        print(f'Found {len(seqs)} sequences to process: {seqs}')

        im_filenames = []
        im_filenames_seq = []
        for seq in seqs:
            seq_dir = osp.join(opt.get('dataset_path'), scene, data_type, seq)
            process_imgs_dir = osp.join(seq_dir, 'stereo', 'centre_processed')
            if not osp.exists(process_imgs_dir):
                os.makedirs(process_imgs_dir)

            ts_filename = osp.join(seq_dir, 'stereo.timestamps')
            with open(ts_filename, 'r') as f:
                ts = [l.rstrip().split(' ')[0] for l in f]
            im_filenames.extend([osp.join(process_imgs_dir, '{:s}.png'.format(t)) for t in ts])
            im_filenames_seq.extend([seq for t in ts])
        assert len(dset) == len(im_filenames)

        # Main processing loop
        cols = ['scene','split','seq','img_path','t1','t2','t3','q1','q2','q3','q4']
        df = pd.DataFrame(np.zeros((len(im_filenames), len(cols))), columns=cols)
        for batch_idx, minibatch in enumerate(dataloader):
            imgs = minibatch.get('img')
            poses = minibatch.get('pose')
            for idx, im in enumerate(imgs):
                img_idx = batch_idx * opt.get('batchsize') + idx
                im_filename = im_filenames[img_idx]
                seq = im_filenames_seq[img_idx]
                im = Image.fromarray(im.numpy())
                try:
                    im.save(im_filename)
                except IOError:
                    print('IOError while saving {:s}'.format(im_filename))
                relative_im_filename = osp.join(scene, data_type, seq, 'stereo', 'centre_processed', osp.basename(im_filename))
                df.iloc[img_idx, :] = [scene, data_type, seq, relative_im_filename] + poses[idx].tolist()

            if batch_idx % 10 == 0:
                print('Processed {:d} / {:d}'.format(batch_idx * opt.get('batchsize'), len(dset)))

        # Saving output .csv file
        df.to_csv(opt.get('output_csv_filename'))
