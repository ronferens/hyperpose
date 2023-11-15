import os.path as osp
from torch.utils import data
import numpy as np
from .robotcar_sdk.interpolate_poses import interpolate_vo_poses, interpolate_ins_poses
from .robotcar_sdk.camera_model import CameraModel
from .robotcar_sdk.image import load_image
from functools import partial
import transforms3d.quaternions as txq
import pickle


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

class RobotCar(data.Dataset):
    def __init__(self, scene, dataset_path, labels_path, train, transform=None,
                 target_transform=None, real=False, skip_images=False, seed=7,
                 undistort=False, vo_lib='stereo'):
        """
        :param scene: e.g. 'full' or 'loop'. collection of sequences.
        :param data_path: Root RobotCar data directory.
        Usually '../data/deepslam_data/RobotCar'
        :param train: flag for training / validation
        :param transform: Transform to be applied to images
        :param target_transform: Transform to be applied to poses
        :param real: if True, load poses from SLAM / integration of VO
        :param skip_images: return None images, only poses
        :param seed: random seed
        :param undistort: whether to undistort images (slow)
        :param vo_lib: Library to use for VO ('stereo' or 'gps')
        (`gps` is a misnomer in this code - it just loads the position information
        from GPS)
        """
        np.random.seed(seed)
        self.transform = transform
        self.target_transform = target_transform
        self.skip_images = skip_images
        self.undistort = undistort
        self.scene = scene
        labels_dir = osp.expanduser(osp.join(labels_path, self.scene))
        data_dir = osp.expanduser(osp.join(dataset_path, self.scene))

        # decide which sequences to use
        if train:
            split_filename = osp.join(labels_dir, 'train_split.txt')
            split_dir = 'train'
        else:
            split_filename = osp.join(labels_dir, 'test_split.txt')
            split_dir = 'test'
        with open(split_filename, 'r') as f:
            seqs = [l.rstrip() for l in f if not l.startswith('#')]

        ps = {}
        ts = {}
        vo_stats = {}
        self.imgs = []
        for seq in seqs:
            seq_labels_dir = osp.join(labels_dir, seq)
            seq_data_dir = osp.join(data_dir, split_dir, seq)

            # read the image timestamps
            ts_filename = osp.join(seq_data_dir, 'stereo.timestamps')
            with open(ts_filename, 'r') as f:
                ts[seq] = [int(l.rstrip().split(' ')[0]) for l in f]

            if real:  # poses from integration of VOs
                if vo_lib == 'stereo':
                    vo_filename = osp.join(seq_labels_dir, 'vo', 'vo.csv')
                    p = np.asarray(interpolate_vo_poses(vo_filename, ts[seq], ts[seq][0]))
                elif vo_lib == 'gps':
                    vo_filename = osp.join(seq_labels_dir, 'gps', 'gps_ins.csv')
                    p = np.asarray(interpolate_ins_poses(vo_filename, ts[seq], ts[seq][0]))
                else:
                    raise NotImplementedError
                vo_stats_filename = osp.join(seq_labels_dir, '{:s}_vo_stats.pkl'.format(vo_lib))
                # with open(vo_stats_filename, 'rb') as f:
                #     vo_stats[seq] = pickle.load(f)
                vo_stats[seq] = load_pickle(vo_stats_filename)
                ps[seq] = np.reshape(p[:, :3, :], (len(p), -1))
            else:  # GT poses
                pose_filename = osp.join(seq_labels_dir, 'gps', 'ins.csv')
                p = np.asarray(interpolate_ins_poses(pose_filename, ts[seq], ts[seq][0]))
                ps[seq] = np.reshape(p[:, :3, :], (len(p), -1))
                vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}

            self.imgs.extend([osp.join(seq_data_dir, 'stereo', 'centre', '{:d}.png'.format(t)) for t in ts[seq]])

        # read / save pose normalization information
        poses = np.empty((0, 12))
        for p in ps.values():
            poses = np.vstack((poses, p))
        pose_stats_filename = osp.join(data_dir, 'pose_stats.txt')
        if train and not real:
            mean_t = np.mean(poses[:, [3, 7, 11]], axis=0)
            std_t = np.std(poses[:, [3, 7, 11]], axis=0)
            np.savetxt(pose_stats_filename, np.vstack((mean_t, std_t)), fmt='%8.7f')
        else:
            mean_t, std_t = np.loadtxt(pose_stats_filename)

        # convert the pose to translation + log quaternion, align, normalize
        self.poses = np.empty((0, 7))
        for seq in seqs:
            pss = self.process_poses(poses_in=ps[seq], mean_t=mean_t, std_t=std_t,
                                     align_R=vo_stats[seq]['R'], align_t=vo_stats[seq]['t'],
                                     align_s=vo_stats[seq]['s'])
            self.poses = np.vstack((self.poses, pss))
        self.gt_idx = np.asarray(range(len(self.poses)))

        # camera model and image loader
        camera_model = CameraModel(osp.join(labels_path, 'robotcar_camera_models'), osp.join('stereo', 'centre'))
        self.im_loader = partial(load_image, model=camera_model)

    def __getitem__(self, index):
        if self.skip_images:
            img = None
            pose = self.poses[index]
        else:
            img = None
            while img is None:
                if self.undistort:
                    img = np.uint8(load_image(self.imgs[index]))
                else:
                    img = load_image(self.imgs[index])
                pose = np.float32(self.poses[index])
                index += 1
            index -= 1

        if self.target_transform is not None:
            pose = np.array(self.target_transform(pose))

        if self.skip_images:
            sample = {'img': img, 'pose': pose}
            return sample

        if self.transform is not None:
            img = self.transform(img)

        sample = {'img': img, 'pose': pose}
        return sample

    def __len__(self):
        return len(self.poses)

    @staticmethod
    def load_image(filename, loader):
        try:
            img = loader(filename)
        except IOError as e:
            print('Could not load image {:s}, IOError: {:s}'.format(filename, e))
            return None
        except:
            print('Could not load image {:s}, unexpected error'.format(filename))
            return None
        return img

    @staticmethod
    def qlog(q):
        if all(q[1:] == 0):
            q = np.zeros(3)
        else:
            q = np.arccos(q[0]) * q[1:] / np.linalg.norm(q[1:])
        return q

    def process_poses(self, poses_in, mean_t, std_t, align_R, align_t, align_s):
        poses_out = np.zeros((len(poses_in), 7))
        poses_out[:, 0:3] = poses_in[:, [3, 7, 11]]

        # align
        for i in range(len(poses_out)):
            R = poses_in[i].reshape((3, 4))[:3, :3]
            q = txq.mat2quat(np.dot(align_R, R))
            q *= np.sign(q[0])  # constrain to hemisphere
            # q = self.qlog(q)
            poses_out[i, 3:] = q
            t = poses_out[i, :3] - align_t
            poses_out[i, :3] = align_s * np.dot(align_R, t[:, np.newaxis]).squeeze()

        # normalize translation
        poses_out[:, :3] -= mean_t
        poses_out[:, :3] /= std_t
        return poses_out
