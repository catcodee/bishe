import os
import os.path as osp
import pandas as pd
import numpy as np 
from scipy.io import loadmat

class Mars(object):
    """
    MARS

    Reference:
    Zheng et al. MARS: A Video Benchmark for Large-Scale Person Re-identification. ECCV 2016.
    
    Dataset statistics:
    # identities: 1261
    # tracklets: 8298 (train) + 1980 (query) + 9330 (gallery)
    # cameras: 6

    Args:
        min_seq_len (int): tracklet with length shorter than this value will be discarded (default: 0).
    """
    #root = '/data/MARS'
    root = 'F:/Dataset/Mars'
    train_name_path = osp.join(root, 'info/train_name.txt')
    test_name_path = osp.join(root, 'info/test_name.txt')
    track_train_info_path = osp.join(root, 'info/tracks_train_info.mat')
    track_test_info_path = osp.join(root, 'info/tracks_test_info.mat')
    query_IDX_path = osp.join(root, 'info/query_IDX.mat')

    def __init__(self, min_seq_len=0):
        self._check_before_run()

        # prepare meta data
        train_names = self._get_names(self.train_name_path)
        test_names = self._get_names(self.test_name_path)
        track_train = loadmat(self.track_train_info_path)[
            'track_train_info']  # numpy.ndarray (8298, 4)
        track_test = loadmat(self.track_test_info_path)[
            'track_test_info']  # numpy.ndarray (12180, 4)
        query_IDX = loadmat(self.query_IDX_path)[
            'query_IDX'].squeeze()  # numpy.ndarray (1980,)
        query_IDX -= 1  # index from 0
        track_query = track_test[query_IDX, :]
        gallery_IDX = [i for i in range(
            track_test.shape[0]) if i not in query_IDX]
        track_gallery = track_test[gallery_IDX, :]

        train = self._process_data(
            train_names, track_train, home_dir='bbox_train', relabel=True, min_seq_len=min_seq_len)

        query = self._process_data(
            test_names, track_query, home_dir='bbox_test', relabel=False, min_seq_len=min_seq_len)

        gallery = self._process_data(
            test_names, track_gallery, home_dir='bbox_test', relabel=False, min_seq_len=min_seq_len)

        self.train = pd.DataFrame(
            train, columns=['path', 'pid', 'camid', 'tracklet_idx'])
        #self.train = np.array(train)
        self.query = pd.DataFrame(
            query, columns=['path', 'pid', 'camid', 'tracklet_idx'])
        self.gallery = pd.DataFrame(
            gallery, columns=['path', 'pid', 'camid', 'tracklet_idx'])

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.train_name_path):
            raise RuntimeError(
                "'{}' is not available".format(self.train_name_path))
        if not osp.exists(self.test_name_path):
            raise RuntimeError(
                "'{}' is not available".format(self.test_name_path))
        if not osp.exists(self.track_train_info_path):
            raise RuntimeError("'{}' is not available".format(
                self.track_train_info_path))
        if not osp.exists(self.track_test_info_path):
            raise RuntimeError("'{}' is not available".format(
                self.track_test_info_path))
        if not osp.exists(self.query_IDX_path):
            raise RuntimeError(
                "'{}' is not available".format(self.query_IDX_path))

    def _get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names

    def _process_data(self, names, meta_data, home_dir=None, relabel=False, min_seq_len=0):
        assert home_dir in ['bbox_train', 'bbox_test']
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:, 2].tolist()))
        num_pids = len(pid_list)

        if relabel:
            pid2label = {pid: label for label, pid in enumerate(pid_list)}
        tracklets = []
        num_imgs_per_tracklet = []

        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx, ...]
            start_index, end_index, pid, camid = data
            if pid == -1:
                continue  # junk images are just ignored
            assert 1 <= camid <= 6
            if relabel:
                pid = pid2label[pid]
            camid -= 1  # index starts from 0
            img_names = names[start_index-1:end_index]

            # make sure image names correspond to the same person
            pnames = [img_name[:4] for img_name in img_names]
            assert len(
                set(pnames)) == 1, "Error: a single tracklet contains different person images"

            # make sure all images are captured under the same camera
            camnames = [img_name[5] for img_name in img_names]
            assert len(
                set(camnames)) == 1, "Error: images are captured under different cameras!"

            # append image names with directory information
            img_paths = [osp.join(self.root, home_dir, img_name[:4], img_name)
                         for img_name in img_names]

            for i in range(len(img_paths)):
                if home_dir == 'bbox_train':
                    pid = -2
                else:
                    pid = int(img_names[i][0:4])
                tracklets.append(
                    (img_paths[i], int(pid), int(camid), int(tracklet_idx)))

        return tracklets


class DukeMTMC(object):

    #root = '/data/DukeMTMC-reID'
    root ='F:/Dataset/DukeMTMC-reID/DukeMTMC-reID'
    train_name_path = 'bounding_box_train'
    test_name_path = 'bounding_box_test'
    query_name_path = 'query'

    def __init__(self):

        train_names = os.listdir(osp.join(self.root, self.train_name_path))
        test_names = os.listdir(osp.join(self.root, self.test_name_path))
        query_names = os.listdir(osp.join(self.root, self.query_name_path))

        self.train = self._process_data(
            train_names, self.train_name_path, True)
        self.gallery = self._process_data(
            test_names, self.test_name_path, False)
        self.query = self._process_data(
            query_names, self.query_name_path, False)

        self.train_all = self._get_train_all(True)

    def _get_train_all(self, relabel):
        train_all = pd.concat(
            [self.train, self.gallery, self.query], ignore_index=True)

        pids = [int(path.replace('\\', '/').split('/')[-1][:4])
                for path in train_all['path']]
        pids = list(set(pids))

        if relabel:
            pid2label = {pid: label for label, pid in enumerate(pids)}

        for i in range(len(train_all)):
            pid = int(train_all['path'][i].replace('\\','/').split('/')[-1][:4])
            if relabel:
                pid = pid2label[pid]
            train_all['pid'][i] = pid

        return train_all

    def _process_data(self, names, home_dir=None, relabel=False):

        pids = [int(name[:4]) for name in names]
        pids = list(set(pids))
        if relabel:
            pid2label = {pid: label for label, pid in enumerate(pids)}

        dataset = []

        for name in names:
            img_path = osp.join(self.root, home_dir, name)
            camid = int(name.split('_')[1][1:]) - 1
            pid = int(name[:4])
            if relabel:
                pid = pid2label[pid]
            dataset.append([img_path, pid, camid])

        return pd.DataFrame(dataset, columns=['path', 'pid', 'camid'])
