from PIL import Image
from torch.utils.data import DataLoader,Dataset
import numpy as np
import torch


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(
                img_path))
            pass
    return img


class ImgDataset(Dataset):

    def __init__(self, samples, transform=None):

        self.samples = samples
        self.transform = transform

    def __getitem__(self, index):

        path = self.samples['path'][index]
        label = self.samples['pid'][index]
        camid = self.samples['camid'][index]

        img = read_image(path)
        if self.transform:
            img = self.transform(img)

        return img, label, camid, path 

    def __len__(self):
        return len(self.samples)


class TgtImgDataset(Dataset):

    def __init__(self, samples, transform=None):

        self.samples = samples
        self.transform = transform

    def __getitem__(self, index):

        path = self.samples['path'][index]
        label = self.samples['pid'][index]
        camid = self.samples['camid'][index]
        tracklet_idx = self.samples['tracklet_idx'][index]

        img = read_image(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, label, camid, tracklet_idx,path

    def __len__(self):
        return len(self.samples)


class VideoDataset(Dataset):

    def __init__(self, samples, transform, seq_len):

        self.samples = samples
        self.transform = transform
        self.seq_len = seq_len

    def __getitem__(self, index):

        frames = self.samples[self.samples['tracklet_idx'] == index]
        num = len(frames)
        if num < self.seq_len:
            indices = list(range(num)) + [num-1]*(self.seq_len-num)
        else:
            start = np.random.randint(num-self.seq_len)
            end = start + self.seq_len
            indices = list(range(num))[start:end]

        imgs = []
        pids = []
        camids = []
        tracklet_idxs = []
        paths = []

        for index in indices:
            img_path = frames['path'].iloc[index]
            img = read_image(img_path)
            if self.transform is not None:
                img = self.transform(img)
            img = img.unsqueeze(0)
            imgs.append(img)
            pids.append(frames['pid'].iloc[index])
            tracklet_idxs.append(index)
            paths.append(img_path)
        imgs = torch.cat(imgs, dim=0)
        return imgs, pids, camids, tracklet_idxs,paths
    def __len__(self):
        return len(set(self.samples['tracklet_idx']))
