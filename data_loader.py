from PIL import Image
from torch.utils.data import DataLoader,Dataset
import numpy as np
import torch


def tgt_collate_fn(batch):
    imgs = []
    pids = []
    camids = []
    tracklets = []
    paths = []
    for img, pid, camid, path, tracklet in batch:
        imgs += [img.unsqueeze(dim=0)]
        pids += pid
        camids += camid
        tracklets += tracklet
        paths += path
    imgs = torch.cat(imgs, dim=0)
    b, t, c, h, w = imgs.shape
    imgs = imgs.view(b*t, c, h, w)
    return imgs, torch.IntTensor(pids), torch.IntTensor(camids), paths, torch.IntTensor(tracklets)
    
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

        return img, label, camid, path, tracklet_idx

    def __len__(self):
        return len(self.samples)


class VideoDataset(Dataset):

    def __init__(self, samples, sampler = 'random',transform = None, seq_len = 4):

        self.samples = samples
        self.transform = transform
        self.seq_len = seq_len
        self.sampler = sampler

    def __getitem__(self, index):

        frames = self.samples[self.samples['tracklet_idx'] == index]
        num = len(frames)
        if self.sampler == 'random':
            if num < self.seq_len:
                indices = list(range(num)) + [num-1]*(self.seq_len-num)
            else:
                start = np.random.randint(num-self.seq_len)
                end = start + self.seq_len
                indices = list(range(num))[start:end]

            imgs = []
            pids = [frames['pid'].iloc[0]]*self.seq_len
            camids = [frames['camid'].iloc[0]]*self.seq_len
            tracklet_idxs = [index]*self.seq_len
            paths = []

            for index in indices:
                img_path = frames['path'].iloc[index]
                img = read_image(img_path)
                if self.transform is not None:
                    img = self.transform(img)
                img = img.unsqueeze(0)
                imgs.append(img)
                paths.append(img_path)
            imgs = torch.cat(imgs, dim=0)
            
        elif self.sampler == 'dense':
            if num < self.seq_len:
                indices = list(range(num)) + [num-1]*(self.seq_len-num)
            else:

                length = num - num%self.seq_len
                indices = list(range(num))[0:length]
            imgs = []
            pids = [frames['pid'].iloc[0]]*length
            camids = [frames['camid'].iloc[0]]*length
            tracklet_idxs = [index]*end
            paths = []

            for index in indices:
                img_path = frames['path'].iloc[index]
                img = read_image(img_path)
                if self.transform is not None:
                    img = self.transform(img)
                img = img.unsqueeze(0)
                imgs.append(img)
                paths.append(img_path)
            imgs = torch.cat(imgs, dim=0)

        return imgs, pids, camids, paths,tracklet_idxs
    def __len__(self):
        return len(set(self.samples['tracklet_idx']))
