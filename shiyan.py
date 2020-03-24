#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import os
import os.path as osp
import pandas as pd
import numpy as np 
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as T 

import torch.nn as nn
import torch.nn.functional as F 
from torchvision.datasets import ImageFolder
from torch.utils.data import Sampler
from scipy.io import loadmat
from PIL import Image


# In[2]:


train_transform = T.Compose(
    [
        T.Resize((256,128)),
        T.ToTensor(),
        T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ]
)
train_dataset = ImageFolder('/data/MARS/bbox_train',transform=train_transform)
train_loader = DataLoader(train_dataset,4,shuffle=True,pin_memory=False)


# In[ ]:


print(train_dataset[0])


# In[3]:


loader_iter = iter(train_loader)

a = loader_iter.next()

print(a)
train_loader.batch_size = 1

b = loader_iter.next()
print(b)


# In[19]:


class shiyan(object):
    def __init__(self):
        self.index = [1,2,3,4,5,6,7,8]
    def __iter__(self):
        self.num_call = 0 
        self.epoch = 0
        self.i = 0
        self.step = 1
        return self
    def __getitem__(self,i):
    
        return self.index[i]
    def __next__(self):
        
        ret = []
        self.num_call += 1
        for j in range(self.step):
            pos = (self.i + j) % len(self.index)
            ret.append(self.index[pos])
        self.i = (self.i + self.step)%len(self.index)
        self.step += 1
        if self.step > 10:
            raise StopIteration
        return ret
    
sp = shiyan()
num = 0 
for i in sp:
    print(i)
    


# In[21]:


b = range(2)
a = iter(b)

for i in range(10):
    try:
        print(next(a))
    except StopIteration:
        a = iter(b)


# In[23]:


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
    root = '/data/MARS'
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
        track_train = loadmat(self.track_train_info_path)['track_train_info'] # numpy.ndarray (8298, 4)
        track_test = loadmat(self.track_test_info_path)['track_test_info'] # numpy.ndarray (12180, 4)
        query_IDX = loadmat(self.query_IDX_path)['query_IDX'].squeeze() # numpy.ndarray (1980,)
        query_IDX -= 1 # index from 0
        track_query = track_test[query_IDX,:]
        gallery_IDX = [i for i in range(track_test.shape[0]) if i not in query_IDX]
        track_gallery = track_test[gallery_IDX,:]

        train =           self._process_data(train_names, track_train, home_dir='bbox_train', relabel=True, min_seq_len=min_seq_len)

        query =           self._process_data(test_names, track_query, home_dir='bbox_test', relabel=False, min_seq_len=min_seq_len)

        gallery =           self._process_data(test_names, track_gallery, home_dir='bbox_test', relabel=False, min_seq_len=min_seq_len)

        

        self.train = pd.DataFrame(train,columns = ['path','pid','camid','tracklet_idx'])
        #self.train = np.array(train)
        self.query = pd.DataFrame(query,columns = ['path','pid','camid','tracklet_idx'])
        self.gallery = pd.DataFrame(gallery,columns = ['path','pid','camid','tracklet_idx'])

        

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.train_name_path):
            raise RuntimeError("'{}' is not available".format(self.train_name_path))
        if not osp.exists(self.test_name_path):
            raise RuntimeError("'{}' is not available".format(self.test_name_path))
        if not osp.exists(self.track_train_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_train_info_path))
        if not osp.exists(self.track_test_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_test_info_path))
        if not osp.exists(self.query_IDX_path):
            raise RuntimeError("'{}' is not available".format(self.query_IDX_path))

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
        pid_list = list(set(meta_data[:,2].tolist()))
        num_pids = len(pid_list)

        if relabel: pid2label = {pid:label for label, pid in enumerate(pid_list)}
        tracklets = []
        num_imgs_per_tracklet = []

        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx,...]
            start_index, end_index, pid, camid = data
            if pid == -1: continue # junk images are just ignored
            assert 1 <= camid <= 6
            if relabel: pid = pid2label[pid]
            camid -= 1 # index starts from 0
            img_names = names[start_index-1:end_index]
            
            # make sure image names correspond to the same person
            pnames = [img_name[:4] for img_name in img_names]
            assert len(set(pnames)) == 1, "Error: a single tracklet contains different person images"

            # make sure all images are captured under the same camera
            camnames = [img_name[5] for img_name in img_names]
            assert len(set(camnames)) == 1, "Error: images are captured under different cameras!"

            # append image names with directory information
            img_paths = [osp.join(self.root, home_dir, img_name[:4], img_name) for img_name in img_names]
            
            for i in range(len(img_paths)):
                if home_dir == 'bbox_train':
                    pid = -2
                else:
                    pid = int(img_names[i][0:4])
                tracklets.append((img_paths[i],int(pid),int(camid),int(tracklet_idx)))
                
        return tracklets

    
    


# In[24]:


mars = Mars()
train = mars.train


# In[43]:


print(train.loc[0])


# In[ ]:


class DukeMTMC(object):
    
    root = '/data/DukeMTMC-reID'
    train_name_path = 'bounding_box_train'
    test_name_path = 'bounding_box_test'
    query_name_path = 'query'
    
    def __init__(self):
        
        train_names = os.listdir(osp.join(self.root,self.train_name_path))
        test_names = os.listdir(osp.join(self.root,self.test_name_path))
        query_names = os.listdir(osp.join(self.root,self.query_name_path))
        
        self.train = self._process_data(train_names,self.train_name_path,True)
        self.gallery = self._process_data(test_names,self.test_name_path,False)
        self.query = self._process_data(query_names,self.query_name_path,False)
        
        self.train_all = self._get_train_all(True)
            
    def _get_train_all(self,relabel):
        train_all = pd.concat([self.train,self.gallery,self.query],ignore_index = True)
        
        pids = [int(path.split('/')[-1][:4]) for path in train_all['path']]
        pids = list(set(pids))

        if relabel:
            pid2label = {pid:label for label,pid in enumerate(pids)}
        
        for i in range(len(train_all)):
            pid = int(train_all['path'][i].split('/')[-1][:4])
            if relabel:
                pid = pid2label[pid]
            train_all['pid'][i] = pid
            
        return train_all
        
    def _process_data(self, names, home_dir=None, relabel=False):
        
        pids = [int(name[:4]) for name in names]
        pids = list(set(pids))
        if relabel: 
            pid2label = {pid:label for label, pid in enumerate(pids)}
        
        dataset = []
        
        for name in names:
            img_path = osp.join(self.root,home_dir,name)
            camid = int(name.split('_')[1][1:]) - 1
            pid = int(name[:4])
            if relabel:
                pid = pid2label[pid]
            dataset.append([img_path,pid,camid])
        
        return pd.DataFrame(dataset,columns = ['path','pid','camid'])
    


# In[22]:


duke = Duke()
train = duke.train


# In[ ]:


print(duke.train_all.iloc[0:14,1])


# In[ ]:


x = torch.randn(64,2048,1,1)
y = x.view(x.size(0),x.size(1))
print(y.shape)


# In[ ]:


dic = dict()
dic['f'] = torch.ones(10)
print(dic)
f = dic['f']
fc = nn.Linear(10,20)
dic['f'] = fc(f)
print(f)
print(dic)


# In[ ]:


import glob
a = glob.glob(osp.join('data/MARS/bbox_train','*.jpg'))
print(a[0])


# # 三元损失函数以图像为单位
# 
# # 聚类以一个视频序列为单位
# 
# # sampler在目标域以tracklet为单位，在源域以id为单位()
# 
# # 聚类时目标域
# 
# # dataset的源域图片索引，目标域tracklet索引
# 
# 

# In[ ]:


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

class ImgDataset(Dataset):
    
    def __init__(self, samples, transform = None):
        
        self.samples = samples
        self.transform = transform
        
    def __getitem__(self, index):
        
        path = self.samples['path'][index]
        label = self.samples['id'][index]
        camid = self.samples['camid'][index]
        
        img = read_image(path)
        if transform:
            img = transform(img)
        
        return img,label,camid
    def __len__(self):
        return len(self.samples)
    
class TgtImgDataset(Dataset):

    def __init__(self, samples, transform = None):
        
        self.samples = samples
        self.transform = transform
        
    def __getitem__(self, index):
        
        path = self.samples['path'][index]
        label = self.samples['id'][index]
        camid = self.samples['camid'][index]
        tracklet_idx = self.samples['tracklet_idx'][index]
        
        img = read_image(path)
        if transform is not None:
            img = transform(img)
        
        return img,label,camid,tracklet_idx

    def __len__(self):
        return len(self.samples)


class VideoDataset(Dataset):
    
    def __init__(self, samples, transform, seq_len, is_video):
        
        self.samples = samples
        self.transform = transform
        self.seq_len = seq_len
        self.is_video = seq
    
    def __getitem__(self, index):
        
        frames = samples[samples['tracklet_idx'] == index]
        num = len(frames)
        if num < seq_len:
            indices = list(range(num)) + [num-1]*(seq_len-num)
        else:
            indices = np.random.randint(num-seq_len ,size=seq_len)
        
        imgs = []
        pid = []
        camid = []
        tracklet_idx = []
        
        for index in indices:
            img_path = frames['path'].iloc[index]
            img = read_image(img_path)
            if self.transform is not None:
                img = self.transform(img)
            img = img.unsqueeze(0)
            imgs.append(img)
            imgs = torch.cat(imgs, dim=0)
            pid.append(frames['pid'])
            
        return imgs, pid, camid
    
    def __len__（self）:
        return len(set(self.samples['tracklet_idx']))


class GlobalVar(object):
    def __init__(self,opt):
        
        self.epoch_now = 0
        self.src_reid_epoch_max = 200
        self.tgt_part_batch_idx = 0
        self.batch_size_sum = 64
        self._tgt_batch_size = 32
        self.seq_len = 4
    
    @property
    def tgt_batch_size(self):
        return self._tgt_batch_size
    
    @tgt_batch_size.setter
    def tgt_batch_size(self, value):

        if value*self.seq_len > self.batch_size_sum:
            self._tgt_batch_size = (self.batch_size_sum// self.seq_len)*self.seq_len
        elif value < 0:
            self._tgt_batch_size = 0
        else:
            self._tgt_batch_size = value
        
    
    @property
    def src_batch_size(self):
        return self.batch_size_sum - self._tgt_batch_size*self.seq_len

        

    def step(self):
        pass
    
    def set_global_var(self):
        pass
    
    def get_global_var(self):
        pass

from collections import defaultdict
class RandomTrackletBatchSampler(Sampler):
    def __init__(self,dataset,glob_var):

        self.dataset = dataset 
        self.glob_var = glob_var
        self.index_dict = defaultdict(list)
        self.tracklets = list(set(self.dataset['tracklet_idx']))
        self.num_tracklets = len(tracklets)

    def __len__(self):
        return self.num_tracklets

    def __iter__(self):
        self.indices = np.random.randperm(len(self.tracklets)).numpy().tolist()
        self.index = 0
        return self

    def __next__(self):
        
        k = self.index + self.glob_var.tgt_batch_size
        if k >= self.num_tracklets:
            raise StopIteration
        self.index = k
        return self.indices[self.index:k]
        
        
# In[ ]:


class RandomIdentitySampler(Sampler):
    """Modified from https://github.com/Cysu/open-reid/blob/master/reid/utils/data/sampler.py"""
    def __init__(self, data_source, k=1):
        self.data_source = data_source
        self.k = k
        self.index_dic = defaultdict(list)
        for index in range(len(data_source)):
            self.index_dic[datasource.loc[index]['pid']].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_pids = len(self.pids)

    def __len__(self):
        return self.num_pids * self.k

    def __iter__(self):
        indices = torch.randperm(self.num_pids)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            if len(t) >= self.k:
                t = np.random.choice(t, size=self.k, replace=False)
            else:
                t = np.random.choice(t, size=self.k, replace=True)
            ret.extend(t)
        return iter(ret)


# In[ ]:


import sklearn
from sklearn.cluster import DBSCAN
help(DBSCAN)


# In[ ]:


help(Dataset)


# In[ ]:


help(DataLoader)


# In[ ]:
help(np.random.randperm)




# %%
