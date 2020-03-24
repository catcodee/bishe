from data_loader import ImgDataset,TgtImgDataset, VideoDataset
from sampler import RandomIdentityBatchSampler,RandomTrackletBatchSampler
from data_manager import Mars,DukeMTMC
from utils import GlobalVar
import torch
from torchvision import transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

if __name__ == '__main__':

    glob_var = GlobalVar()

    mars = Mars()
    dukemtmc = DukeMTMC()


    train_transform = T.Compose(
        [
            T.Resize((256,128)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ]
    )

    tgt_train_dataset = VideoDataset(mars.train,tgt_transform,4)
    tgt_train_loader = DataLoader(tgt_train_dataset, \
                       batch_sampler = RandomTrackletBatchSampler(dataset=mars.train,glob_var=glob_var),
                       num_workers = 0,
                       pin_memory = False)
    
    src_train_dataset = ImgDataset(dukemtmc.train,train_transform)
    src_reid_loader = DataLoader(dataset = src_train_dataset, batch_size = 4,shuffle=True,num_workers=0,drop_last=False)
    for batch in src_reid_loader:
        print(batch)
        break

