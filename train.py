from data_loader import ImgDataset, TgtImgDataset, VideoDataset,tgt_collate_fn
from sampler import RandomIdentityBatchSampler, RandomTrackletBatchSampler, RandomIdentitySampler
from utils import GlobalVar, AverageMeter
import torch
from torch.utils.data import DataLoader
from models import FeatureExtractor, SrcReidModel, TgtPartModel, TripletLoss, ModifiedTripletLoss
from test import test, extract_feature
from torch import optim
from torch import nn
from torch.optim import lr_scheduler
import shutil
import os.path as osp
from tqdm import tqdm
import time

class TrainStage1(object):
    def __init__(self, glob_var):

        self.glob_var = glob_var
        self.save_resume_path = osp.join(self.glob_var.save_path,'stage1_resume.pth.tar')
        self.save_best_path = osp.join(self.glob_var.save_path, 'stage1_best.pth.tar')
        self.logs = glob_var.logs
        self.log_name = 'stage1/'+str(time.localtime())+'/'

        # 数据加载参数
        self.reid_train_data = self.glob_var.src_data.train
        self.src_batchsize = 32
        self.tgt_batchsize = 32
        self.num_workers = 4
        self.reid_sampler = RandomIdentitySampler(
            glob_var.src_data.train, glob_var.seq_len)

        # 模型参数
        self.feature_extractor = self.glob_var.feature_extractor
        self.num_classes = len(set(self.reid_train_data['pid']))
        self.num_parts = self.glob_var.num_parts

        # 优化器参数
        self.src_extract_lr = 0.01
        self.src_model_lr = 0.02
        self.src_weight_decay = 5e-04
        self.src_step_size = 400
        self.src_gamma = 0.1
        self.tgt_extract_lr = 0.0005
        self.tgt_model_lr = 0.005
        self.tgt_weight_decay = 5e-05
        self.tgt_step_size = 100
        self.tgt_gamma = 0.1
        self.src_optim = optim.SGD
        self.tgt_optim = optim.SGD

        # Loss函数
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.triplet_loss = TripletLoss()

        # 训练参数
        self.max_epoch = 400
        self.use_triplet_loss = False
        self.src_losses = AverageMeter()
        self.tgt_losses = AverageMeter()
        self.tgt_train_freq = 2
        self.tgt_start_epoch = 500
        self.src_test_freq = 20

        self.tgt_labels = [torch.LongTensor(
            [i]*self.tgt_batchsize) for i in range(self.num_parts)]

        if self.glob_var.use_gpu:
            for i in range(len(self.tgt_labels)):
                self.tgt_labels[i] = self.tgt_labels[i].cuda(3)

        #记录及保存参数

    def get_dataloader(self):

        self.src_loader = DataLoader(ImgDataset(self.reid_train_data, self.glob_var.train_transform),
                                     batch_size=self.src_batchsize,
                                     num_workers=self.num_workers,
                                     drop_last=True,
                                     pin_memory=False,
                                     sampler=self.reid_sampler)

        self.tgt_loader = DataLoader(VideoDataset(self.glob_var.tgt_data.train, transform=self.glob_var.train_transform, 
                                                  seq_len=self.glob_var.seq_len),
                                     batch_size=self.tgt_batchsize//self.glob_var.seq_len,
                                     shuffle=True,
                                     num_workers=self.num_workers,
                                     drop_last=True,
                                     pin_memory=False,
                                     collate_fn=tgt_collate_fn)

        self.src_query_loader = DataLoader(ImgDataset(self.glob_var.src_data.query, self.glob_var.test_transform),
                                           batch_size=self.src_batchsize,
                                           num_workers=self.num_workers,
                                           shuffle=False,
                                           drop_last=False,
                                           pin_memory=False
                                           )

        self.src_gallery_loader = DataLoader(ImgDataset(self.glob_var.src_data.gallery, self.glob_var.test_transform),
                                           batch_size=self.src_batchsize,
                                           num_workers=self.num_workers,
                                           shuffle=False,
                                           drop_last=False,
                                           pin_memory=False
                                           )

    def get_model(self):

        self.src_model = SrcReidModel(self.num_parts, self.num_classes)
        self.tgt_model = TgtPartModel(self.num_parts)

        if self.glob_var.use_gpu:
            self.src_model = self.src_model.cuda(3)
            self.tgt_model = self.tgt_model.cuda(3)

    def get_optimizer(self):

        self.src_optimizer = self.src_optim(
            [{'params': self.feature_extractor.parameters(), 'lr': self.src_extract_lr},
             {'params': self.src_model.parameters(), 'lr': self.src_model_lr}],
            weight_decay=self.src_weight_decay,
            momentum=0.9,
            nesterov=False
        )

        self.tgt_optimizer = self.tgt_optim(
            [{'params': self.feature_extractor.parameters(), 'lr': self.src_extract_lr},
             {'params': self.tgt_model.parameters(), 'lr': self.tgt_model_lr}],
            weight_decay=self.tgt_weight_decay,
            momentum=0.9,
            nesterov=False
        )

        self.src_lr_scheduler = lr_scheduler.StepLR(
            self.src_optimizer, step_size=self.src_step_size, gamma=self.src_gamma)
        self.tgt_lr_scheduler = lr_scheduler.StepLR(
            self.tgt_optimizer, step_size=self.tgt_step_size, gamma=self.tgt_gamma)

    def train_tgt_one_batch(self, batch):
        imgs = batch[0]
        if self.glob_var.use_gpu:
            imgs = imgs.cuda(3)
        features = self.feature_extractor(imgs)
        predicts = self.tgt_model(features)

        loss = self.cross_entropy_loss(predicts[0], self.tgt_labels[0])
        for i in range(len(predicts)-1):
            loss += self.cross_entropy_loss(predicts[i+1],
                                            self.tgt_labels[i+1])

        self.tgt_optimizer.zero_grad()
        loss.backward()
        self.tgt_optimizer.step()
        return loss.data

    def train_src_one_batch(self, batch):
        imgs = batch[0]
        labels = batch[1]
        if self.glob_var.use_gpu:
            imgs = imgs.cuda(3)
            labels = labels.cuda(3)
        features = self.feature_extractor(imgs)
        predicts = self.src_model(features)

        cross_loss = self.cross_entropy_loss(predicts[0], labels)
        for i in range(len(predicts) - 1):
            cross_loss += self.cross_entropy_loss(predicts[i+1], labels)
        loss = cross_loss
        '''
        if self.use_triplet_loss:
            tri_loss = self.triplet_loss(features[:, :, 0], labels)
            loss += tri_loss
        '''
        self.src_optimizer.zero_grad()
        loss.backward()
        self.src_optimizer.step()
        return loss.data

    def _init_train(self):
        self.get_dataloader()
        self.get_model()
        self.get_optimizer()

    def train_one_epoch(self,epoch):
        for src_batch in self.src_loader:
            src_loss = self.train_src_one_batch(src_batch)
            self.src_losses.update(src_loss)

            if epoch > self.tgt_start_epoch and epoch % self.tgt_train_freq == 0:
                try:
                    tgt_loss = self.train_tgt_one_batch(
                        next(tgt_loader_iter))
                    self.tgt_losses.update(tgt_loss)

                except StopIteration:
                    tgt_loader_iter = iter(self.tgt_loader)
            

    def train(self):
        self._init_train()
        print('------------complete init--------------')
        best_mAP = 0
        tgt_loader_iter = iter(self.tgt_loader)
        for epoch in tqdm(range(self.max_epoch)):
            self.train_one_epoch(epoch)

            self.logs.add_scalar(self.log_name+'src_loss',self.src_losses.avg, epoch)
            self.logs.add_scalar(self.log_name+'tgt_loss',self.tgt_losses.avg, epoch)

            if epoch % self.src_test_freq == 0:

                print('------start test------')

                cmc_all, mAP = test(self.src_query_loader, 
                                    self.src_gallery_loader, 
                                    self.feature_extractor,
                                    use_gpu=self.glob_var.use_gpu)
                print(cmc_all)
                print(mAP)
                self.logs.add_scalar(self.log_name+'test_mAP', mAP,epoch//self.src_test_freq)

                torch.save({'feature_extractor':self.feature_extractor.state_dict(),
                    'src_model':self.src_model.state_dict(),
                    'tgt_model':self.tgt_model.state_dict()},self.save_resume_path)
                if mAP > best_mAP:
                    shutil.copy(self.save_resume_path,self.save_best_path)
                    best_mAP = mAP


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    glob_var = GlobalVar()
    train_stage1 = TrainStage1(glob_var)
    train_stage1.train()
