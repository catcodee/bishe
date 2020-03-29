from data_loader import ImgDataset,TgtImgDataset, VideoDataset
from sampler import RandomIdentityBatchSampler,RandomTrackletBatchSampler,RandomIdentitySampler
from utils import GlobalVar
import torch
from torch.utils.data import DataLoader
from models import FeatureExtractor,SrcReidModel,TgtPartModel

def TrainStage1(object):
    def __init__(self, glob_var, opt):

        self.glob_var = glob_var
        self.save_name = '_stage1_'

        # 数据加载参数
        self.reid_train_data = self.glob_var.src_data.train
        self.src_batchsize = 64
        self.tgt_batchsize = 64
        self.num_workers = 4
        self.reid_sampler = RandomIdentitySampler(
            glob_var.src_data.train, glob_var.seq_len)

        # 模型参数
        self.feature_extractor = self.glob_var.feature_extractor
        self.num_classes = len(set(self.reid_train_data['pid']))
        self.num_parts = self.feature_extractor.num_parts

        # 优化器参数
        self.src_extract_lr = 0.005
        self.src_model_lr = 0.05
        self.src_weight_decay = 5e-04
        self.src_step_size = 40
        self.src_gamma = 0.1
        self.tgt_extract_lr = 0.005
        self.tgt_model_lr = 0.05
        self.tgt_weight_decay = 5e-04
        self.tgt_step_size = 40
        self.tgt_gamma = 0.1
        self.src_optim = optim.Adam
        self.tgt_optim = optim.Adam

        # Loss函数
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.triplet_loss = TripletLoss()

        # 训练参数
        self.max_epoch = 200
        self.use_triplet_loss = True
        self.tgt_labels = [torch.IntTensor(
            [i]*self.tgt_batchsize) for i in range(self.num_parts)]

        if self.glob_var.use_gpu:
            for i in range(len(self.tgt_labels)):
                self.tgt_labels[i] = self.tgt_labels[i].cuda()

    def get_dataloader(self):

        self.src_loader = DataLoader(ImgDataset(self.reid_train_data, self.glob_var.train_transform),
                                     batch_size=self.src_datasize,
                                     shuffle=True,
                                     num_workers=self.num_workers,
                                     drop_last=True,
                                     pin_memory=False,
                                     sampler=self.reid_sampler)

        self.tgt_loader = DataLoader(VideoDataset(self.glob_var.tgt_data.train, self.glob_var.train_transform, self.glob_var.seq_len),
                                     batch_size=self.tgt_batchsize//self.glob_var.seq_len,
                                     shuffle=True,
                                     num_workers=self.num_workers,
                                     drop_last=True
                                     pin_memory=False)

    def get_model(self):

        self.src_model = SrcReidModel(self.num_classes)
        self.tgt_model = TgtPartModel(self.num_parts)

        if self.glob_var.use_gpu:
            self.src_model = nn.DataParallel(
                self.src_model, self.glob_var.gpu_ids)
            self.tgt_model = nn.DataParallel(
                self.tgt_model, self.glob_var.gpu_ids)

    def get_optimizer(self):

        self.src_optimizer = self.src_optim(
            [{'params': self.feature_extractor.parameters(), 'lr': self.src_extract_lr},
             {'params': self.src_model.parameters(), 'lr': self.src_model_lr}],
            weight_decay=self.src_weight_decay
        )

        self.tgt_optimizer = self.tgt_optim(
            [{'params': self.feature_extractor.parameters(), 'lr': self.src_extract_lr},
             {'params': self.tgt_model.parameters(), 'lr': self.tgt_model_lr}],
            weight_decay=self.tgt_weight_decay
        )

        self.src_lr_scheduler = lr_scheduler(
            self.src_optimizer, step_size=self.src_step_size, gamma=self.src_gamma)
        self.tgt_lr_scheduler = lr_scheduler(
            self.tgt_optimizer, step_size=self.tgt_step_size, gamma=self.tgt_gamma)

    def train_tgt_one_batch(self, batch):
        imgs = batch[0]
        if self.glob_var.use_gpu:
            imgs = imgs.cuda()
        features = self.feature_extractor(imgs)
        predicts = self.tgt_model(features)

        loss = self.cross_entropy_loss(predicts[0], self.tgt_labels[0])
        for i in range(len(predicts)-1):
            loss += self.cross_entropy_loss(predicts[i+1],
                                            self.tgt_labels[i+1])

        tgt_optimizer.zero_grad()
        loss.backward()
        tgt_optimizer.step()
        return loss

    def train_src_one_batch(self, batch):
        imgs = batch[0]
        labels = batch[1]
        if self.glob_var.use_gpu:
            imgs = imgs.cuda()
            labels = labels.cuda()
        features = self.feature_extractor(imgs)
        predicts = self.src_model(features)

        cross_loss = self.cross_entropy_loss(predicts[0], labels)
        for i in range(len(predicts) - 1):
            cross_loss += self.cross_entropy_loss(predicts[i+1], labels)
        loss = cross_loss

        if self.use_triplet_loss:
            tri_loss = self.triplet_loss(featuers[:, :, 0], labels)
            loss += tri_loss
        src_optimizer.zero_grad()
        loss.backward()
        src_optimizer.step()
        return loss

    def _init_train(self):
        self.get_dataloader()
        self.get_model()
        self.get_optimizer()

    def train(self):
        self._init_train()
        for epoch in range(self.max_epoch):
            for src_batch in self.src_loader:
                src_loss = self.train_src_one_batch(src_batch)


