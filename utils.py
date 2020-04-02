import torch.backends.cudnn as cudnn
from data_manager import Mars, DukeMTMC
import transforms as T
from models import FeatureExtractor
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch
import os

class AverageMeter(object):
    """Computes and stores the average and current value.
       
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
class GlobalVar(object):
    def __init__(self):

        self._tgt_batch_size = 8
        self.seq_len = 4
        self.batch_size_sum = 64
        self.num_parts = 6
        self.gpu_ids = '3'
        self.feature_extractor = FeatureExtractor(self.num_parts)
        if len(self.gpu_ids) > 0:
            self.use_gpu = True

            cudnn.benchmark = True
            self.feature_extractor = self.feature_extractor.cuda(3)
        else:
            self.use_gpu = False

        self.save_path = './save'
        self.logs_dir = './logs'
        self.logs = SummaryWriter(self.logs_dir)

        self.src_data = DukeMTMC()
        self.tgt_data = Mars()

        transform = T.Compose([
            T.Resize([256, 128]),
            T.RandomHorizontalFlip(p=0.5),
            T.Pad(10),
            T.RandomCrop([256, 128]),
            T.ToTensor(),
            normalize_transform,
            RandomErasing(probability=0.5)
        ])

        self.test_transform = T.Compose(
            [
                T.Resize((384, 128)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ]
        )


    @property
    def tgt_batch_size(self):
        return self._tgt_batch_size

    @tgt_batch_size.setter
    def tgt_batch_size(self, value):

        if value*self.seq_len > self.batch_size_sum:
            self._tgt_batch_size = (
                self.batch_size_sum // self.seq_len)*self.seq_len
        elif value < 0:
            self._tgt_batch_size = 0
        else:
            self._tgt_batch_size = value

    @property
    def src_batch_size(self):
        return self.batch_size_sum - self._tgt_batch_size*self.seq_len

    def step(self):
        pass

    def set_global_var(self, name, value):
        setattr(self, name, value)
