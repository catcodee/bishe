from data_manager import Mars, DukeMTMC
from torchvision import transforms as T
from data_manager import Mars, DukeMTMC
from torchvision import transforms as T
from model import FeatureExtractor

class GlobalVar(object):
    def __init__(self):

        self._tgt_batch_size = 8
        self.seq_len = 4
        self.batch_size_sum = 64
        self.num_parts = 6
        self.gpu_ids = [3]
        self.feature_extractor = FeatureExtractor(self.num_parts)
        if len(self.gpu_ids > 0):
            self.use_gpu = True
            torch.cuda.set_device(self.gpu_ids)
            self.feature_extractor = nn.DataParallel(
                self.feature_extractor, self.gpu_ids)
        else:
            self.use_gpu = False

        self.save_path = './Logs'
        self.src_data = DukeMTMC()
        self.tgt_data = Mars()

        self.train_transfrom = T.Compose(
            [
                T.Resize((256, 128)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]
        )

        self.test_transform = T.Compose(
            [
                T.Resize((256, 128)),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
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
