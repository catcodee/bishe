
class GlobalVar(object):
    def __init__(self):

        self.epoch_now = 0
        self.src_reid_epoch_max = 200
        self.tgt_part_batch_idx = 0
        self.batch_size_sum = 64
        self._tgt_batch_size = 8
        self.seq_len = 4

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

    def set_global_var(self,name,value):
        setattr(self,name,value)