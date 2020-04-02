import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.nn import init
import torch

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        # For old pytorch, you may use kaiming_normal.
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return x, f
        else:
            x = self.classifier(x)
            return x


class FeatureExtractor(nn.Module):
    def __init__(self, num_parts,mode='PCB'):

        super(FeatureExtractor, self).__init__()
        self.num_parts = num_parts
        resnet50 = models.resnet50(pretrained=True)
        if mode == 'PCB':
            resnet50.layer4[0].downsample[0].stride = (1, 1)
            resnet50.layer4[0].conv2.stride = (1, 1)

        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.pcb_pool = nn.AdaptiveMaxPool2d((num_parts, 1))
        self.half_pool = nn.AdaptiveMaxPool2d((2, 1))
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.mode = mode

    def forward(self, x):

        x = self.base(x)

        glob = self.global_pool(x)
        glob = glob.view(glob.size(0), glob.size(1), glob.size(2))

        if self.num_parts > 0 and (mode in ['PCB','PAP']:


            stripes = self.pcb_pool(x)
            stripes = stripes.view(stripes.size(0), stripes.size(1), stripes.size(2))

            features = torch.cat([glob, halfs, stripes], dim=2)

        else:
            features = glob

        return features


class SrcReidModel(nn.Module):
    def __init__(self, num_parts, num_classes,mode='PCB'):
        super(SrcReidModel, self).__init__()
        self.num_parts = num_parts
        self.classifier1 = ClassBlock(
            2048, class_num=num_classes, num_bottleneck=512, droprate=-1)
        if self.num_parts > 0:
            if 
            for i in range(self.num_parts+2):
                setattr(self, 'classifier'+str(i+2), 
                        ClassBlock(2048,class_num=num_classes, num_bottleneck=256, droprate=-1))

    def forward(self, x):

        y = []
        for i in range(x.size(2)):
            y.append(getattr(self, 'classifier'+str(i+1))(x[:, :, i]))

        return y


class TgtPartModel(nn.Module):
    def __init__(self, num_parts):
        super(TgtPartModel, self).__init__()

        self.num_parts = num_parts

        if self.num_parts > 0:
            for i in range(self.num_parts):
                setattr(self, 'classifier'+str(i+1), 
                        ClassBlock(2048,class_num=self.num_parts, num_bottleneck=256, droprate=0.5))

    def forward(self, x):

        y = []
        for i in range(3, x.size(2)):
            y.append(getattr(self, 'classifier'+str(i-2))(x[:, :, i]))

        return y


class ModifiedTripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    
    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.
    
    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0.3):
        super(ModifiedTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, src_input, tgt_input, tracklet_label):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        if not isinstance(tracklet_label, torch.Tensor):
            tracklet_label = torch.Tensor(tracklet_label)
        if len(tracklet_label.shape) > 1:
            tracklet_label = torch.flatten(tracklet_label)
        m = src_input.size(0)
        n = tgt_input.size(0)

        tgt_distmat = torch.pow(tgt_input, 2).sum(dim=1, keepdim=True).expand(n, n) + \
            torch.pow(tgt_input, 2).sum(dim=1, keepdim=True).expand(n, n).t()
        tgt_distmat.addmm_(1, -2, tgt_input, tgt_input.t())
        tgt_distmat = tgt_distmat.clamp(min=1e-12).sqrt()
        tgt_src_dismat = torch.pow(tgt_input, 2).sum(dim=1, keepdim=True).expand(n, m) + \
            torch.pow(src_input, 2).sum(dim=1, keepdim=True).expand(m, n).t()
        tgt_src_dismat.addmm_(1, -2, tgt_input, src_input.t())
        tgt_src_dismat = tgt_src_dismat.clamp(min=1e-12).sqrt()

        mask = tracklet_label.expand(n, n).eq(tracklet_label.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(tgt_distmat[i][mask[i]].max().unsqueeze(0))
            dist_an.append(tgt_src_dismat[i].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    
    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.
    
    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)
