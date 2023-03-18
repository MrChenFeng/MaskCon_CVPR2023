import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet

dim_dict = {
    'resnet18': 512,
    'resnet34': 512,
    'resnet50': 2048,
}


class ModelBase(nn.Module):
    """
    For small size figures:
    (i) replaces conv1 with kernel=3, str=1
    (ii) removes pool1
    """

    def __init__(self, figsize=32, num_classes=10, projection_dim=128, arch=None):
        super(ModelBase, self).__init__()
        resnet_arch = getattr(resnet, arch)

        self.net = resnet_arch(pretrained=True)
        if figsize <= 64:  # adapt to small-size images
            self.net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.net.maxpool = nn.Identity()
        self.net.fc = nn.Identity()

        self.feat_dim = dim_dict[arch]
        self.projector = nn.Sequential(
            nn.Linear(self.feat_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, projection_dim)
        )

        self.classifer = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x, feat=False):
        x = self.net(x)
        if feat:
            return x
        else:
            cls, proj = self.classifer(x), self.projector(x)
            return cls, proj


"""### MaskCon backbone"""


class MaskCon(nn.Module):
    def __init__(self, num_classes_coarse=10,  dim=128, K=4096, m=0.9, T1=0.1, T2=0.1, arch='resnet18', mode='mixcon', size=32):
        '''
        Modifed based on MoCo framework.

        :param num_classes_coarse: num of coarse classes
        :param dim: dimension of feature projections
        :param K: size of memory bank
        :param m: momentum encoder
        :param T1: temperature of original contrastive loss
        :param T2: temperature for soft labels generation
        :param arch: architecture of encoder
        :param mode: method mode [maskcon, grafit or coins]
        :param size: dataset image size
        '''
        super(MaskCon, self).__init__()
        self.K = K
        self.m = m
        self.T1 = T1
        self.T2 = T2
        self.mode = mode
        # create the encoders
        self.encoder_q = ModelBase(figsize=size, num_classes=num_classes_coarse, projection_dim=dim, arch=arch)
        self.encoder_k = ModelBase(figsize=size, num_classes=num_classes_coarse, projection_dim=dim, arch=arch)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.num_classes_coarse = num_classes_coarse
        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer('coarse_labels', torch.randint(0, num_classes_coarse, [self.K]).long())
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, coarse_labels):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.t()  # transpose
        self.coarse_labels[ptr:ptr + batch_size] = coarse_labels

        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # random shuffle index
        idx_shuffle = torch.randperm(x.shape[0]).cuda()

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle]


    def initiate_memorybank(self, dataloader):
        print('Initiate memory bank!')
        num = 0
        iter_data = iter(dataloader)
        for i in range(self.K):  # update the memory bank with image representation
            if num == self.K:
                break
            # print(num)
            try:
                [im_k, _], coarse_label, _ = next(iter_data)
            except:
                iter_data = iter(dataloader)
                [im_k, _], coarse_label, _ = next(iter_data)
            num = num + len(im_k)
            im_k, coarse_label = im_k.cuda(non_blocking=True), coarse_label.cuda(non_blocking=True)
            im_k_, idx_unshufflek = self._batch_shuffle_single_gpu(im_k)
            _, k = self.encoder_k(im_k_)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)  # already normalized
            # undo shuffle
            k = self._batch_unshuffle_single_gpu(k, idx_unshufflek)
            self._dequeue_and_enqueue(k, coarse_label)

    def forward(self, im_k, im_q, coarse_label, args):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """
        # update the key encoder
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()
        cls_q, q = self.encoder_q(im_q)  # queries:
        q = nn.functional.normalize(q, dim=1)  # already normalized

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            im_k_, idx_unshufflek = self._batch_shuffle_single_gpu(im_k)
            _, k = self.encoder_k(im_k_)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)  # already normalized
            # undo shuffle
            k = self._batch_unshuffle_single_gpu(k, idx_unshufflek)

            # soft-labels
            coarse_z = torch.ones(len(q), self.K).cuda()
            new_label = coarse_label.reshape(-1, 1).repeat(1, self.K)
            memory_labels = self.coarse_labels.reshape(1, -1).repeat(len(q), 1)
            coarse_z = coarse_z * (new_label == memory_labels)
            logits_pd = torch.einsum('nc,ck->nk', [k, self.queue.clone().detach()])
            logits_pd /= self.T2
            logits_pd = logits_pd * coarse_z  # mask out non-same-coarse class samples
            logits_pd = logits_pd - logits_pd.max(dim=1, keepdim=True)[0]
            pseudo_soft_z = logits_pd.exp() * coarse_z
            pseudo_sum = torch.sum(pseudo_soft_z, dim=1, keepdim=True)
            maskcon_z = torch.zeros(len(q), self.K + 1).cuda()
            maskcon_z[:, 0] = 1
            tmp = pseudo_soft_z / pseudo_sum
            # rescale by maximum
            tmp = tmp / tmp.max(dim=1, keepdim=True)[0]
            maskcon_z[:, 1:] = tmp
            # generate weighted inter-sample relations
            maskcon_z = maskcon_z / maskcon_z.sum(dim=1, keepdim=True)

            # self-supervised inter-sample relations
            self_z = torch.zeros(len(q), self.K + 1).cuda()
            self_z[:, 0] = 1.0


            labels = args.w * maskcon_z + (1 - args.w) * self_z

        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # logits: Nx(1+K)
        logits_all = torch.cat([l_pos, l_neg], dim=1)
        logits_all /= self.T1

        loss = -torch.sum(F.log_softmax(logits_all, 1) * labels.detach(), 1).mean()
        # inside vs outside?
        self._dequeue_and_enqueue(k, coarse_label)

        return loss

    def forward_explicit(self, im_k, im_q, coarse_label, args):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """
        # update the key encoder
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()
        cls_q, q = self.encoder_q(im_q)  # queries:
        q = nn.functional.normalize(q, dim=1)  # already normalized

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            im_k_, idx_unshufflek = self._batch_shuffle_single_gpu(im_k)
            _, k = self.encoder_k(im_k_)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)  # already normalized
            # undo shuffle
            k = self._batch_unshuffle_single_gpu(k, idx_unshufflek)

            # supcon: coarse inter-sample relations
            coarse_z = torch.zeros(len(q), self.K + 1).cuda()
            coarse_z[:, 0] = 1.0
            tmp_z = torch.ones(len(q), self.K).cuda()
            new_label = coarse_label.reshape(-1, 1).repeat(1, self.K)
            memory_labels = self.coarse_labels.reshape(1, -1).repeat(len(q), 1)
            tmp_z = tmp_z * (new_label == memory_labels)
            coarse_z[:, 1:] = tmp_z

            # self-supervised inter-sample relations
            self_z = torch.zeros(len(q), self.K + 1).cuda()
            self_z[:, 0] = 1.0

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T1
        loss_selfcon = -torch.sum(F.log_softmax(logits, 1) * self_z.detach(), 1)
        loss_cls = torch.nn.functional.cross_entropy(cls_q, coarse_label, reduction='none')
        loss_supcon = -torch.sum(F.log_softmax(logits, 1) * (coarse_z / coarse_z.sum(dim=1, keepdim=True)).detach(), 1)

        logits_pd = torch.einsum('nc,ck->nk', [k, self.queue.clone().detach()])
        logits_pd /= self.T2
        if self.mode == 'grafit':
            loss = args.w * loss_supcon + (1 - args.w) * loss_selfcon
        else:  # self.mode == 'coins'
            loss = args.w * loss_cls + (1 - args.w) * loss_selfcon
        self._dequeue_and_enqueue(k, coarse_label)

        return loss.mean()


class LinearHead(nn.Module):
    def __init__(self, net, dim_in=512, num_class=10):
        super().__init__()
        self.net = net
        self.fc = nn.Linear(dim_in, num_class)

        for param in self.net.parameters():
            param.requires_grad = False

        self.fc.weight.data.normal_(mean=0.0, std=0.01)
        self.fc.bias.data.zero_()

    def forward(self, x):
        with torch.no_grad():
            feat = self.net(x, feat=True)
        return self.fc(feat)
