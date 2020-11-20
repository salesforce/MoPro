import torch
import torch.nn as nn
from random import sample
import numpy as np
import torch.nn.functional as F

class MoPro(nn.Module):

    def __init__(self, base_encoder, args, width):
        super(MoPro, self).__init__()
        
        #encoder
        self.encoder_q = base_encoder(num_class=args.num_class,low_dim=args.low_dim,width=width)
        #momentum encoder
        self.encoder_k = base_encoder(num_class=args.num_class,low_dim=args.low_dim,width=width)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(args.low_dim, args.moco_queue))        
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))        
        self.register_buffer("prototypes", torch.zeros(args.num_class,args.low_dim))

    @torch.no_grad()
    def _momentum_update_key_encoder(self, args):
        """
        update momentum encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * args.moco_m + param_q.data * (1. - args.moco_m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, args):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert args.moco_queue % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % args.moco_queue  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]
    

    def forward(self, batch, args, is_eval=False, is_proto=False, is_clean=False):

        img = batch[0].cuda(args.gpu, non_blocking=True)        
        target = batch[1].cuda(args.gpu, non_blocking=True) 
        
        output,q = self.encoder_q(img)
        if is_eval:  
            return output, q, target
            
        img_aug = batch[2].cuda(args.gpu, non_blocking=True)          
        # compute augmented features 
        with torch.no_grad():  # no gradient 
            self._momentum_update_key_encoder(args)  # update the momentum encoder
            # shuffle for making use of BN
            img_aug, idx_unshuffle = self._batch_shuffle_ddp(img_aug)
            _, k = self.encoder_k(img_aug)  
            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute instance logits
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        # apply temperature
        logits /= args.temperature
        inst_labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        
        # dequeue and enqueue
        self._dequeue_and_enqueue(k, args) 
        
        if is_proto:     
            # compute protoypical logits
            prototypes = self.prototypes.clone().detach()
            logits_proto = torch.mm(q,prototypes.t())/args.temperature        
        else:
            logits_proto = 0
            
        if is_clean:
            # noise cleaning
            soft_label = args.alpha*F.softmax(output, dim=1) + (1-args.alpha)*F.softmax(logits_proto, dim=1)
            
            # keep ground truth label
            gt_score = soft_label[target>=0,target]
            clean_idx = gt_score>(1/args.num_class)          
            
            # assign a new pseudo label
            max_score, hard_label = soft_label.max(1)
            correct_idx = max_score>args.pseudo_th
            target[correct_idx] = hard_label[correct_idx]
            
            # confident sample index
            clean_idx = clean_idx | correct_idx            
            clean_idx_all = concat_all_gather(clean_idx.long())                       
        
        # aggregate features and (pseudo) labels across all gpus
        targets = concat_all_gather(target)
        features = concat_all_gather(q)

        if is_clean:                 
            clean_idx_all = clean_idx_all.bool()
            # update momentum prototypes with pseudo-labels
            for feat,label in zip(features[clean_idx_all],targets[clean_idx_all]):
                self.prototypes[label] = self.prototypes[label]*args.proto_m + (1-args.proto_m)*feat            
            # select only the confident samples to return
            q = q[clean_idx]
            target = target[clean_idx]            
            logits_proto = logits_proto[clean_idx]
            output = output[clean_idx]            
        else:
            # update momentum prototypes with original labels
            for feat,label in zip(features,targets):
                self.prototypes[label] = self.prototypes[label]*args.proto_m + (1-args.proto_m)*feat            

        # normalize prototypes    
        self.prototypes = F.normalize(self.prototypes, p=2, dim=1)        

        return output, target, logits, inst_labels, logits_proto


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
