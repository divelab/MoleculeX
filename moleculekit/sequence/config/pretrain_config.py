"""
    This is the configuration file for pretraining.
    We implemented two pretrain tasks: mask prediction and mask contrastive learning.
    --- Mask prediction is just the same pretrain task used in original BERT paper, predicting the ids of masked tokens
    --- Mask contrastive learning is our propsed pretraining task.  
"""

######################################################################################################################
# Settings for BERT network
## hidden: number of hidden units in each Transformer layer
## n_layers: number of Transformer layers
## attn_heads: number of attention heads in each Transformer layer
## dropout: dropout rate in each dropout layer
## activation: the non-linear activation function used in feed-forward network of each Transformer layer, use 'relu' or 'gelu'
######################################################################################################################
conf_net = {'hidden':1024, 'n_layers':6, 'attn_heads':4, 'dropout':0.1, 'activation':'gelu'}

######################################################################################################################
# Setting for optimizer
## Type can be 'adam' (torch.optim.Adam), 'rms' (torch.optim.RMSprop), or 'sgd' (torch.optim.SGD)
## Param is just the parameters of optimizer, the parameter names are the same as those in pytorch
######################################################################################################################
conf_optim = {}
conf_optim['type'] = 'adam'
conf_optim['param'] = {'betas':(0.9,0.999), 'weight_decay':0, 'lr': 1e-4}

######################################################################################################################
# Setting for learning rate scheduler
##  We provide three types of learning scheduler:
## --- 'cos' Cosine Annealing
## --- 'linear' Learing rate is decayed linearly
## --- 'square' Learing rate is inversely proportional to the square root of current epoch num
## --- If you set 'type' as None, no learning rate scheduler will be used, i.e. learning rate is a constant
## Learning rate is linearly increased from 'init_lr' (conf_lr_scheduler['param']['init_lr'] to 'base_lr' (the 'lr' in conf_optim['param']), i.e
## warming up, in the first several epoches setted by 'warm_up_epoches' (set 'warm_up_epoches' as 1 to skip warming up), then it is decayed.
## The decaying type is different for different learning rate scheduler.
######################################################################################################################
conf_lr_scheduler = {}
conf_lr_scheduler['type'] = 'cos'
conf_lr_scheduler['param'] = {'warm_up_epoches': 1, 'init_lr': 0}

conf_trainer = {}
conf_trainer['optim'] = conf_optim
conf_trainer['lr_scheduler'] = conf_lr_scheduler
conf_trainer['net'] = conf_net

######################################################################################################################
# Other settings for training
## epoches: number of epoches
## batch_size: batch size
## seq_max_len: if it is not None, the dataloader will automatically filter the sequences with a length longer than it. It can help save memory.
## verbose: after how many iterations the loss value will be displayed in terminal
## save_ckpt: after how many epoches the current checkpoint will be saved
## ckpt_file: checkpoint file path, if it is not None, the program will start training from this saved checkpoint
## use_aug: whether data augmentation will be applied in training
## use_cls_token: set it to True when using BERT model
## pretrain_task: 'mask_pred' for mask prediction and 'mask_con' for mask contrastive learning
######################################################################################################################
conf_trainer['epoches'] = 100
conf_trainer['batch_size'] = 128
conf_trainer['seq_max_len'] = 128
conf_trainer['verbose'] = 1
conf_trainer['save_ckpt'] = 100
conf_trainer['ckpt_file'] = None
conf_trainer['use_aug'] = True
conf_trainer['use_cls_token'] = True
conf_trainer['pretrain_task'] = 'mask_con'

######################################################################################################################
# Other settings for validation/test
## batch_size: batch size
## seq_max_len: if it is not None, the dataloader will automatically filter the sequences with a length longer than it. It can help save memory.
## use_aug: whether data augmentation will be applied in training
## use_cls_token: set it to True when using BERT model
## pretrain_task: 'mask_pred' for mask prediction and 'mask_con' for mask contrastive learning
######################################################################################################################
conf_tester = {}
conf_tester['net'] = conf_net
conf_tester['batch_size'] =16
conf_tester['seq_max_len'] = conf_trainer['seq_max_len']
conf_tester['use_aug'] = False
conf_tester['use_cls_token'] = True
conf_tester['pretrain_task'] = 'mask_con'

assert conf_optim['type'] in ['adam', 'rms', 'sgd']
assert conf_lr_scheduler['type'] in ['cos', 'linear', 'square', None]
assert (conf_trainer['pretrain_task'] in ['mask_con', 'mask_pred']) and (conf_trainer['pretrain_task'] == conf_tester['pretrain_task'])