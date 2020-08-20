conf_data_io = {}
conf_data_io['split'] = 'random'
conf_data_io['split_ratio'] = [0.8,0.1]
conf_data_io['seed'] = 122
conf_data_io['smile_id'] = None
conf_data_io['label_id'] = None

conf_net = {'hidden':1024, 'n_layers':6, 'attn_heads':4, 'dropout':0.1, 'embed':['pos'], 'activation':'gelu'}

conf_optim = {}
conf_optim['type'] = 'adam'
conf_optim['param'] = {'betas':(0.9,0.999), 'weight_decay':0, 'lr': 1e-4}

conf_lr_scheduler = {}
conf_lr_scheduler['type'] = 'cos'
conf_lr_scheduler['param'] = {'warm_up_epoches': 1, 'init_lr': 0}

conf_trainer = {}
conf_trainer['data_io'] = conf_data_io
conf_trainer['optim'] = conf_optim
conf_trainer['lr_scheduler'] = conf_lr_scheduler
conf_trainer['net'] = conf_net

conf_trainer['epoches'] = 100
conf_trainer['batch_size'] = 128
conf_trainer['seq_max_len'] = 128
conf_trainer['verbose'] = 1
conf_trainer['use_gpu'] = True
conf_trainer['save_ckpt'] = 100
conf_trainer['ckpt_file'] = None
conf_trainer['use_aug'] = False
conf_trainer['use_cls_token'] = True
conf_trainer['pretrain_task'] = 'mask_con'

conf_tester = {}
conf_tester['net'] = conf_net
conf_tester['use_gpu'] = True
conf_tester['batch_size'] =64
conf_tester['use_aug'] = False
conf_tester['use_cls_token'] = True
conf_tester['pretrain_task'] = 'mask_con'

assert conf_optim['type'] in ['adam', 'rms', 'sgd']
assert conf_lr_scheduler['type'] in ['cos', 'linear', 'square', None]
assert (conf_trainer['pretrain_task'] in ['mask_con', 'mask_pred']) and (conf_trainer['pretrain_task'] == conf_tester['pretrain_task'])