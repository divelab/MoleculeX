conf_data_io = {}
conf_data_io['split'] = 'random'
conf_data_io['split_ratio'] = [0.8,0.1]
conf_data_io['seed'] = 122
conf_data_io['smile_id'] = 0
conf_data_io['label_id'] = list(range(1,13))

conf_net = {}
conf_net['type'] = 'bert_tar'
conf_net['param'] = {'task':'reg', 'n_out':12, 'hidden':1024, 'n_layers':6, 'attn_heads':4, 'dropout':0.1, 'embed':['pos'], 'activation':'gelu'}

conf_optim = {}
conf_optim['type'] = 'adam'
conf_optim['param'] = {'betas':(0.9,0.999), 'weight_decay':0, 'lr': 2e-5}

conf_loss = {}
conf_loss['type'] = 'mse'
conf_loss['param'] = None

conf_lr_scheduler = {}
conf_lr_scheduler['type'] = None
conf_lr_scheduler['param'] = None

conf_trainer = {}
conf_trainer['data_io'] = conf_data_io
conf_trainer['optim'] = conf_optim
conf_trainer['loss'] = conf_loss
conf_trainer['lr_scheduler'] = conf_lr_scheduler
conf_trainer['net'] = conf_net

conf_trainer['epoches'] = 100
conf_trainer['batch_size'] = 128
conf_trainer['seq_max_len'] = None
conf_trainer['verbose'] = 1
conf_trainer['use_gpu'] = True
conf_trainer['save_ckpt'] = 100
conf_trainer['ckpt_file'] = None
conf_trainer['use_aug'] = True
conf_trainer['use_cls_token'] = True
conf_trainer['save_model'] = 'best_valid'
conf_trainer['save_valid_records'] = False
conf_trainer['pretrain_model'] = None

conf_tester = {}
conf_tester['loss'] = conf_loss
conf_tester['net'] = conf_net
conf_tester['use_gpu'] = True
conf_tester['batch_size'] =64
conf_tester['task'] = 'reg'
conf_tester['use_aug'] = True
conf_tester['use_cls_token'] = True

assert conf_optim['type'] in ['adam', 'rms', 'sgd']
assert conf_lr_scheduler['type'] in ['cos', 'linear', 'square', None]
assert conf_net['param']['task'] == conf_tester['task']
assert len(conf_data_io['label_id']) == conf_net['param']['n_out']