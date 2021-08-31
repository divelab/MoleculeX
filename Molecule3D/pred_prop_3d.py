import torch

from molx.dataset import Molecule3DProps
from molx.model import Deepergcn_dagnn_dist, SchNet
from molx.mol3d import TransformPred3D, TransformGT3D, TransformRDKit3D
from molx.proppred import RegTrainer, eval_reg

conf = {}
conf['epochs'] = 100
conf['early_stopping'] = 200
conf['lr'] = 0.0001
conf['lr_decay_factor'] = 0.8
conf['lr_decay_step_size'] = 10
conf['weight_decay'] = 0
conf['batch_size'] = 20
conf['save_ckpt'] = 'best_valid'
conf['target'] = 0
conf['geoinput'] = 'pred'
conf['geo_model_path'] = 'results/exp0/ckpt_best_val.pth'
conf['metric'] = 'mae'
conf['out_path'] = 'pred_prop_results'
conf['split'] = 'random' #'scaffold'

conf['hidden_channels'] = 128
conf['num_filters'] = 128
conf['num_interactions'] = 6
conf['num_gaussians'] = 50
conf['cutoff'] = 10.0
conf['readout'] = 'add'
conf['dipole'] = False
conf['mean'] = None
conf['std'] = None
conf['atomref'] = None

conf['depth'] = 3
conf['emb_dim'] = 256
conf['dropout'] = 0
conf['norm'] = 'batch'
conf['JK'] = 'last'
conf['aggr'] = 'softmax'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if conf['geoinput'] == 'pred':
    geo_model = Deepergcn_dagnn_dist(num_layers=conf['depth'], emb_dim=conf['emb_dim'], drop_ratio=conf['dropout'], 
        JK=conf['JK'], aggr=conf['aggr'], norm=conf['norm']).to(device)
    geo_model_ckpt = torch.load(conf['geo_model_path'])
    geo_model.load_state_dict(geo_model_ckpt['model'])
    transform=TransformPred3D(geo_model, target_id=conf['target'], device=device)
elif conf['geoinput'] == 'gt':
    transform = TransformGT3D(target_id=conf['target'])
elif conf['geoinput'] == 'rdkit':
    transform = TransformRDKit3D(target_id=conf['target'])

train_dataset = Molecule3DProps(root='/mnt/data/shared/Molecule3D', transform=transform, split='train', split_mode=conf['split'])
val_dataset = Molecule3DProps(root='/mnt/data/shared/Molecule3D', transform=transform, split='val', split_mode=conf['split'])

trainer = RegTrainer(train_dataset, val_dataset, conf, device)
model = SchNet(hidden_channels=conf['hidden_channels'], num_filters=conf['num_filters'], num_interactions=conf['num_interactions'],
    num_gaussians=conf['num_gaussians'], cutoff=conf['cutoff'], readout=conf['readout'], dipole=conf['dipole'],
    mean=conf['mean'], std=conf['std'], atomref=conf['atomref']).to(device)
model = trainer.train(model)
eval_reg(model, val_dataset, conf['metric'], conf['geoinput'])