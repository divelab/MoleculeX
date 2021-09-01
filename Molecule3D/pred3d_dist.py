import torch
import sys
sys.path.insert(1, '/mnt/data/shared/zhaoxu/workspace/MoleculeX/')

from molx.dataset import Molecule3D
from molx.model import Deepergcn_dagnn_dist, Deepergcn_dagnn_coords, SchNet, SchNet2D
from molx.mol3d import Mol3DTrainer, eval3d

conf = {}
conf['epochs'] = 100
conf['early_stopping'] = 200
conf['lr'] = 0.0001
conf['lr_decay_factor'] = 0.8
conf['lr_decay_step_size'] = 10
conf['dropout'] = 0
conf['weight_decay'] = 0
conf['depth'] = 3
conf['hidden'] = 256
conf['batch_size'] = 20
conf['save_ckpt'] = 'best_valid'
conf['norm'] = 'batch'
conf['JK'] = 'last'
conf['out_path'] = 'results/exp0/'
conf['split'] = 'random' #'scaffold'
conf['criterion'] = 'mse'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = Molecule3D(root='/mnt/data/shared/Molecule3D/', transform=None, split='train', split_mode=conf['split'])
val_dataset = Molecule3D(root='/mnt/data/shared/Molecule3D/', transform=None, split='val', split_mode=conf['split'])
test_dataset = Molecule3D(root='/mnt/data/shared/Molecule3D/', transform=None, split='test', split_mode=conf['split'])
model = Deepergcn_dagnn_dist(num_layers=conf['depth'], emb_dim=conf['hidden'], drop_ratio=conf['dropout'], JK="last", aggr='softmax', norm='batch').to(device)

print('start train')
trainer = Mol3DTrainer(train_dataset, val_dataset, conf,
                       device=device)
model = trainer.train(model)
print('start evaluate')
eval3d(model, test_dataset)