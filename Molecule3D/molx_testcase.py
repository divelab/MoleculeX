from torch_geometric.data import DataLoader

from molx.dataset import QM9, Molecule3D
from molx.models import DeeperGCN, Schnet
from molx.mol3d import EDMTrainer, eval3d, Transform3D
from molx.proppred import HomoLumoTrainer, evalHomoLumo
# We can also put downstream tasks under molx in the future, being parallel to mol3d


# Stage 1: train, eval and predict 3d coordinates

# get dataset without 3d coordinates
# We provide fixed split for reproducibility. Please refer to PPI from torch_geometric.
# We can also add options for users to customize their split.
dataset = Molecule3D(root='path_to_data', transform=None, split='train')
val_dataset = Molecule3D(root='path_to_data', transform=None, split='val')
loader = DataLoader(dataset)
val_loader = DataLoader(val_dataset)
model = DeeperGCN(n_layers = 32, activation = torch.nn.ReLU())

# train 3d prediction model
trainer = EDMTrainer(loader, val_loader, configs, device)
model = trainer.train(model) # can also use customized

# evaluate predicted distance or 3d coordinates
val_dataset = PubChemQCDataset(root='path_to_data', transform=None, split='test')
test_loader = DataLoader(test_dataset)
eval3d(model, test_loader)
# >>> MSE: xxxx, MAE: xxxx

# Stage 2a: downstream prediction with predicted 3d

# use a wrapped transform fn to obtain the 3d version of a mol graph dataset
transform = Transform3D(mode='pred', model=model)
dataset3d = QM9(root='path_to_data', transform=transform, split='train')
val_dataset3d = QM9(root='path_to_data', transform=transform, split='val')
loader = DataLoader(dataset3d)
val_loader = DataLoader(val_dataset3d)
hl_model = Schnet(n_layers = 5, cutoff = 10)

trainer = HomoLumoTrainer(loader, val_loader, configs, device)
hl_model = trainer.train(hl_model)
evalHomoLumo(hl_model, test_loader)


# Stage 2b: downstream prediction with groundtruth 3d
transform = Transform3D(mode='gt', model=None)
dataset3d = PubChemQCDataset(root='path_to_data', transform=transform, split='train')
val_dataset3d = PubChemQCDataset(root='path_to_data', transform=transform, split='val')
loader = DataLoader(dataset3d)
val_loader = DataLoader(val_dataset3d)
hl_model = Schnet(n_layers = 5, cutoff = 10)

trainer = HomoLumoTrainer(loader, val_loader, configs, device)
hl_model = trainer.train(hl_model)
evalHomoLumo(hl_model, test_loader)


# Stage 2c: using 2d input


'''
molx
 |- __init__.py
 |- dataset
 	|- QM9.py...
 |- models
	|- __init__.py
	|- DeeperGCN.py
	|- Schnet.py
 |- mol3d
 	|- __init__.py
 	|- utils.py
	|- train_eval.py
	|- transform_wrap.py
 |- proppred


'''