import torch
from rdkit import Chem
from torch_geometric.data import Data, Batch

class TransformPred3D(torch.nn.Module):
    """ Converting :obj:`torch_geometric.data.Data` objects into 3d versions
    using 3d prediction models.
    """
    def __init__(self, model3d=None, device=None):
        super(TransformPred3D, self).__init__()
        self.model3d = model3d
        self.device = device
        
    def forward(self, data):
        in_device = data.x.device
        batch_data = Batch.from_data_list([data])
        if device is not None:
            batch_data.to(self.device)
            self.model3d.to(self.device)
            
        pred, _, _ = self.model3d(batch_data)
        dist_index = torch.nonzero(pred, as_tuple=False)
        dist_weight = pred[torch.nonzero(pred, as_tuple=True)]
        
        transformed = data.clone()
        transformed.__setitem__('dist_index', dist_index.to(in_device))
        transformed.__setitem__('dist_weight', dist_weight.to(in_device))
        return transformed
    
    
class TransformGT3D():
    """ Converting datasets containing 3d groundtruths into accepted format
    by downstream 3D networks.
    """
    def __init__(self, key_mapping):
        self.key_mapping = key_mapping
        
    def __call__(self, data):
        transformed = data.clone()
        for orig_key, target_key in key_mapping.items():
            transformed.__delitem__(orig_key)
            transformed.__setitem__(target_key, data.__getitem__(orig_key))
        return transformed
    

class TransformRDKit3D():
    """ Converting :obj:`torch_geometric.data.Data` objects containing SMILES 
    attribute into 3d versions using RDKit.
    """
    def __init__(self, conf_id=-1):
        self.conf_id = conf_id
    
    def __call__(self, data):
        mol = Chem.MolFromSmiles(data.smiles)
        coords = mol.GetConformer(self.conf_id).GetPositions()
        xyz = torch.tensor(coords, dtype=torch.float32)
        
        transformed = data.clone()
        transformed.__setitem__('xyz', data.__getitem__(orig_key))
        return transformed