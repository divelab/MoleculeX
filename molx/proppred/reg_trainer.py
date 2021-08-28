import torch
import numpy as np


def eval_reg(model, dataloader, metric, target, geoinput):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    errors = 0.
    for batch_data in dataloader:
        batch_data = batch_data.to(device)
        with torch.no_grad():

            if geoinput in ['gt', 'rdkit']:
                # use schnet model
                pred = model(batch_data, dist_index=None, dist_weight=None)

            elif geoinput == 'pred':
                # use schnet model
                pred = model(batch_data, dist_index=batch_data.dist_index, dist_weight=batch_data.dist_weight)

            elif geoinput == '2d':
                # use schnet_2d model
                pred = model(batch_data)

            else:
                raise NameError('Must use gt, rdkit, pred or 2d for geoinput in arguments!')

        if metric == 'mae':
            mae_sum = ((pred.view(-1) - batch_data.y[:, target]).abs()).sum()
            errors += mae_sum.cpu().detach().numpy()
        elif metric == 'rmse':
            mse_sum = (torch.square((pred.view(-1) - batch_data.y[:, target]))).sum()
            errors += mse_sum.cpu().detach().numpy()

    if metric == 'mae':
        out = errors / len(dataloader)
    elif metric == 'rmse':
        out = np.sqrt(errors / len(dataloader))

    return out

