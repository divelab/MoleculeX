from numpy import linalg as LA
import numpy as np


def generate_xyz(ds, batch):
    num_nodes = ds.shape[0]
    d_list = []
    start = 0
    for i in range(1, num_nodes):
        if batch[i] > batch[i-1] or i == num_nodes:
            end = i
            d_list.append(ds[start:end, start:end])
            start = end
    d_list.append(ds[start:num_nodes, start:num_nodes])

    xyz_list = []
    fail_edm_count, fail_3d_count = 0, 0
    for d in d_list:
        valid_edm, xyz = from_d_to_xyz(d)
        if valid_edm == 'fail':
            fail_edm_count += 1
        if xyz == 'fail':
            fail_3d_count += 1
        xyz_list.append(xyz)

    return xyz_list, fail_edm_count, fail_3d_count


def from_d_to_xyz(d, threshold=4e-4):
    n = d.shape[0]
    d1j = d[0].unsqueeze(0).expand(n, n)
    di1 = d1j.T
    m = 0.5 * (d1j + di1 - d)

    w, v = LA.eig(m.cpu().numpy())

    w = (1 - (w < threshold) * (w > -threshold)) * w

    if np.sum(w < 0) > 0:
        valid_edm = 'fail'
        xyz = 'fail'
        return valid_edm, xyz

    if np.sum(w != 0) > 3:
        valid_edm = 'good'
        xyz = 'fail'
        return valid_edm, xyz

    s = np.eye(n) * w
    s_sqrt = np.sqrt(s)
    xyz = np.matmul(v, s_sqrt)
    return 'good', xyz


