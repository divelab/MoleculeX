configs = {}

configs['example_config'] = {
    "n": 10, # Parameter in the subsequence kernel.
    "lambda": 1, # Parameter in the subsequence kernel.
    "n_iters": 3, # Iteration of WL kernel.
    "norm": False, # Whether to normalize kernel.
    "base_k": 'subtree' # Base kernel, choose from subtree, sp
}


delaney = {
    "n": 10,
    "lambda": 1,
    "n_iters": 2,
    "norm": False,
    "base_k": 'subtree'
}
configs['delaney'] = delaney


freesolv = {
    "n": 4,
    "lambda": 1,
    "n_iters": 3,
    "norm": False,
    "base_k": 'subtree'
}
configs['freesolv'] = freesolv


lipo = {
    "n": 6,
    "lambda": 1,
    "n_iters": 5,
    "norm": True,
    "base_k": 'subtree'
}
configs['lipo'] = lipo


hiv = {
    "n": 5,
    "lambda": 0.8,
    "n_iters": 7,
    "norm": False,
    "base_k": 'subtree'
}
configs['hiv'] = hiv


bace = {
    "n": 5,
    "lambda": 0.8,
    "n_iters": 5,
    "norm": True,
    "base_k": 'subtree'
}
configs['bace'] = bace


bbbp = {
    "n": 7,
    "lambda": 0.7,
    "n_iters": 7,
    "norm": False,
    "base_k": 'subtree'
}
configs['bbbp'] = bbbp


sider = {
    "n": 5,
    "lambda": 0.7,
    "n_iters": 6,
    "norm": True,
    "base_k": 'subtree'
}
configs['sider'] = sider


clintox = {
    "n": 3,
    "lambda": 1,
    "n_iters": 3,
    "norm": True,
    "base_k": 'subtree'
}
configs['clintox'] = clintox


tox21 = {
    "n":15,
    "lambda":0.8,
    "n_iters": 11,
    "norm": True,
    "base_k": 'subtree'
}
configs['tox21'] = tox21


qm8 = {
    "n":7,
    "lambda":1,
    "n_iters": 1,
    "norm": True,
    "base_k": 'subtree'
}
configs['qm8'] = qm8
