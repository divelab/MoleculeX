import numpy as np
from utils import smile_to_graph
from shogun import StringCharFeatures, SubsequenceStringKernel, RAWBYTE

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from grakel.utils import graph_from_networkx
from grakel.kernels import WeisfeilerLehman, VertexHistogram, EdgeHistogram, ShortestPath

class Kernel(object):
    def __init__(self, config, kernel_type):
        self.config = config
        self.kernel_type = kernel_type
        
        if not kernel_type in ['graph', 'sequence', 'combined']:
            raise NotImplementedError
        elif kernel_type == 'graph':
            base_kernel = {"sp": ShortestPath, 
                           "subtree": VertexHistogram
                          }[config['base_k']]
            self.kernel = GraphKernel(n_iter=config['n_iters'], 
                                      normalize=config['norm'], 
                                      base_graph_kernel=base_kernel)
        elif kernel_type == 'sequence':
            self.kernel = SequenceKernel(n=config['n'], 
                                         lambd=config['lambda'])
        else:
            base_kernel = {"sp": ShortestPath, 
                           "subtree": VertexHistogram
                          }[config['base_k']]
            self.kernel = [SequenceKernel(n=config['n'], 
                                          lambd=config['lambda']),
                           GraphKernel(n_iter=config['n_iters'], 
                                       normalize=True, 
                                       base_graph_kernel=base_kernel)]

    def fit_transform(self, X_smiles):
        if type(self.kernel) is list:
            K = sum([k.fit_transform(X_smiles) for k in self.kernel])/len(self.kernel)
        else:
            K = self.kernel.fit_transform(X_smiles)
        return K
    
    def transform(self, X_smiles):
        if self.kernel is list:
            K = sum([k.transform(X_smiles) for k in self.kernel])/len(self.kernel)
        else:
            K = self.kernel.transform(X_smiles)
        return K

    
    
class GraphKernel(WeisfeilerLehman):

    def fit_transform(self, train_smiles):
        X_G = np.array([smile_to_graph(s) for s in train_smiles])
        X_G = graph_from_networkx(X_G, node_labels_tag='label')
        return super().fit_transform(X_G)
    
    def transform(self, test_smiles):
        X_G = np.array([smile_to_graph(s) for s in test_smiles])
        X_G = graph_from_networkx(X_G, node_labels_tag='label')
        return super().transform(X_G)


    
class SequenceKernel(object):
    def __init__(self, n, lambd):
        self.n = n
        self.lambd = lambd
        self.train_smiles = None

    def fit_transform(self, train_smiles):
        self.train_smiles = train_smiles
        phi_train = StringCharFeatures(train_smiles.tolist(), RAWBYTE)
        K_train =  SubsequenceStringKernel(phi_train, 
                                           phi_train, 
                                           self.n, 
                                           self.lambd).get_kernel_matrix()
        return K_train
    
    def transform(self, test_smiles):
        phi_train = StringCharFeatures(self.train_smiles.tolist(), RAWBYTE)
        phi_test = StringCharFeatures(test_smiles.tolist(), RAWBYTE)
        K_test = SubsequenceStringKernel(phi_test, 
                                         phi_train, 
                                         self.n, 
                                         self.lambd).get_kernel_matrix()
        return K_test
    
        
