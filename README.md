<p align="center">
<img src="https://github.com/divelab/MoleculeKit/blob/master/imgs/logo.png" width="500" class="center" alt="logo"/>
    <br/>
</p>

------

# MoleculeKit: Machine Learning Methods for Molecular Property Prediction and Drug Discovery [[Paper]](https://arxiv.org/abs/2012.01981)

## Overview

MoleculeKit is an advanced machine learning tool for molecular property prediction and drug discovery. 

MoleculeKit is unique in three aspects:

* MoleculeKit consists of a suite of comprehensive machine learning methods across different data types and method types. We expect them to provide complementary information for molecular property prediction and yield better performance. 
* An effective graph-based deep learning method named multi-level message passing neural network (ML-MPNN) is proposed to make full use of richly informative molecular graphs.
* A new sequence-based deep learning method named contrastive-BERT, pretrained by a novel self-supervised task via contrastive learning, is incorporated.

<p align="center">
<img src="https://github.com/divelab/MoleculeKit/blob/master/imgs/overview.png" width="1000" class="center" alt="overview"/>
    <br/>
</p>

## Usage

MoleculeKit has four modules covering deep and non-deep methods based on both molecular graphs and SMILES sequence:
* [ML-MPNN](https://github.com/divelab/MoleculeKit/tree/master/moleculekit/graph)
* [Weisfeiler-Lehman subtree kernel](https://github.com/divelab/MoleculeKit/tree/master/moleculekit/kernels)
* [contrastive-BERT](https://github.com/divelab/MoleculeKit/tree/master/moleculekit/sequence)
* [subsequence kernel](https://github.com/divelab/MoleculeKit/tree/master/moleculekit/kernels)

The use of MoleculeKit requires the running of above four models with four output results. The four output results are then ensembled as the final prediction. Users of MoleculeKit are also given the freedom of employing fewer modules.

The environment requirements for these models might have conflict and we hence recommend create individual environments for each of them. To get started with MoleculeKit, access the above links for your desired modules.

## Reference
```
@article{wang2020moleculekit,
  title={MoleculeKit: Machine Learning Methods for Molecular Property Prediction and Drug Discovery},
  author={Wang, Zhengyang and Liu, Meng and Luo, Youzhi and Xu, Zhao and Xie, Yaochen and Wang, Limei and Cai, Lei and Ji, Shuiwang},
  journal={arXiv preprint arXiv:2012.01981},
  year={2020}
}
```

