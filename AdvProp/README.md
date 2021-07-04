# Advanced Graph and Sequence Neural Networks for Molecular Property Prediction and Drug Discovery
[[Paper]](https://arxiv.org/abs/2012.01981)[[Supplementary]](https://documentcloud.adobe.com/link/track?uri=urn:aaid:scds:US:d0ca85d1-c6f9-428b-ae2b-c3bf3257196d#pageNum=1)

## Overview

Properties of molecules are indicative of their functions and thus are useful in many applications. With the advances of deep learning methods, computational approaches for predicting molecular properties are gaining increasing momentum. However, there lacks customized and advanced methods and comprehensive tools for this task currently. Here we develop a suite of comprehensive machine learning methods and tools spanning different computational models, molecular representations, and loss functions for molecular property prediction and drug discovery. Specifically, we represent molecules as both graphs and sequences. Built on these
representations, we develop novel deep models for learning from molecular graphs and sequences. In order to learn effectively from highly imbalanced datasets, we develop advanced loss functions that optimize areas under precision-recall curves. Altogether, our work not only serves as a comprehensive tool, but also contributes towards developing novel and advanced graph and sequence learning methodologies. Results on both online and offline antibiotics discovery and molecular property prediction tasks show that our methods achieves consistent improvements over prior methods. In particular, our methods achieve \#1 ranking in terms of both ROC-AUC and PRC-AUC on the AI Cures Open Challenge for drug discovery related to COVID-19.

AdvProp is unique in four aspects:

* AdvProp consists of a suite of comprehensive machine learning methods across different data types and method types. We expect them to provide complementary information for molecular property prediction and yield better performance. 
* An effective graph-based deep learning method named multi-level message passing neural network (ML-MPNN) is proposed to make full use of richly informative molecular graphs.
* A new sequence-based deep learning method named contrastive-BERT, pretrained by a novel self-supervised task via contrastive learning, is incorporated.
* Effective loss functions for optmizing ROC-AUC and PRC-AUC.

<p align="center">
<img src="https://github.com/divelab/MoleculeX/blob/master/imgs/overview.png" width="1000" class="center" alt="overview"/>
    <br/>
</p>

## Usage

MoleculeX has four modules covering deep and non-deep methods based on both molecular graphs and SMILES sequence:
* [ML-MPNN](graph)
* [Weisfeiler-Lehman subtree kernel](kernels)
* [contrastive-BERT](sequence)
* [subsequence kernel](kernels)

The use of AdvProp requires the running of above four models with four output results. The four output results are then ensembled as the final prediction. Users of AdvProp are also given the freedom of employing fewer modules.

The environment requirements for these models might have conflict and we hence recommend create individual environments for each of them. To get started with AdvProp, access the above links for your desired modules.

We use the LibAUC package to optimize ROC-AUC and PRC-AUC with effective surrogate loss functions. See the [package website](https://libauc.org/) for more information.

## Reference
```
@article{wang2020advanced,
  title={Advanced Graph and Sequence Neural Networks for Molecular Property Prediction and Drug Discovery},
  author={Wang, Zhengyang and Liu, Meng and Luo, Youzhi and Xu, Zhao and Xie, Yaochen and Wang, Limei and Cai, Lei and Ji, Shuiwang},
  journal={arXiv preprint arXiv:2012.01981},
  year={2020}
}
```

