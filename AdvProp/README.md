# Advanced Graph and Sequence Neural Networks for Molecular Property Prediction and Drug Discovery
[Paper [Bioinformatics]](https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/btac112/6531963?login=true)|[Supplementary](https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/bioinformatics/PAP/10.1093_bioinformatics_btac112/2/btac112_supplementary_data.pdf?Expires=1649088997&Signature=o6h8d3HSp7cKuIrUIOg5YnBeZVFQrugkMbMnh~NDzeMpRjuYvYKgB~VCrF~wTnnH5Hw7oENdHqdzVh7hIbWlLkc0R6I9P6fmly3ICQN4F~CxUqiI4LOr7xnZkOhqT8c0ONypTaFhEJhEu4AA8PVsCpCWxfMN0TiNvt753boZSbcHeZlxHx-ji3z89wrWN-Ac7NGM-bEFRUnY8oD9RmHrDHsrc5VML8xcCe30~wvZ40PHYJRtVDnawuWfMX8Zh0LSBbVzhkvx8Zp5xEfVab~qv3U5jAMVQT6a8CDJXaoYqBDwWkHQHH87orTO2RLIAueH9G4WxIvtocFOTMaQ6TLuiQ__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA)

## Overview

Properties of molecules are indicative of their functions and thus are useful in many applications. With the advances of deep learning methods, computational approaches for predicting molecular properties are gaining increasing momentum. However, there lacks customized and advanced methods and comprehensive tools for this task currently. Here we develop a suite of comprehensive machine learning methods and tools spanning different computational models, molecular representations, and loss functions for molecular property prediction and drug discovery. Specifically, we represent molecules as both graphs and sequences. Built on these
representations, we develop novel deep models for learning from molecular graphs and sequences. In order to learn effectively from highly imbalanced datasets, we develop advanced loss functions that optimize areas under precision-recall curves. Altogether, our work not only serves as a comprehensive tool, but also contributes towards developing novel and advanced graph and sequence learning methodologies. Results on both online and offline antibiotics discovery and molecular property prediction tasks show that our methods achieves consistent improvements over prior methods. In particular, our methods achieve \#1 ranking in terms of both ROC-AUC and PRC-AUC on the [AI Cures Open Challenge](https://www.aicures.mit.edu/tasks) for drug discovery related to COVID-19.

AdvProp is unique in four aspects:

* AdvProp consists of a suite of comprehensive machine learning methods across different data types and method types. We expect them to provide complementary information for molecular property prediction and yield better performance. 
* An effective graph-based deep learning method named multi-level message passing neural network (ML-MPNN) is proposed to make full use of richly informative molecular graphs.
* A new sequence-based deep learning method named contrastive-BERT, pretrained by a novel self-supervised task via contrastive learning, is incorporated.
* Effective stochastic methods towards optimizing ROC-AUC and PRC-AUC for deep learning.

<p align="center">
<img src="https://github.com/divelab/MoleculeX/blob/master/imgs/overview.png" width="1000" class="center" alt="overview"/>
    <br/>
</p>

## Usage

AdvProp has four modules covering deep and non-deep methods based on both molecular graphs and SMILES sequence:
* [ML-MPNN](graph)
* [Weisfeiler-Lehman subtree kernel](kernels)
* [contrastive-BERT](sequence)
* [subsequence kernel](kernels)

The use of AdvProp requires the running of above four models with four output results. The four output results are then ensembled as the final prediction. Users of AdvProp are also given the freedom of employing fewer modules.

The environment requirements for these models might have conflict and we hence recommend create individual environments for each of them. To get started with AdvProp, access the above links for your desired modules.

We use the LibAUC package to optimize [ROC-AUC](https://arxiv.org/abs/2012.03173) and [PRC-AUC](https://arxiv.org/abs/2104.08736) with effective stochastic methods for appropriate surrogate loss functions of these two AUC scores. See the [package website](https://libauc.org/) for more information.

## Reference
```
@article{wang2020advanced,
  title={Advanced Graph and Sequence Neural Networks for Molecular Property Prediction and Drug Discovery},
  author={Wang, Zhengyang and Liu, Meng and Luo, Youzhi and Xu, Zhao and Xie, Yaochen and Wang, Limei and Cai, Lei and Qi, Qi and Yuan, Zhuoning and Yang, Tianbao and Ji, Shuiwang},
  journal={arXiv preprint arXiv:2012.01981},
  year={2020}
}
```

