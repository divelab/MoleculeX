<p align="center">
<img src="https://github.com/divelab/MoleculeX/blob/master/imgs/MX-logo.jpg" width="500" class="center" alt="logo"/>
    <br/>
</p>

------

## Overview
MoleculeX is a new and rapidly growing suite of machine learning methods and software tools for molecule exploration. The ultimate goal of MoleculeX is to enable a variety of basic and complex molecular modeling tasks, such as molecular property prediction, 3D geometry modeling, etc. Currently, MoleculeX includes a set of machine learning methods for molecular property prediction. Specifically, BasicProp includes basic supervised learning methods based on graph neural networks for molecular property prediction. BasicProp is suitable for tasks in which large numbers of labeled samples are available, and thus only supervised learning is required. BasicProp has been used to participate in the 2021 KDD Cup on OGB Large-Scale Challenge (OGB-LSC). AdvProp includes machine learning methods for molecular property prediction when only a small number of labeled samples are available, and thus self-supervised learning is necessary to achieve desirable performance. In addition, AdvProp is able to deal with tasks in which samples from different classes are highly imbalanced. In these cases, we employ advanced loss functions that optimize various areas under curves (AUC), such as areas under the receiver operating characteristic (AUROC) and the precision recall curve (AUPRC).