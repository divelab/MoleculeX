<p align="center">
<img src="https://github.com/divelab/MoleculeX/blob/molx/imgs/MX-logo.jpg" width="800" class="center" alt="logo"/>
    <br/>
</p>

------

## Project Links:
- [Molecule3D](https://github.com/divelab/MoleculeX/tree/molx/Molecule3D)
- [BasicProp](https://github.com/divelab/MoleculeX/tree/molx/BasicProp/kddcup2021)
- [AdvProp](https://github.com/divelab/MoleculeX/tree/molx/AdvProp)

## Overview
MoleculeX is a new and rapidly growing suite of machine learning methods and software tools for molecule exploration. The ultimate goal of MoleculeX is to enable a variety of basic and complex molecular modeling tasks, such as molecular property prediction, 3D geometry modeling, etc. Currently, MoleculeX includes a set of machine learning methods for ground-state 3D molecular geometry prediction and molecular property prediction. Specifically, **BasicProp** includes basic supervised learning methods based on graph neural networks for molecular property prediction. **BasicProp** is suitable for tasks in which large numbers of labeled samples are available, and thus only supervised learning is required. **BasicProp** has been used to participate in the [2021 KDD Cup on OGB Large-Scale Challenge (OGB-LSC)](https://ogb.stanford.edu/kddcup2021/leaderboard/#final_pcqm4m) and is **one of the winners**. **AdvProp** includes machine learning methods for molecular property prediction when only a small number of labeled samples are available, and thus self-supervised learning is necessary to achieve desirable performance. In addition, **AdvProp** is able to deal with tasks in which samples from different classes are highly imbalanced. In these cases, we employ advanced loss functions that optimize various areas under curves (AUC), such as areas under the receiver operating characteristic (AUROC) and the precision recall curve (AUPRC). **AdvProp** has been used to participate in the [AI Cures open challenge for COVID-19](https://www.aicures.mit.edu/tasks) and is now **ranked #1** in terms of both AUROC and AUPRC on the leaderboard. Besides, **[Molecule3D](https://github.com/divelab/MoleculeX/tree/molx/Molecule3D)** provides a set of software tools for processing our proposed Molecule3D dataset, a novel dataset specifically designed for ground-state 3D molecular geometry prediction. It also includes several baseline methods for geometry prediction, and quantum property prediction methods using the predicted 3D geometries as inputs. Currently, the pip package of MoleculeX only includes the code of the Molecule3D module. We will include other modules gradually in the future.

<p align="center">
<img src="https://github.com/divelab/MoleculeX/blob/molx/imgs/moleculex_overview.jpg" width="800" class="center" alt="logo"/>
    <br/>
</p>
