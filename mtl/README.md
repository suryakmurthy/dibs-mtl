# DiBS-MTL

This repository contains the MTL experiments used in **"DiBS-MTL: Transformation-Invariant Multitask Learning with Direction Oracles".**

## Setup environment

```bash
conda create -n mtl python=3.9.23
conda activate mtl
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=10.2 -c pytorch
conda install pyg -c pyg -c conda-forge
```

Install the repo:

```bash
git clone https://github.com/suryakmurthy/dibs-mtl.git
cd dibs-mtl
pip install -e .
```

## Run experiment

To run experiments:

```bash
cd experiment/<experiment name>
python trainer.py --method=dibsmtl
```

Follow instructions in the experiment README file for more information regarding datasets.  

Here `<experiment name>` is one of `[toy, nyuv2]`. You can also replace `nashmtl` with one of the following MTL methods.

| Method (code name) | Paper (notes) |
| :---: | :---: |
| DiBS-MTL (`dibsmtl`) | [TBD](TBD) |
| Nash-MTL (`nashmtl`) | [Multi-Task Learning as a Bargaining Game](https://arxiv.org/pdf/2202.01017v1.pdf) |
| FAMO (`famo`) | [FAMO: Fast Adaptive Multitask Optimization](https://arxiv.org/abs/2306.03792) |
| CAGrad (`cagrad`) | [Conflict-Averse Gradient Descent for Multi-task Learning](https://arxiv.org/pdf/2110.14048.pdf) |
| PCGrad (`pcgrad`) | [Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782) |
| IMTL-G (`imtl`) | [Towards Impartial Multi-task Learning](https://openreview.net/forum?id=IMPnRXEWpvr) |
| MGDA (`mgda`) | [Multi-Task Learning as Multi-Objective Optimization](https://arxiv.org/abs/1810.04650) |
| DWA (`dwa`) | [End-to-End Multi-Task Learning with Attention](https://arxiv.org/abs/1803.10704) |
| Uncertainty weighting (`uw`) | [Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics](https://arxiv.org/pdf/1705.07115v3.pdf) |
| Linear scalarization (`ls`) | - (equal weighting) |
| Scale-invariant baseline (`scaleinvls`) | - (see Nash-MTL paper for details) |
| Random Loss Weighting (`rlw`) | [A Closer Look at Loss Weighting in Multi-Task Learning](https://arxiv.org/pdf/2111.10603.pdf) |

Following NashMTL, this code supports experiment tracking with **[Weights & Biases](https://wandb.ai/site)** with two additional parameters:

```bash
python trainer.py --method=nashmtl --wandb_project=<project-name> --wandb_entity=<entity-name>
```

## Citation

This repository was built on the Nash-MTL and FAMO repositories. If you wish to cite this repository, please cite the following:

```bibtex
@article{liu2021conflict,
  title={Conflict-Averse Gradient Descent for Multi-task Learning},
  author={Liu, Bo and Liu, Xingchao and Jin, Xiaojie and Stone, Peter and Liu, Qiang},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}

@article{navon2022multi,
  title={Multi-Task Learning as a Bargaining Game},
  author={Navon, Aviv and Shamsian, Aviv and Achituve, Idan and Maron, Haggai and Kawaguchi, Kenji and Chechik, Gal and Fetaya, Ethan},
  journal={arXiv preprint arXiv:2202.01017},
  year={2022}
}

@misc{liu2023famo,
  title={FAMO: Fast Adaptive Multitask Optimization},
  author={Bo Liu and Yihao Feng and Peter Stone and Qiang Liu},
  year={2023},
  eprint={2306.03792},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```