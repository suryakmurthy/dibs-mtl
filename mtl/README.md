# DiBS-MTL

Code used for the multi-task learning results for _*Monotonic Transformation Invariant Multi-task Learning*_. This codebase is built on the repository released by the authors of *Improvable Gap Balancing for Multi-Task Learning*.

## Setup environment

```bash
conda create -n mtl python=3.8.13
conda activate mtl
python -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

Install the repo:

```bash
cd mtl
pip install -r requirements.txt
```

## Run experiment

Follow instruction in the experiment README file for more information regarding, e.g., datasets.

We support our IGB methods and other existing MTL methods with a unified API. To run experiments:

```bash
cd experiments/<experiment name>
python trainer.py --loss_method=<loss balancing method> --gradient_method=<gradient balancing method>
```
  
Here,
- `<experiment name>` is one of `[toy, cityscapes, quantum_chemistry, nyuv2]`.
- `<loss balancing method>` is one of `igbv1`, `igbv2` and the following loss balancing MTL methods.
- `<gradient balancing method>` is one of the following gradient balancing MTL methods.
- Both `<loss balancing method>` and `<gradient balancing method>` are optional:
  - only using `<loss balancing method>` is to run a loss balancing method;
  - only using `<gradient balancing method>` is to run a gradient balancing method;
  - using neither is to run Equal Weighting (EW) method.
  - using both is to run a combined MTL method by both loss balancing and gradient balancing.

## MTL methods

We support the following loss balancing and gradient balancing methods. This repository has been updated to include additional baselines beyond the original implementation.


|   Loss Balancing Method (code name)   |                                                          Paper (notes)                                                           |
|:-------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------:|
|        Equal Weighting (`ls`)         |                                                     - (linear scalarization)                                                     |
|     Random Loss Weighting (`rlw`)     |                  [A Closer Look at Loss Weighting in Multi-Task Learning](https://arxiv.org/pdf/2111.10603.pdf)                  |
|    Dynamic Weight Average (`dwa`)     |                        [End-to-End Multi-Task Learning with Attention](https://arxiv.org/abs/1803.10704)                         |
|     Uncertainty Weighting (`uw`)      | [Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics](https://arxiv.org/pdf/1705.07115v3.pdf) |
| Improvable Gap Balancing v1 (`igbv1`) |                                                     [Improvable Gap Balancing for Multi-Task Learning](https://arxiv.org/pdf/2307.15429)                                                     |
| Improvable Gap Balancing v2 (`igbv2`) |                                                    [Improvable Gap Balancing for Multi-Task Learning](https://arxiv.org/pdf/2307.15429)                                                     |


| Gradient Balancing Method (code name) |                                          Paper (notes)                                           |
|:-------------------------------------:|:------------------------------------------------------------------------------------------------:|
|             MGDA (`mgda`)             |     [Multi-Task Learning as Multi-Objective Optimization](https://arxiv.org/abs/1810.04650)      |
|           PCGrad (`pcgrad`)           |           [Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782)           |
|           CAGrad (`cagrad`)           | [Conflict-Averse Gradient Descent for Multi-task Learning](https://arxiv.org/pdf/2110.14048.pdf) |
|            IMTL-G (`imtl`)            |       [Towards Impartial Multi-task Learning](https://openreview.net/forum?id=IMPnRXEWpvr)       |
|         Nash-MTL (`nashmtl`)          |        [Multi-Task Learning as a Bargaining Game](https://arxiv.org/pdf/2202.01017v1.pdf)        |
|         FairGrad (`fairgrad`)          |        [FairGrad: Fairness Aware Gradient Descent](https://arxiv.org/pdf/2206.10923)        |
|         GradNorm (`gradnorm`)          |        [GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks](https://arxiv.org/pdf/1711.02257)        |
|         FAMO (`famo`)          |        [FAMO: Fast Adaptive Multitask Optimization](https://arxiv.org/pdf/2306.03792) |
|         DiBS-MTL (`dibsmtl`)          |        (1-Step DiBS-MTL)        |
|         Multi-Step DiBS-MTL (`multi_step_dibsmtl`)          |        (T-Step DiBS-MTL)        |


## Acknowledgements / Attribution

This codebase is built on top of the implementation from:

- Yanqi Dai, Nanyi Fei, and Zhiwu Lu. *Improvable Gap Balancing for Multi-Task Learning.*  
  In **Uncertainty in Artificial Intelligence (UAI)**, PMLR, 2023.  
  Repository: https://github.com/YanqiDai/IGB4MTL

We retain the original experiment structure and extend it with additional benchmarks (cityscapes, QM9) and baselines (FAMO, FairGrad, GradNorm) in addition to the DiBS-MTL methods. If you use this repository, please also cite the original IGB work (in addition to citing our paper).
