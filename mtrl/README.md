# DiBS MTRL

[![CircleCI](https://circleci.com/gh/facebookresearch/mtrl.svg?style=svg&circle-token=8cc8eb1b9666a65e27a21c39b5d5398744365894)](https://circleci.com/gh/facebookresearch/mtrl)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/facebookresearch/mtrl/blob/main/LICENSE)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Zulip Chat](https://img.shields.io/badge/zulip-join_chat-brightgreen.svg)](https://mtenv.zulipchat.com)

---

## Overview  
This repository contains the MTRL experiments used in **"DiBS-MTL: Transformation-Invariant Multitask Learning with Direction Oracles".**

This codebase builds on the [MTRL Repository](https://github.com/facebookresearch/mtrl).  
Our experiments are performed on [MetaWorld Benchmarks](https://github.com/Farama-Foundation/Metaworld).

**Python version:** 3.6.13  

---

## Table of Contents  
- [License](#license)  
- [Citing MTRL](#citing-mtrl)  
- [Setup](#setup)  

---

## License  
- MTRL uses the [MIT License](https://github.com/facebookresearch/mtrl/blob/main/LICENSE).  
- [Terms of Use](https://opensource.facebook.com/legal/terms)  
- [Privacy Policy](https://opensource.facebook.com/legal/privacy)  

---

## Citing MTRL  
If you use MTRL in your research, please cite it using the following BibTeX entry:

```bibtex
@Misc{Sodhani2021MTRL,
  author       = {Shagun Sodhani and Amy Zhang},
  title        = {MTRL - Multi Task RL Algorithms},
  howpublished = {Github},
  year         = {2021},
  url          = {https://github.com/facebookresearch/mtrl}
}
```

---

## Setup (Conda)

> **Tested Python:** 3.6.13

### Quick Commands

1. **Clone the repository**
   ```bash
   git clone <this_repo_url>
   cd <this_repo>
   ```

2. **Create and activate a conda environment**
   ```bash
   conda create -n dibs-mtrl python=3.6.13 -y
   conda activate dibs-mtrl
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements/dev.txt
   pip install -r requirements.txt
   ```

4. **Install MetaWorld (pinned commit, editable)**
   ```bash
   git clone https://github.com/Farama-Foundation/Metaworld.git
   cd Metaworld
   git checkout d9a75c451a15b0ba39d8b7a8b6d18d883b8655d8
   pip install -e .
   cd ..
   ```

5. **Run methods**
   ```bash
   # From the repo root
   bash run_{method}.sh
   ```

6. **Toggle Transforms**

To enable or disable task transformations, update the `transform_flag` parameter in these files:

- `mtrl/config/experiment/metaworld.yaml`  
- `mtrl/tests/experiment/utils.py`

Currently, the only included transform is an **exponential transformation applied to the Window-Open task**.  
You can inspect or modify the transform implementation in:

- `mtrl/mtrl/env/vec_env_transformed.py`  
  (additional transformation equations are listed in the comments)

After editing the `transform_flag`, re-run your experiment to apply the new transform settings.

