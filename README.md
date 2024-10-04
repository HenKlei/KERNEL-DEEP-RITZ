```
# ~~~
# This file is part of the paper:
#
#           " Kernel Methods in the Deep Ritz framework:
#                       Theory and practice "
#
#   https://github.com/HenKlei/KERNEL-DEEP-RITZ.git
#
# Copyright 2024 all developers. All rights reserved.
# License: Licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
# Authors:
#   Hendrik Kleikamp, Tizian Wenzel
# ~~~
```

# Solution of elliptic PDEs via kernel methods
In this repository, we provide the code used for the numerical experiments in the paper "Kernel Methods in the Deep Ritz framework: Theory and practice" by Hendrik Kleikamp and Tizian Wenzel.

You find the preprint [here](https://arxiv.org/abs/TBA).

## Installation
On a system with `git` (`sudo apt install git`), `python3` (`sudo apt install python3-dev`) and
`venv` (`sudo apt install python3-venv`) installed, the following commands should be sufficient
to install the `kernelDR` package with all required dependencies in a new virtual environment:
```
git clone https://github.com/HenKlei/KERNEL-DEEP-RITZ.git
cd KERNEL-DEEP-RITZ
./_venv_setup.sh
source venv/bin/activate
pip install -e .
```

## Running the experiments
To reproduce the results, we provide the original script creating the results presented in
the paper in the directory [`kernelDR/experiments/`](kernelDR/experiments/).

## Questions
If you have any questions, feel free to contact us via email at <hendrik.kleikamp@uni-muenster.de>.
