# Total Variation Optimization Layers for Computer Vision

### CVPR 2022 ([PDF](https://arxiv.org/abs/2204.03643))
[Raymond A. Yeh](https://www.raymond-yeh.com/)<sup>1</sup> ,
[Yuan-Ting Hu](https://sites.google.com/view/yuantinghu),
[Zhongzheng Ren](https://jason718.github.io/),
[Alexander G. Schwing](http://www.alexander-schwing.de/)<br/>
Toyota Technological Institute at Chicago<sup>1</sup><br/>
University of Illinois at Urbana-Champaign <br/>

# Overview
This repository contains code for Total Variation Optimization Layers for Computer Vision accepted at CVPR 2022.

If you used this code or found it helpful, please consider citing the following paper:

<pre>
@inproceedings{YehCVPR2022,
               author = {R.~A. Yeh and Y.-T. Hu and Z. Ren and A.~G. Schwing},
               title = {Total Variation Optimization Layers for Computer Vision},
               booktitle = {Proc. CVPR},
               year = {2022},
}
</pre>

## Setup Dependencies
To install the dependencies, run the following
```bash
conda create -n tv_opt python=3.7
conda activate tv_opt
conda install conda-build
conda install pytorch=1.9.0 torchvision torchaudio cudatoolkit=11.1.74 -c pytorch -c nvidia
conda install -c anaconda scipy
pip install prox-tv
cd tv_layers_for_cv
conda develop .
```
We have tested these instructions with Ubuntu 20.04 using GCC 9.3.0.

## Compile CUDA Extensions
Make sure to use a GCC version 4.9 or above
```bash
cd tv_opt_layers/helpers/cuda/
python setup.py install
```

## Run Tests
```bash
python -m unittest discover tests/
```

## Demo
We illustrate how to use our TV layer in a juypter-notebook (`demo/tv_2d_illustration.ipynb`). To run the notebook, one will need to install a few more dependices.
```bash
conda install -c conda-forge notebook
conda install -c conda-forge nb_conda_kernels
pip install opencv-python
conda install -c conda-forge matplotlib
jupyter-notebook demo/tv_2d_illustration.ipynb
```

## Acknowledgements
We thank [proxTV](https://github.com/albarji/proxTV) and [carpet](https://github.com/hcherkaoui/carpet) for open sourcing their implementation, which we referred to during the development of this codebase.
