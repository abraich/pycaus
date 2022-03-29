
**pycaus** is a python package for causal survival analysis and counterfactual classaification predection with [PyTorch](https://pytorch.org), built on the [torchtuples](https://github.com/havakv/torchtuples) package for training PyTorch models. 

The package contains implementations of various [survival models](#methods), some useful [evaluation metrics](#evaluation-criteria), and a collection of [event-time datasets](#datasets).
In addition, some useful preprocessing tools are available in the `pycox.preprocessing` module.

# Get Started

To get started you first need to install [PyTorch](https://pytorch.org/get-started/locally/).
You can then install **pycox** via pip: 
```sh
pip install pycaus
```


We recommend to start with [01_introduction.ipynb](https://nbviewer.jupyter.org/github/havakv/pycox/blob/master/examples/01_introduction.ipynb), which explains the general usage of the package in terms of preprocessing, creation of neural networks, model training, and evaluation procedure.


# SurvCaus
## Evaluation Criteria
# ClassCaus
## Evaluation Criteria
The following evaluation metrics are available with `pycox.evalutation.EvalSurv`.


# Simulated Datasets
A collection of datasets are available through the `pycox.datasets` module.
For example, the following code will download the `metabric` dataset and load it in the form of a pandas dataframe
```python
from pycox import datasets
df = datasets.metabric.read_df()
```



# Installation

**Note:** *This package is still in its early stages of development, so please don't hesitate to report any problems you may experience.* 

The package only works for python 3.6+.

Before installing **pycox**, please install [PyTorch](https://pytorch.org/get-started/locally/) (version >= 1.1).
You can then install the package with
```sh
pip install pycox
```
For the bleeding edge version, you can instead install directly from github (consider adding `--force-reinstall`):
```sh
pip install git+git://github.com/havakv/pycox.git
```

## Install from Source

Installation from source depends on [PyTorch](https://pytorch.org/get-started/locally/), so make sure a it is installed.
Next, clone and install with
```sh
git clone https://github.com/havakv/pycox.git
cd pycox
pip install .
```

# References

  \[1\] Håvard Kvamme, Ørnulf Borgan, and Ida Scheel. Time-to-event prediction with neural networks and Cox regression. *Journal of Machine Learning Research*, 20(129):1–30, 2019. \[[paper](http://jmlr.org/papers/v20/18-424.html)\]

 
