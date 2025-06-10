# Offline Changepoint Detection With Gaussian Processes

This folder contains the code associated with the paper *Offline Changepoint Detection With Gaussian Processes*. In the paper, we introduced Segmenting Changepoint Gaussian Process regression, or SegCPGP, a method for offline, Gaussian process regression-based changepoint detection. Essentially, SegCPGP combines Gaussian process regression with binary search to detect multiple changepoints with unknown locations.
SegCPGP builds upon [GPFlow](https://gpflow.github.io/GPflow/2.9.1/index.html).

The folder `cpgp` contains `segcpgp.py`, which contains an implementation of the SegCPGP algorithm, and `spectralmixture.py`, which contains an implementation of the spectral mixture kernel (originally from [here](https://github.com/DavidLeeftink/Spectral-Discontinuity-Design/blob/master/SpectralMixture.py)). The file `demo.ipynb` contains a few examples of SegCPGP's use on mean and trend change datasets.

### Installation

1. Create a virtual environment with Python version 3.11.7.
2. Install `requirements.txt` into the environment via pip.
3. Run `demo.ipynb`.

We tested the installation on MacOS and Linux (Fedora 40).

### Running SegCPGP

In principle, SegCPGP can be imported from `cpgp/segcpgp.py` as a class, which you can then call `.fit()` on as you would do with a `scikit-learn` model. The most minimal example would be the following:

```python
import numpy 
from cpgp.segcpgp import SegCPGP

# This example data does not contain a changepoint.
X = np.linspace(0, 40, 100).reshape(-1, 1)
y = np.random.normal(size=100).reshape(-1, 1)

segcpgp = SegCPGP(pval=0.05)
segcpgp.fit(X, y)
```

By default, SegCPGP uses a [constant kernel](https://gpflow.github.io/GPflow/2.9.1/api/gpflow/kernels/index.html#gpflow.kernels.Constant), but in principle any valid kernel function implemented as an extension of GPFlow's Kernel class can be used.

### Experiments

#### Synthetic/benchmark experiments.

Data is included for the *Benchmark* and *Synthetic* experiments in the paper, in data/benchmark and data/synthetic. The notebooks benchmark-experiment.ipynb and synthetic-experiment.ipynb provide code that runs SegCPGP with the RBF and 4-component spectral mixture kernel on the benchmark datasets and synthetic datasets described in our paper.

In the experiments, we used a pipeline running a SegCPGP instance on each dataset as a separate job on our local HPC cluster (via Slurm, using a fork of [abed](https://github.com/GjjvdBurg/abed). Hence, when you run these notebooks, the behavior of the random seed (which we set in segcpgp.py) will be different from the situation we had on our cluster - running the same job in parallel with random seed 42 is different from running a loop for 40 iterations using random seed 42. Thus, the results may differ from those in the paper, although it should still be possible to get some nice and comparable results via the notebooks.

The full repository, including pipeline code, will be available on Zenodo at a later date.

#### Likelihood ratio experiments

Code and data for the likelihood ratio experiments are included in notebooks/likelihood-ratio.ipynb. The experiments are i) computing and investigating the empirical distribution of the LR statistic under the null assumption and ii) comparing the FPR for SegCPGP to its significance level.

### Using our work

We would love it if you used our work. If you do, we would appreciate it if you cited us:

```
citation
```

If you are using our work and you have questions about the implementation, possible extensions, or if you just want to have a chat with us, mail janneke.verbeek at ru.nl
