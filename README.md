# Computing the Epidemic Threshold on Temporal Networks
A **Python** library for computing the epidemic threshold on temporal network, as explained in paper

[**Analytical Computation of The Epidemic Threshold on Temporal Networks**](http://journals.aps.org/prx/abstract/10.1103/PhysRevX.5.021005)

Valdano E, Ferreri L, Poletto C, Colizza V, *Phys Rev X* 5, 021005 2015.

**When you use this code, please cite the above reference.**

Further details on terms of use: see LICENSE

# Intro

This version has been restructured using the Python module `setuptools`. This means that the library will install just like one you would get through `pip`.

The package will look for the needed dependencies, and try to install them if necessary, through pip. However, there can be issues sometimes. For instance, if you're using [Anaconda](https://www.continuum.io/) (and **I recommend you do** use it, especially if you're starting with Python), there are conflicts, as packages are usually installed through `conda`. So it is always **better to check beforehand if you have the packages needed**. They are listed in `requirements.txt`, along with their versions.

# Install

* Download the entire directory,
* `cd` inside it (where `setup.py` is),
* execute `python setup.py test`. You should get an output like this:

```
running test
running egg_info
writing requirements to Epidemic_Threshold.egg-info/requires.txt
writing Epidemic_Threshold.egg-info/PKG-INFO
writing top-level names to Epidemic_Threshold.egg-info/top_level.txt
writing dependency_links to Epidemic_Threshold.egg-info/dependency_links.txt
reading manifest file 'Epidemic_Threshold.egg-info/SOURCES.txt'
reading manifest template 'MANIFEST.in'
writing manifest file 'Epidemic_Threshold.egg-info/SOURCES.txt'
running build_ext
copying build/lib.macosx-10.6-x86_64-2.7/threshold/utilc.so -> threshold
test_comput (tests.1.myTest) ... threshold: 0.045617
ok
test_upload (tests.1.myTest) ... ok
test_comput (tests.cython.myTest) ... threshold with CYTHON: 0.045945
ok
```
* execute `python setup.py install`

## Cython
Some functions exists both in pure Python and in [Cython](http://cython.org/). Cython translates these functions into C, greatly increasing peformance. When installing, the program tries to understand if you have what Cython needs (the Cython module, a C compiler) and if so, you will have both versions: pure Python and C. If you do not have what Cython needs, the program will install only the Python versions. A keyword in `threshold.threshold.threshold.compute` will let you choose between Python and Cython. You should choose `cython=False` if you want something more numerically stable, more versatile (only unweighted computation is implemented in Cython). You should choose `cython=True` if your concern is performance (large networks, and/or little time available).


# Load

At the beginning of your script, load the libraries like this:

```python
import threshold.threshold as thr
import threshold.threshold_util as thu
```

# Content
Module `threshold.threshold` contains two main classes:

* `tnet` (handles temporal networks),
* `threshold` (computes the threshold).

Module `threshold.threshold_util` contains some useful functions for converting formats. At this stage, it contains

* `DataFrame_to_lA` (from `pandas.DataFrame` to a list of `scipy.sparse.csr_matrix`),
* `DataFrame_to_lG` (from `pandas.DataFrame` to a list of `networkx.Graph` or `networkx.DiGraph`).

## `tnet`

The constructor has only one compulsory argument: `thr.tnet(my_network)`.

`my_network` can be

* **a path to a text file** containing the whole edge list. First two columns represent edges' origin and destination, while last column is the time stamp. Time stamps are assumed to be integers from 0. If there are more than 3 columns, then 3rd column is interpreted as edge weight. Further columns between the 3rd and the last (time) are disregarded. Default separator is `\t`; different separators (e.g. `separator=','`) can be input via the optional keyword `separator` in the `tnet` constructor. By default the edge list is assumed undirected; this can be changed via the optional keyword `directed` in the `tnet` constructor.
* a **list of `networkx.Graph` or `networkx.DiGraph` objects**. If the network is weighted, weights must be assigned to edges as `weight` keywords.

Other optional keywords of the `thr.tnet` constructor are

* `period` (default `None`): if not `None`, will override the computation of the period resulting from the input, by taking the first `period` snapshots. For example, if `my_network` is a list of 20 `networkx.Graph` snapshots, and `period=15`, then the last 5 snapshots are discarded. If the specified `period` is longer than the period of the dataset, you will get an `AssertionError`.
* `dtype` (default `float128`): it is a `str` argument. Set it to `float64` if you really want to use 64-bit floating point numbers. Any other value of `dtype`, including the default one, will lead to using 128-bit floating points.
* `attributes` (default `None`): `None`, or a `dict` with node IDs as keys, and arbitrary node attributes as attributes.

`thr.tnet` has the following members, accessible through `@property` decorator syntax (some them can be manually set):

* `lA`: list of `scipy.sparse.csr_matrix` adjacency matrices,
* `lG`: list of `networkx` graphs,
* `weighted`: boolean,
* `N`: number of nodes,
* `T`: number of time steps. _It can be set_,
* `nodelist`: list of all nodes
* `attributes`: returns list of attributes, ordered as `nodelist`. _Can be set by providing a `dict`_,

For instance this works
```
R = thr.tnet(my_network)
print R.T # say we get 15
R.T = 10 # set it to 10
print R.T # now we get 10. The last 5 snapshots have been discarded.
```

If you try to set members other than the _settable_ ones, the program will simply tell you you can't do it.

## `threshold`

The constructor has again one compulsory argument: `thr.threshold(my_network)`, where `my_network` can be

* A `tnet` object,
* A list of `scipy.sparse.csr_matrix` objects.

Other keyword arguments:

* `eval_max=20000, tol=1e-6, store=10`: parameters of the _modified power methods_
* `additional_time` (default 0): It allows to add an arbitrary number of empty snapshots (empty means no edges in them). This is a convenient way to do it, as the order of empty snapshots inside the sequence does not matter.
* `weighted` (default `None`). **This is important**: the meaning of `weighted` is different in `thr.tnet` and in `thr.threshold`. In the former it means if the edges have weights or not. In the latter it refers to the way the infection propagator is computed. When `weighted=False` in `threshold`, the _t_-th term in the infection propagator is $1-\mu + \lambda A(t)$ regardless of the nature of $A(t)$, which is itself binary if `tnet.weighted` is `False`, or real-valued otherwise. If `threshold.weighted` is instead `True`, then binomial transmission is assumed, so that the _t_-th term in the infection propagator is $1-\mu + [1-(1-\lambda)^E(t)]$, with $E(t)$ being the entry-wise exponential of $A(t)$. Hence, when `tnet.weighted` is `False`, the result is the same regardless of `threshold.weighted`. However, the algorithm is **much** faster when `threshold.weighted` is `False`. Despite the difference between `weighted` in the two classes, if `my_network` is a `tnet` object and `weighted=None`, then `thr.threshold` will inherit the `weighted` attribute from `my_network`. `weighted=True/False` overrides the inheritance. If `my_network` is a list of matrices, `weighted` must be **explicitly** set to either `True` or `False`. **If all this gives you headache, always set `weighted=False` in `thr.threshold`.**
* `convergence_on_eigenvector` (default `True`) check the convergence of the algorithm on the stability of the eigenvector, rather than the eigenvalue (recommended).
* `attributes` (default `None`): see `tnet`. Inherited from `tnet` when applicable.
* `cython` (default `False`): in addition to pure Python, the _power method_ algorithm is implemented in [Cython](http://cython.org/), in order to make it faster. **Cython requires a C compiler, which must be present on your machine**.

This class hass many methods/variables (`@property` style) you can access and (sometimes) set. They are (when not explained, similar to `tnet`'s, or repetitions of the keywords of the constructor)

* `N`
* `T`
* `avg_k`: time average of the average degree of the snapshots
* `avg_A`: time-averaged adjacency matrix
* `avg_sr`: spectral radius of the time-averaged adjacency matrix
* `weighted`
* `convergence_on`
* `eval_max`, `tol`, `store`
* `additional_time`
* `lA`
* `l_indptr`, `l_indices`, `l_data`, `l_place` (see doc of `scipy.sparse.csr_matrix` for some of them)

`thr.threshold` has two methods:

### `thr.threshold.compute`
This function computes the threshold, using optimization algorithms in [`scipy.optimize`](https://docs.scipy.org/doc/scipy/reference/optimize.html). It needs one compulsory argument: `mu`.
`mu` can be either a (floating point) number, in which case it is interpreted as the recovery probability, or a `dict`. This `dict` must have attributes as keys (the same node attributes of the network), pointing to their corresponding values of recovery probability. This implements **heterogeneous recovery rates**. Optional keyword arguments are

* `vmin` ( default 1e-3), `vmax` (default 1) : range of transmissibility in which to look for the threshold
* `root_finded` (default 'brentq'). It can be either 'brentq' or 'bisect'
* `maxiter` (default 50) and arguments inherited from `thr.threshold.__init__`: see documentation of `scipy.optimize`

### `thr.threshold.sr_point`
Computes one point of the spectral radius. Its arguments are

* transmissibilty
* `mu` (see above)
* other optional keywords (see above)



# Minimal example

```python
# import threshold modules
import threshold.threshold as thr
import threshold.threshold_util as thu

# import additional modules
import networkx as nx
import numpy as np

# create a sequence of ER random graphs
N,T = 500,400
lG = []
for t in range(T):
    lG.append(nx.gnm_random_graph(N,N))

# load it as a tnet object, and print
R = thr.tnet(lG)
print R

# threshold object, and print
Z = thr.threshold(R)
print Z

# compute the threshold
mu = 0.01
lc = Z.compute(mu,vmin=0.003,vmax=0.005)
print mu, lc

```


## `threshold_util`

This module contains two functions: `DataFrame_to_lG` and `DataFrame_to_lA`. They turn a `pandas.DataFrame` object into a list of `networkx` graphs or `scipy.sparse` CSR matrix. The former is a suitable input for `threshold.tnet`, the latter for `threshold.threshold`.

### `DataFrame_to_lG`

```python
lG = thu.DataFrame_to_lG(df, directed=False, weight=None, source='source', target='target', time='time')
```

- `df` is a `pandas.DataFrame`.
- `directed` bool variable about (un)directedness.
- `source` name of the column of source nodes.
- `target` name of the column of target nodes.
- `time` name of the column with timestamps.
- `weight` can be `None` (unweighted network) or a string with the name of the column to be interpreted as weights.

It returns a list of `networkx` `Graph` or `DiGraph` objects.

### `DataFrame_to_lA`

**Assumes node id's are integers from 0 to N-1**, where N is the number of nodes.

```python
lA = thu.DataFrame_to_lA(df, directed=False, source='source', target='target', time='time', weight='weight', dtype=np.float128, force_beg=None, force_end=None)
```

- `df` is a `pandas.DataFrame`.
- `directed` bool variable about (un)directedness.
- `source` name of the column of source nodes.
- `target` name of the column of target nodes.
- `time` name of the column with timestamps.
- `weight` can be `None` (unweighted network) or a string with the name of the column to be interpreted as weights.
- `force_beg` if not `None`, will discard all timesteps smaller than this.
- `force_end` if not `None`, will discard all timesteps larger than this.

