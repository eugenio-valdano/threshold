# Computing the Epidemic Threshold on Temporal Networks
Provides **Python** tools for computing the epidemic threshold on temporal network, as explained in paper

[**Analytical Computation of The Epidemic Threshold on Temporal Networks**](http://journals.aps.org/prx/abstract/10.1103/PhysRevX.5.021005)

Valdano E, Ferreri L, Poletto C, Colizza V, *Phys Rev X* 5, 021005 2015.

**When you use this code, please cite the above reference.**

Further details on terms of use: see LICENSE

## Content
- `test_system.py` checks if your system has all the needed libraries.
- `threshold.py` main module.
- `threshold_util.py` additional methods for network handling.


## Required external modules
- `numpy`
- `scipy`
- `networkx`
- `pandas` (for `threshold_util.py`)

Run `test_system.py` to check if you have everything you need.

# Overview

The package consists of two objects: the class `tnet` for uploading and managing the temporal network, and the class `threshold`, for the actual computation of the threshold.

## import 

The directory containing `threshold.py` must be in your Python search path. You can temporarily add it using

```python
from sys import path
path.append('<dir to threshold.py>')
```

Then actually import the module as, for instance,

```python
import threshold as thr # main module
import threshold_util as thu # additional utils
```

## `tnet`: manage your temporal network

Class `tnet` is able to load a temporal network given in different formats:

- path to a text file containing the whole edge list. First two columns represent edges' origin and destination, while last column is the time stamp. Time stamps are assumed to be integers from 0. If there are more than 3 columns, then 3rd column is interpreted as edge weight. Further columns between the 3rd and the last (time) are disregarded. Default separator is `\t`; different separators (e.g. `separator=','`) can be input via the optional keyword `separator` in the `tnet` constructor. By default the edge list is assumed undirected; this can be changed via the optional keyword `directed` in the `tnet` constructor.
- (Python) list of networkx `Graph` or `DiGraph` objects. If the network is weighted, weights must be assigned to edges as `weight` keywords.

The network can then be loaded in class `tnet` as follows:

`R = thr.tnet(my_network)`


### Arguments for `tnet`, with their default values

- `my_network`: where to look for the network, according to supported formats (see above);
- `period = None`: set period like this, if only a part of the network is to be used, up to period `T` (less than the one inferred from time stamps);
- `dtype = 'float128'`: the bit length of the used float. 'float128' is the default because it is often needed. Every string that is not `'float64'` is interpreted as `'float128'`.

##### other optional keywords
- `directed`: it may be used when loading from text file. If `directed=True`, then the edge list is assumed to be directed. If not specified, treated as `directed=False`. When loading from a list of `networkx` graphs, it inherits from them the fact of being (un)directed.
- `attributes=None`: with this keyword you can provide a dictionary for assigning node attributes. Imagine your nodes are people, you could set `attributes={'id1':'male','id2':'female'}`. The dictionary does not have to be exhaustive. Nodes without attribute are allowed.
- `separator`: it may be used when loading from text file, to specify the separator. If not specified, treated as `separator='\t'`.

### Attributes
|  name  |  description |
|---|---|
| `N`  |  number of nodes.  |
| `T`  |  period. You can manually reduce it. It will drop the time steps in excess from the end.  |
| `weighted`  |  `True/False`  |
| `lG`  |  list of `networkx` graphs  |
| `lA`  |  list of adjacency matrices in `scipy.sparse.csr_matrix` format  |
| `attributes`  |  node attributes  |
| `nodelist`  |  list of nodes  |


## `threshold`: compute the threshold

Intstantiate a `threshold` object like this:

```python
myth = th.threshold(X)
```
 Where `X` can be either a `tnet` object or a `list` of adjacency matrices in `scipy.sparse.csr_matrix`. Additional optional arguments are
 
##### related to power method:
 
 - `eval_max=20000`: maximum number of eigenvalue evaluations.
 - `tol=1e-6` : tolerance for power method convergence.
 - `store=10` : number of eigenvector(value) values to use to check convergence.
 - `convergence_on_eigenvector=True`. If `True` uses the algorithm that checks convergence on the L1 norm of the principal eigenvector (probably more accurate). If `False`, checks the convergence of the eigenvalue estimate itself.

##### related to the temporal network:
 
 - `weighted=None`. You have to specify it when you provide a list of adjacency matrices instead of a `tnet` object. You can specify it also with a `tnet` object if you want to override the `.weighted` attribute of the `tnet` object. If the network itself is weighted, you still can set `weighted=False` here. It simply means it multiplies transmissibility directly to the adjacency matrices. To know more about weights, read [this article](http://epjb.epj.org/articles/epjb/abs/2015/12/b150620/b150620.html). `weighted=False` is more time-efficient than `weighted=True`.
 - `attributes=None`. It is ignored when `X` is a `tnet` object, as it will inherit the attributes from `X`. When `X` is a list of matrices, you can use this to provide a **`list`** of length `N` containing the attribute of each node. If you do not wish to set an attribute for node `i`, put `None` in the `list` at place `i`.

You can access and edit `eval_max`,  `tol`, `store` and `weighted` as class attributes.

The class has also the attribute `convergente_on` which is either `eigenvector` or `eigenvalue`. You can access it and edit it.

For instance:

``` python
myth.tol = 1e-5
myth.convergence_on = 'eigenvalue'
```

The class has the attribute `lA` which is the list of adjacency matrices. You can access it and set it safely.

Finally, the attribute `avg_k` returns the average (weighted) degree of the network, i.e., `\frac{\sum_{t=1}^T\sum_{i,j}A_{t,ij}}{NT}`

### `compute` method

This carries out the actual computation of the threshold.

```python
x = th.compute(mu, vmin=1e-3, vmax=1, maxiter=50, root_finder='brentq', **kwargs)
```

- `mu` is the only compulsory argument. It can be either a single value (recovery probability) or a dictionary having a recovery probability for every attribute: `{'attr 1': 0.1, 'attr 2': 0.3, 'default':0.6}`. It must always have a 'default' value, which will be assigned to nodes with no attribute.
- `vmin` and `vmax` are the boundaries of the intervals in which to look for the threshold.
- `maxiter` is the maximum number of iterations of the root finding algorithm.
- `root_finder` can be either `'brentq'` or `'bisect'`, referring to the functions in `scipy.optimize`. For further details see, for instance, [scipy documentation]( http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html#scipy.optimize.brentq ).
- Other keyword arguments are directly sent to the root finding scipy function (e.g. `xtol` and `rtol`).

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

