# Computing the Epidemic Threshold on Temporal Networks
Provides tools for computing the epidemic threshold on temporal network, as explained in paper

[**Analytical Computation of The Epidemic Threshold on Temporal Networks**](http://journals.aps.org/prx/abstract/10.1103/PhysRevX.5.021005)

Valdano E, Ferreri L, Poletto C, Colizza V, *Phys Rev X* 5, 021005 2015.

**When you use this code, please cite the above reference.**

Further details on terms of use: see LICENSE


## Required software
- Python
- numpy
- scipy
- networkx

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
import threshold as th
```

## test your system

Check you have everything you need by executing

```python
th.test_system()
```
Hopefully you will get an output like this:

```
---------
---------
All required modules are present
---------
---------
MODULE          TESTED FOR      YOURS           
python          2.7.11          2.7.11           * Same version *
scipy           0.17.0          0.17.0           * Same version *
networkx        1.10            1.10             * Same version *
geopy           1.11.0          1.11.0           * Same version *
numpy           1.11.0          1.11.0           * Same version *
pandas          0.18.0          0.18.0           * Same version *
--------
--------
--------
--------
```

*tested for* simply means those are the versions we are using, it does not mean it will not work for other versions. Warning: it is unlikely to work with networkx 1.7 or older.

## `tnet`: manage your temporal network

Class `tnet` is able to load a temporal network given in different formats:

- path to a text file containing the whole edge list. First two columns represent edges' origin and destination, while last column is the time stamp. Time stamps are assumed to be integers from 0. If there are more than 3 columns, then 3rd column is interpreted as edge weight. Further columns between the 3rd and the last (time) are disregarded. Default separator is `\t`; different separators (e.g. `separator=','`) can be input via the optional keyword `separator` in the `tnet` constructor. By default the edge list is assumed undirected; this can be changed via the optional keyword `directed` in the `tnet` constructor.
- (Python) list of networkx `Graph` or `DiGraph` objects. If the network is weighted, weights must be assigned to edges as `weight` keywords.

The network can then be loaded in class `tnet` as follows:

`R = tnet(my_network)`


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
 
 - `weighted=None`. You have to specify it when you provide a list of adjacency matrices instead of a `tnet` object. You can specify it also with a `tnet` object if you want to override the `.weighted` attribute of the `tnet` object. (to know how weights are treated, see Supporting Information of ref Valdano et al.)
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