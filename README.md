# Computing the Epidemic Threshold on Temporal Networks
Provides tools for computing the epidemic threshold on temporal network, as explained in paper

**(2015) Analytical Computation of The Epidemic Threshold on Temporal Networks**

Valdano E, Ferreri L, Poletto C, Colizza V, *Phys Rev X*, 2015.

## Required software
- Python 2.7
- numpy 1.9.0
- scipy 0.14.0
- networkx 1.9.1

Tested for the aforementioned versions. It may work with previous/successive versions, too.

## Overview

The package consists of two objects: the class `tnet` for uploading and managing the temporal network, and the function `find_threshold`, that does the job of computing the epidemic threshold.

## step 1: load network with `tnet`

`from threshold import tnet,find_threshold # importing objects`

In order for Python to find the library, file `threshold.py` needs to be in Python search path, which normally includes the working directory. Should the file be in the directory outside search path, the following code should be executed prior to the above line:


```
from sys import path
path.append('<directory containing file threshold.py>')
```

Class `tnet` is able to load a temporal network given in several formats:

- directory containing text files, each one containing the edge list of each time snapshot. **Nodes must be named from 0 to N-1**;
- directory containing (dense) adjacency matrices, each represented by a file `.npy` (`numpy` binary file);
- `.npz` archive (`numpy` binary archive), containing all adjacency matrices;
- (Python) list of `networkx` `Graph` or `DiGraph` objects;
- (Python) list of `numpy.array` objects (dense adjacency matrices);
- Python list of sparse arrays (`scipy.sparse.csr_matrix` objects) as adjacency matrices.

The network can then be loaded in class `tnet` as follows:

`R = tnet(my_network)`


### Arguments for `tnet`, with their default values

- `my_network`: where to look for the network, according to supported formats (see above). Detailed description of this variable is provided below;
- `extension = 'npy'`: must be specified when using directory to files as input (see below);
- `period = None`: period of the network. Compulsory for certain file formats. If given, overrides other period computations;
- `nodes = None`: number of nodes in the networks. Compulsory for certain file formats. If given, overrides other number of nodes computations;
- `weighted = False`: whether the network must be treated as weighted;
- `directed = False`: Useful only if you give edge list and want to treat is as directed. Otherwise assumes matrices as they are, not looking if they are symmetric or not;
- `priority = 'sparse'`: can be `'dense'` or `'sparse'` . Specifies whether, when istantiated, the network is stored as sparse or dense adjacency matrices;
- `dtype = np.float64`: the `numpy` data type. In order to avoid underflow, sometimes `numpy.float128` is needed;
- `nodense = True`: if `True`, prevents from ever computing the dense version of the adj matrices. Useful for large networks.

### variable `my_network`

- `my_network = '<dirtoname>/<name>_t_'` ; needs `extension='npy'`
- `my_network = '<dirtoname>/<name>'` ; needs `extension='npz'`
- `my_network = '<dirtoname>/<name>_t_'` ; needs `extension='txt'` : edge list with columns separated by whitespace or `\t`. Columns represent, in order, source id, destination id and weight. 
- `my_network = `Python list of networkx graphs
- `my_network = `Python list of dense arrays
- `my_network = `Python list of sparse arrays (`scipy.sparse.csr_matrix` objects)

## step 2: compute the threshold

compute the threshold for network R (stored as tnet object) and recovery probability mu:


`threshold = find_threshold( mu , R )`

#### Optional arguments for `find_threshold`, with their default values

- `method = 'power'`: can be `'direct'` (compute spectral radius using `scipy.linalg.eigvals`) or `'power'` (modified version of power method for computing spectral radius).
- `weighted = False`: whether to treat the network as weighted ( to know how weights are treated, see Supporting Information of ref Valdano et al)
- `findroot = 'brentq'`: can be `'bisect'` or `'brentq'`, referring to respective functions in `scipy.optimize`