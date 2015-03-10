# Computation of the Epidemic Threshold on Temporal Networks

Reference: Valdano E, Ferreri L, Poletto C, Colizza V (2015) Analytical Computation of The Epidemic Threshold on Temporal Networks, Phys Rev X

## Built For
- Python 2.7
- numpy 1.9.0
- scipy 0.14.0
- networkx 1.9.1

## Load network

```
from threshold import tnet,find_threshold # importing objects

# threshold must be in python search path.
# If not:
# from sys import path
# path.append('<dir to threshold>')

R = tnet(my_network) # initialize the object tnet
```

### Optional arguments for `tnet`

- `extension = 'npy'`: must be specified when using directory to files as input (see below)
- `period = None`: period of the network. Compulsory for certain file formats. If given, overrides other period computations
- `nodes = None`: number of nodes in the networks. Compulsory for certain file formats. If given, overrides other number of nodes computations
- `weighted = False`: whether the network must be treated as weighted
- `directed = False`: Useful only if I give edge list and want to treat is as directed. Otherwise assumes matrices as they are, not looking if they are sym or not
- `priority = 'sparse'`: can be 'dense' or 'sparse' . Specifies if, when istantiated, the network is stored as sparse or dense adjacency matrices
- `dtype = np.float64`: the numpy data type
- `nodense = True`: prevents from ever computing the dense version of the adj matrices. Useful for large networks.

### supported formats for network initialization:
- `my_network = '<dirtoname>/<name>_t_'` ; `extension='npy'`
- `my_network = '<dirtoname>/<name>'` ; `extension='npz'`
- `my_network = '<dirtoname>/<name>_t_'` ; `extension='txt'` : edge list with columns separated by whitespace or `\t`. Columns represent, in order, source id, destination id and weight. 
- `my_network = `python list of networkx graphs
- `my_network = `python list of dense arrays
- `my_network = `python list of sparse arrays (`scipy.sparse.csr_matrix` objects)

## Compute threshold

**compute the threshold for network R (stored as tnet object) and recovery probability mu:**
```
threshold = find_threshold( mu , R )
```

### Optional arguments for `find_threshold`

- `method = 'power'`: can be `'direct'` (compute spectral radius using `scipy.linalg.eigvals`) or `'power'` (modified version of power method for computing spectral radius).
- `weighted = False`: whether to treat the network as weighted ( to know how weights are treated, see Supporting Information of ref Valdano et al)
- `findroot = 'brentq'`: can be `'bisect'` or `'brentq'`, referring to respective functions in `scipy.optimize`