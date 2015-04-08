# Computing the Epidemic Threshold on Temporal Networks
Provides tools for computing the epidemic threshold on temporal network, as explained in paper

[**Analytical Computation of The Epidemic Threshold on Temporal Networks**](http://journals.aps.org/prx/abstract/10.1103/PhysRevX.5.021005)

Valdano E, Ferreri L, Poletto C, Colizza V, *Phys Rev X* 5, 021005 2015.

**When you use this code, please cite the above reference.**


## Required software
- Python 2.7.9
- numpy 1.9.2
- scipy 0.15.1
- networkx 1.9.1

Tested for the the above versions. It may work with previous/successive versions, too. It is unlikely to work with networkx 1.7 or older.

## Overview

The package consists of two objects: the class `tnet` for uploading and managing the temporal network, and the function `find_threshold`, that does the job of computing the epidemic threshold.

## step 1: load network with `tnet`

`from threshold import tnet,find_threshold`

In order for Python to find the library, file `threshold.py` needs to be in Python search path, which normally includes the working directory. Should the file be in the directory outside search path, the following code should be executed prior to the above line:


```
from sys import path
path.append('<directory containing file threshold.py>')
```

Class `tnet` is able to load a temporal network given in different formats:

- path to a text file containing the whole edge list. First two columns represent edges' origin and destination, while last column is the time stamp. Time stamps are assumed to be integers from 0. If there are more than 3 columns, then 3rd column is interpreted as edge weight. Further columns between the 3rd and the last (time) are disregarded. Default separator is `\t`; different separators (e.g. `separator=','`) can be input via the optional keyword `separator` in the `tnet` constructor. By default the edge list is assumed undirected; this can be changed via the optional keyword `directed` in the `tnet` constructor.
- (Python) list of networkx `Graph` or `DiGraph` objects. If the network is weighted, weights must be assigned to edges as `weight` keywords.

The network can then be loaded in class `tnet` as follows:

`R = tnet(my_network)`


### Arguments for `tnet`, with their default values

- `my_network`: where to look for the network, according to supported formats (see above);
- `period = None`: set period like this, if only a part of the network is to be used, up to period `T` (less than the one inferred from time stamps);
- `dtype = 'float64'`: the bit length of the used float. In order to avoid underflow, sometimes `float128` is needed. Every string that is not `'float64'` is interpreted as `'float128'`.

##### other optional keywords
- `directed`: it may be used when loading from text file. If `directed=True`, then the edge list is assumed to be directed. If not specified, treated as `directed=False`;
- `separator`: it may be used when loading from text file, to specify the separator. If not specified, treated as `separator='\t'`.


## step 2: compute the threshold

compute the threshold for network R (stored as tnet object) and recovery probability mu:


`threshold = find_threshold( mu , R )`

### Optional arguments for `find_threshold`, with their default values

- `vmin=0.001`, `vmax=0.999`: range for the threshold;
- `maxiter=50`: maximum number of iterations of the modified power method algorithm;

##### other optional keywords
- `weighted`: whether to treat the network as weighted ( to know how weights are treated, see Supporting Information of ref Valdano et al). If not present, treat as weighted when the network is weighted, and unweighted when the network is unweighted.
- `findroot`: can be `'bisect'` or `'brentq'`, referring to respective functions in `scipy.optimize`. If not specified, treated as `findroot='brentq'`
- `xtol` and `rtol`: allow to set the absolute and relative precision of the root finding algorithm. For further details see, for instance, `scipy.optimize.brentq` [documentation]( http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html#scipy.optimize.brentq ).