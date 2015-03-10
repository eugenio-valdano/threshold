# Computation of the Epidemic Threshold on Temporal Networks

Reference: Valdano E, Ferreri L, Poletto C, Colizza V (2015) Analytical Computation of The Epidemic Threshold on Temporal Networks, Phys Rev X

## Built For
- Python 2.7
- numpy 1.9.0
- scipy 0.14.0
- networkx 1.9.1

```
from threshold import tnet,find_threshold # importing objects

# threshold must be in python search path.
# If not:
# from sys import path
# path.append('<dir to threshold>')

R = tnet(my_network) # initialize the object tnet
```
