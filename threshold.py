# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse import csr_matrix
import networkx as nx
from scipy.optimize import brentq,bisect
from sys import version_info
from scipy import __version__ as vers_scipy
from warnings import warn




class NetworkFormatError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)




class ThresholdError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)




def text_to_net (fname, **kwargs):
    if 'separator' in kwargs:
        separator = kwargs['separator']
    else:
        separator = '\t'
    
    if 'directed' in kwargs:
        directed = kwargs['directed']
    else:
        directed = False
    
    # Read file
    raw = open(fname).readlines()
    
    # Assess whether the graph is weighted
    if len(raw[0].strip().split(separator)) > 3:
        isw = True
    else:
        isw = False
    
    # Go through the file to compute the number of nodes and the period.
    snodes = set()
    stimes = set()
    for line in raw:
        azz = line.strip().split(separator)
        stimes.add(int(azz[-1]))
        snodes.add(azz[0])
        snodes.add(azz[1])
    N = len(snodes)
    T = max(stimes) + 1
    
    # Create empty graph objects.
    if directed:
        lG = [nx.DiGraph() for t in range(T)]
    else:
        lG = [nx.Graph() for t in range(T)]
    
    # Fill with edges
    if isw:
        for line in raw:
            azz = line.strip().split(separator)
            x, y, t = azz[0], azz[1], int(azz[-1])
            lG[t].add_edge(x, y, weight = int(azz[2]))
    else:
        for line in raw:
            azz = line.strip().split(separator)
            x,y,t = azz[0],azz[1],int(azz[-1])
            lG[t].add_edge(x,y)

    # Fill with all nodes
    snodes = list(snodes)
    for G in lG:
        G.add_nodes_from(snodes)
    
    # Return list of graphs and number of nodes
    return (lG, N, T)
    


# From networx Graph, create adjacency matrices as sparse CSR objects.
def graph_to_csr (lG,dtype):
    lAs = []
    for G in lG:
        A = nx.adjacency_matrix(G)
        lAs.append(csr_matrix(A, dtype=dtype))
    return lAs
    


# class for handling the temporal network        
class tnet:
    lG = None
    T = None
    N = None
    lA = None
    dtype = np.float64
    weighted = None
    
    # Class constructor. Additional optional keywords: directed (bool), separator (str).
    def __init__ (self, myn, period=None, dtype='float64', **kwargs ):
        
        # If dtype is different from 'float64', then np.float128 is set
        if dtype != 'float64': 
            self.dtype = np.float128
        
        # if: Path to file.
        if type(myn) == str: 
            self.lG, self.N, buT = text_to_net(myn, **kwargs )
            if self.T == None:
                self.T = buT
            else:
                assert self.T <= buT, 'Specified period is longer than dataset.'
        
        # else: list of graphs
        else:
            if not ( str(type(myn[0])) == "<class 'networkx.classes.graph.Graph'>" or str(type(myn[0])) == "<class 'networkx.classes.digraph.DiGraph'>" ): # networkx graph
                raise NetworkFormatError('Unsupported format: could not find neither networkx Graph nor DiGraph objects.')
            self.lG = myn
            
            if self.T == None:
                self.T = len(self.lG)
            else:
                assert self.T <= len(self.lG), 'Specified period is longer than dataset.'
            
            # Fill all graphs with all nodes
            snodes = set()
            if self.N == None:
                self.N = len(snodes)
            for G in self.lG:
                snodes |= set(G.nodes())
            snodes = list(snodes)
            for G in self.lG:
                G.add_nodes_from(snodes)
        
        
        # Check if weighted.
        ct = 0
        while len(self.lG[ct].edges()) == 0:
            ct += 1
        if 'weight' in self.lG[ct].edges(data=True)[0][2] : 
            self.weighted = True
        else:
            self.weighted = False
    
    
    def getMatrices (self):
        if self.lA == None:
            self.lA = []
            for G in self.lG:
                A = nx.adjacency_matrix(G)
                self.lA.append(csr_matrix( A,dtype=self.dtype ))
        return self.lA
    
    
    def __str__ (self):
        spoutp = 'N = %d; T = %d\n' % (self.N,self.T)
        spoutp += 'data type : %s\n' % str(self.dtype)
        
        # Directed.
        if str(type(self.lG[0])) == "<class 'networkx.classes.graph.Graph'>":
            spu = 'False'
        else:
            spu = 'True'
        spoutp += 'directed : ' + spu + '\n'
        
        # Weighted.
        t = 0
        while len(self.lG[t].edges()) == 0:
            t += 1
        if self.weighted:
            spu = 'True'
        else:
            spu = 'False'
        spoutp += 'weighted : ' + spu + '\n'
        #
        # Whether matrices are loaded.
        if self.lA == None: 
            spu = 'not loaded'
        else:
            spu = 'loaded'
        spoutp += 'adjacency matrices : ' + spu + '\n'
        return spoutp
    
    
    def __repr__ (self):
        return self.__str__()




# Now functions related to threshold computation.




# Compute the spectral radius using the modified power method.
def power_spectral_radius(ladda, mu, lA, N, T, valumax=1000, stabint=10, tolerance=0.0001):
    
    rootT = 1.0 / float(T)
    
    # Initialize
    leval = []
    v0 = 0.9*np.random.random(N) + 0.1
    v = v0.copy()
    vold = v.copy()
    interrupt = False # When convergence is reached, it becomes True.
    
    itmax = T * valumax
    for k in range(itmax):
        # Perform iteration:
        v = ladda*lA[k%T].dot(v) + (1.-mu)*v
        
        # Whether period is completed:
        if k%T == T-1:
            autoval = np.dot(vold,v)
            leval.append(autoval**rootT)
            
            # Check convergence
            if len(leval) >= stabint:
                fluct = ( max(leval[-stabint:]) - min(leval[-stabint:]) ) / np.mean(leval[-stabint:])
            else:
                fluct = 1. + tolerance
            if fluct < tolerance:
                interrupt = True
                break
            mnorm = np.linalg.norm(v)
            v = v / mnorm
            vold = v.copy()
    
    # If never interrupted, check now convergence.
    if not interrupt: 
        fluct = ( max(leval[-stabint:]) - min(leval[-stabint:]) ) / np.mean(leval[-stabint:])
        if fluct >= tolerance:
            raise ThresholdError, 'Power method did not converge.'
            
    return leval[-1] - 1.




# Compute the spectral radius using the modified power method, implementing transmission on weighted networks.
def power_spectral_radius_weighted(ladda, mu, lA, N, T, valumax=1000, stabint=10, tolerance=0.0001):

    loglad = np.log(1.-ladda)
    rootT = 1.0 / float(T)
    
    # Initialize
    leval = []
    v0 = 0.9*np.random.random(N) + 0.1
    v = v0.copy()
    vold = v.copy()
    interrupt = False # When convergence is reached, it becomes True.
    #
    itmax = T * valumax
    for k in range(itmax):
        
        # Perform iteration. Meaning of function expm1: -(loglad*lA[k%T]).expm1() = 1-(1-ladda)^Aij
        v = -(loglad*lA[k%T]).expm1().dot(v) + (1.-mu)*v
        
        # Whether period is completed
        if k%T == T-1: 
            autoval = np.dot(vold,v)
            leval.append( autoval**rootT )
            
            # Check convergence
            if len(leval)>=stabint: 
                fluct = (max( leval[-stabint:]) - min(leval[-stabint:]) ) / np.mean(leval[-stabint:])
            else:
                fluct = 1. + tolerance
            if fluct < tolerance:
                interrupt = True
                break
            mnorm = np.linalg.norm(v)
            v = v / mnorm
            vold = v.copy()
    
    # If never interrupted, check now convergence.
    if not interrupt: 
        fluct = ( max(leval[-stabint:]) - min(leval[-stabint:]) ) / np.mean(leval[-stabint:])
        if fluct >= tolerance:
            raise ThresholdError,'Power method did not converge.'
    return leval[-1] - 1.



# Function for computing the threshold. Additional optional keywords: weighted (bool), findroot (='brentq' or ='bisect').
def find_threshold (mu, R, vmin=0.001, vmax=0.999, maxiter=50, xtol=0.0001, **kwargs):
    
    if 'weighted' in kwargs:
        weighted = kwargs['weighted']
    else:
        weighted = R.weighted

    
    if 'rootFinder' in kwargs:
        findroot = kwargs['rootFinder']
    else:
        findroot = 'brentq'
    
    if findroot == 'brentq':
        rootFinder = brentq
    elif findroot == 'bisect':
        rootFinder = bisect
    else:
        raise ThresholdError,'method for root finding '+findroot+' is not supported.'
    
    
    try:
        if weighted:
            result, rr = rootFinder(power_spectral_radius_weighted, vmin, vmax, args=(mu, R.getMatrices(), R.N, R.T), xtol=xtol, maxiter=maxiter, full_output=True, disp=False)
        else:
            result, rr = rootFinder(power_spectral_radius, vmin, vmax, args=(mu, R.getMatrices(), R.N, R.T), xtol=xtol, maxiter=maxiter, full_output=True, disp=False )
    except ThresholdError, err_string:
        print err_string
        return np.nan
    except ValueError:
        print  'ValueError: Interval may not contain zeros (or other ValueError). Value of spectral radius in vmax=', power_spectral_radius_weighted(vmax, mu, R.getMatrices(), R.N, R.T)
        return np.nan
    else:
        if not rr.converged:
            print 'Optimization did not converge.'
            return np.nan
    return result



    
# CHECK VERSIONS 
vers_python0 = '2.7.9'
vers_numpy0  = '1.9.2'
vers_scipy0  = '0.15.1'
vers_netx0   = '1.9.1'
 
vers_python = '%s.%s.%s' % version_info[:3]
vers_numpy  = np.__version__
vers_netx   = nx.__version__    

if vers_python != vers_python0:
    sp = 'This program has been tested for Python %s. Yours is version %s.' % (vers_python0, vers_python)
    warn(sp)
if vers_numpy != vers_numpy0:
    sp = 'This program has been tested for numpy %s. Yours is version %s. It is likely to work anyway.' % (vers_numpy0, vers_numpy)
    warn(sp)
if vers_scipy != vers_scipy0:
    sp = 'This program has been tested for scipy %s. Yours is version %s. It is likely to work anyway.' % (vers_scipy0, vers_scipy)
    warn(sp) 
if vers_netx != vers_netx0:
    sp = 'This program has been tested for scipy %s. Yours is version %s. It may not work if your networkx is version 1.7 or older.' % (vers_netx0, vers_netx)
    warn(sp)
    
    


# Stuff to be printed when executed as main.  
if __name__ == '__main__':
    print '---------'
    print '---------'
    print '---------'
    print '---------'
    print 'MODULE FOR COMPUTING THE EPIDEMIC THRESHOLD ON TEMPORAL NETWORKS.'
    print 'Based on Valdano E et al (2015) Phys Rev X.'
    print '---------'
    print 'import in your program like this: "from threshold import tnet,find threshold". For help read README.md'
    print '---------'
    print 'Required modules:'
    print 'Python:   tested for: %s.  Yours: %s'   % (vers_python0, vers_python)
    print 'numpy:    tested for: %s.  Yours: %s'    % (vers_numpy0, vers_numpy)
    print 'scipy:    tested for: %s. Yours: %s'    % (vers_scipy0, vers_scipy)
    print 'networkx: tested for: %s.  Yours: %s' % (vers_netx0, vers_netx)
    print '--------'
    print 'It may however work with different versions. It will not probably work with networkx 1.7 or older.'
    print '--------'
    print '--------'
    print '--------'
    print '--------'