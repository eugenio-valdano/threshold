# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse import csr_matrix
import networkx as nx
from scipy.optimize import brentq,bisect
#from sys import version_info
#from scipy import __version__ as vers_scipy
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
def graph_to_csr (lG,dtype,nodelist):
    lAs = []
    for G in lG:
        A = nx.adjacency_matrix(G, nodelist=nodelist)
        lAs.append(csr_matrix(A, dtype=dtype))
    return lAs
    


# class for handling the temporal network        
class tnet(object):
    
    # Class constructor. Additional optional keywords: directed (bool), separator (str).
    def __init__ (self, myn, period=None, dtype='float128', attributes=None, **kwargs ):
        
        #self.lG = None
        #self.T = None
        #self.N = None
        #self.lA = None
        self._dtype = np.float64
        #self.weighted = None
        
        # If dtype is different from 'float64', then np.float128 is set
        if dtype != 'float64': 
            self._dtype = np.float128
        
        # if: Path to file.
        if type(myn) == str: 
            self._lG, self._N, buT = text_to_net(myn, **kwargs )
            if not hasattr(self,'_T'):#:self.T == None:
                self._T = buT
            else:
                assert self._T <= buT, 'Specified period is longer than dataset.'
            # set list of nodes
            self._nodelist = list(self._lG[0].nodes())
        
        # else: list of graphs
        else:
            if not ( str(type(myn[0])) == "<class 'networkx.classes.graph.Graph'>" or str(type(myn[0])) == "<class 'networkx.classes.digraph.DiGraph'>" ): # networkx graph
                raise NetworkFormatError('Unsupported format: could not find neither networkx Graph nor DiGraph objects.')
            self._lG = myn
            
            if not hasattr(self,'_T'):#self.T == None:
                self._T = len(self._lG)
            else:
                assert self._T <= len(self._lG), 'Specified period is longer than dataset.'
            
            # Fill all graphs with all nodes
            snodes = set()
            for G in self._lG:
                snodes |= set(G.nodes())
            if not hasattr(self,'_N'):#self.N == None:
                self._N = len(snodes)
            self._nodelist = list(snodes)
            for G in self._lG:
                G.add_nodes_from(self._nodelist)
        
        # if attributes are present, set list of attributes        
        if attributes is not None:
            self._attributes = []
            for x in self._nodelist:
                if x in attributes:
                    self._attributes.append(attributes[x])
                else:
                    self._attributes.append(None)
        
        # Check if weighted.
        ct = 0
        while len(self._lG[ct].edges()) == 0:
            ct += 1
        if 'weight' in self._lG[ct].edges(data=True)[0][2] : 
            self._weighted = True
        else:
            self._weighted = False
            
            
    ##
    ## PROPERTY lA
    ##
    @property
    def lA(self):
        if not hasattr(self,'_lA'):
            self._lA = graph_to_csr(self._lG, self._dtype, self._nodelist)
        return self._lA
        
    @lA.setter
    def lA(self,value):
        raise NotImplementedError, 'You cannot set adjacency matrices directly.'
        
    ##
    ## PROPERTY lG
    ##
    @property
    def lG(self):
        return [ G.copy() for G in self._lG ]
    
    @lG.setter
    def lG(self,value):
        raise NotImplementedError, 'You cannot set graphs directly.'
        
        
    ##
    ## PROPERTY weighted
    ##
    @property
    def weighted(self):
        return self._weighted
    
    @weighted.setter
    def weighted(self,value):
        raise NotImplementedError, 'You cannot change weighted status directly.'        
        
    ##
    ## PROPERTY N
    ##
    @property
    def N(self):
        return self._N
    
    @N.setter
    def N(self,value):
        assert False, 'You cannot manually edit the number of nodes.'
        
    ##
    ## PROPERTY T
    ##
    @property
    def T(self):
        return self._T
    
    @T.setter
    def T(self,value):
        
        assert value<=self._T, 'You can at most reduce period, not increase it.'
        
        # check that you are actually reducing the period, and not giving the same value
        if value < self._T:      
        
            self._T = value
            
            # reduce the number of time steps, and checks if it can throw away some nodes that now never activate
            self._lG = self._lG[:self._T]
            snodes = set([ x for G in self._lG for e in G.edges() for x in e ])            
            if len(snodes) < self._N:
                warn('Decreasing the period has resulted in decreasing the number of nodes in the network.')
                self._N = len(snodes)
                snodes = list(snodes)
                self._lG = [ nx.subgraph(G,snodes) for G in self._lG]
                # reset lA
                if hasattr(self,'_lA'):
                    del self._lA
            else:
                # clip lA
                if hasattr(self,'_lA'):
                    self._lA = self._lA[:self._T]
                    
    ##
    ## PROPERTY attributes
    ##
    @property
    def attributes(self):
        if hasattr(self,'_attributes'):
            return dict(zip(self._nodelist,self._attributes))
        else:
            return None
    
    @attributes.setter
    def attributes(self, d):
        self._attributes = []
        for x in self._nodelist:
            if x in d:
                self._attributes.append(d[x])
            else:
                self._attributes.append(None)
        
                
    
    def __str__ (self):
        spoutp = 'N = %d; T = %d\n' % (self._N,self._T)
        spoutp += 'data type : %s\n' % str(self._dtype)
        
        # Directed.
        if str(type(self._lG[0])) == "<class 'networkx.classes.graph.Graph'>":
            spu = 'False'
        else:
            spu = 'True'
        spoutp += 'directed : ' + spu + '\n'
        
        # Weighted.
        t = 0
        while len(self._lG[t].edges()) == 0:
            t += 1
        if self._weighted:
            spu = 'True'
        else:
            spu = 'False'
        spoutp += 'weighted : ' + spu + '\n'
        #
        # Whether matrices are loaded.
        if not hasattr(self,'_lA'):#self._lA == None: 
            spu = 'not loaded'
        else:
            spu = 'loaded'
        spoutp += 'adjacency matrices : ' + spu + '\n'
        # Whether attributes are present
        if hasattr(self,'_attributes'):
            spu = 'present'
        else:
            spu = 'not present'
        spoutp += 'attributes : ' + spu + '\n'
        return spoutp
    
    
    def __repr__ (self):
        return self.__str__()




# Now functions related to threshold computation.


# There are two kinds of algorithm: psr1 and psr2 (psr=power spectral radius). They're both weighted and unweighted, so psr1uw, psr1w, psr2uw, psr2w
# psr1 is more crude: it checks convergence on the value of the spectral radius itself.
# psr2 is checks convergence on the principal eigenvector

# PSR1

# unweighted
def psr1uw(ladda, mu, lA, N, T, valumax, tolerance, store):
    
    # parameters    
    #valumax = kwargs['valumax'] # 1000
    #tolerance = kwargs['tolerance'] # 1e-5
    #store = kwargs['store'] # 10
    
    rootT = 1.0 / float(T)
    # same data type as the adjacency matrices
    dtype = type(lA[0][0,0])
    
    # Initialize
    leval = np.empty(shape=(store,), dtype=dtype)
    ceval = 0
    v0 = 0.9*np.random.random(N) + 0.1
    v0 = np.array(v0, dtype=dtype)
    v = v0.copy()
    vold = v.copy()
    interrupt = False # When convergence is reached, it becomes True.
    
    itmax = T * valumax
    for k in range(itmax):
        # Perform iteration:
        v = ladda*lA[k%T].dot(v) + (1.-mu)*v
        
        # Whether period is completed:
        if k%T == T-1:
            #autoval = np.dot(vold,v)
            leval[ceval%store] = np.dot(vold,v)**rootT
            ceval += 1
            #leval.append(autoval**rootT)
            
            # Check convergence
            if ceval >= store:
                fluct = ( np.max(leval) - np.min(leval) ) / np.mean(leval)
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
        fluct = ( np.max(leval) - np.min(leval) ) / np.mean(leval)
        if fluct >= tolerance:
            raise ThresholdError, 'Power method did not converge.'
            
    return leval[-1] - 1.




# weighted
def psr1w(ladda, mu, lA, N, T, valumax, tolerance, store):
    
    # parameters    
    #valumax = kwargs['valumax'] # 1000
    #tolerance = kwargs['tolerance'] # 1e-5
    #store = kwargs['store'] # 10
    
    rootT = 1.0 / float(T)
    # same data type as the adjacency matrices
    dtype = type(lA[0][0,0])
    loglad = np.log(1.-ladda)
    
    # Initialize
    leval = np.empty(shape=(store,), dtype=dtype)
    ceval = 0
    v0 = 0.9*np.random.random(N) + 0.1
    v0 = np.array(v0, dtype=dtype)
    v = v0.copy()
    vold = v.copy()
    interrupt = False # When convergence is reached, it becomes True.
    
    itmax = T * valumax
    for k in range(itmax):
        
        # Perform iteration. Meaning of function expm1: -(loglad*lA[k%T]).expm1() = 1-(1-ladda)^Aij
        v = -(loglad*lA[k%T]).expm1().dot(v) + (1.-mu)*v
        
        # Whether period is completed
        if k%T == T-1: 
            leval[ceval%store] = np.dot(vold,v)**rootT
            ceval += 1
            
            # Check convergence
            if ceval >= store:
                fluct = ( np.max(leval) - np.min(leval) ) / np.mean(leval)
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
        fluct = ( np.max(leval) - np.min(leval) ) / np.mean(leval)
        if fluct >= tolerance:
            raise ThresholdError,'Power method did not converge.'
    return leval[-1] - 1.



# PSR2

# unweighted
def psr2uw(ladda, mu, lA, N, T, valumax, tolerance, store):
    
    # stop when the L1 norm of (v_{a}-v_{a-1}) is below tolerance, where a is the a-th loop on the period
    
    # parameters    
    #valumax = kwargs['valumax'] # 20000
    #tolerance = kwargs['tolerance'] # 1e-6
    #store = kwargs['store'] # 10
    
    rootT = 1.0 / float(T)
    # same data type as the adjacency matrices
    dtype = type(lA[0][0,0])
    
    # Initialize eigenvector register
    MV = np.empty(shape=(N,store), dtype=dtype)
    # first vector is a versor with the equal components:
    MV[:,0] = np.array([1./np.sqrt(N)]*N, dtype=dtype)
    # register counter
    c = 0
    # vector to work on:
    v = MV[:,c].copy()
    
    for k in range(T*valumax):
        # Perform iteration:
        v = ladda*lA[k%T].dot(v) + (1.-mu)*v
        
        # Whether period is completed:
        if k%T == T-1:
            
            # spectral radius
            sr = np.dot(MV[:,c%store],v)
            
            # normalize
            v = v/np.linalg.norm(v)
        
            # Compute tolerance, and return if reached:
            delta = np.sum( np.abs(MV[:,c%store]-v) )
            if delta < tolerance:
                # return spectral radius^(1/T) - 1, as usual.
                return sr**rootT - 1. #, v/np.linalg.norm(v)
            
            # increment index, and update storage
            c += 1
            MV[:,c%store] = v.copy()
            
    # if it goes out of the loop without returning the sr
    raise ThresholdError,'Power method did not converge.'
    
    
# weighted
def psr2w(ladda, mu, lA, N, T, valumax, tolerance, store):
    
    # stop when the L1 norm of (v_{a}-v_{a-1}) is below tolerance, where a is the a-th loop on the period
    
    # parameters    
    #valumax = kwargs['valumax'] # 20000
    #tolerance = kwargs['tolerance'] # 1e-6
    #store = kwargs['store'] # 10
    
    loglad = np.log(1.-ladda)
    rootT = 1.0 / float(T)
    # same data type as the adjacency matrices
    dtype = type(lA[0][0,0])
    
    # Initialize eigenvector register
    MV = np.empty(shape=(N,store), dtype=dtype)
    # first vector is a versor with the equal components:
    MV[:,0] = np.array([1./np.sqrt(N)]*N, dtype=dtype)
    # register counter
    c = 0
    # vector to work on:
    v = MV[:,c].copy()
    
    for k in range(T*valumax):
        # Perform iteration:
        v = -(loglad*lA[k%T]).expm1().dot(v) + (1.-mu)*v
        
        # Whether period is completed:
        if k%T == T-1:
            
            # spectral radius
            sr = np.dot(MV[:,c%store],v)
            
            # normalize
            v = v/np.linalg.norm(v)
        
            # Compute tolerance, and return if reached:
            delta = np.sum( np.abs(MV[:,c%store]-v) )
            if delta < tolerance:
                # return spectral radius^(1/T) - 1, as usual.
                return sr**rootT - 1. #, v/np.linalg.norm(v)
            
            # increment index, and update storage
            c += 1
            MV[:,c%store] = v.copy()
            
    # if it goes out of the loop without returning the sr
    raise ThresholdError,'Power method did not converge.'
    
    
    
    
class threshold(object):
    """
    Class for computing the epidemic threshold on a temporal network.
    """
    
    _df = {('eigenvalue',False):psr1uw, ('eigenvalue',True):psr1w, ('eigenvector',False):psr2uw, ('eigenvector',True):psr2w}
    
    # weighted can be None, True, False
    # attributes has to be given only if X is not a tnet
    def __init__(self, X, eval_max=20000, tol=1e-6, store=10, weighted=None, convergence_on_eigenvector=True, attributes=None):
        
        # check the type of input, and store accordingly
        if str(type(X)) == "<class 'threshold.tnet'>":
            # it's a tnet
            self._lA = X.lA
            self._N = X.N
            self._T = X.T
            self._weighted = X.weighted
            # in case i use the "weighted" keyword, override
            if weighted is not None:
                self._weighted = weighted
            if hasattr(X, '_attributes'):
                self._attributes = X.attributes
        elif type(X) == list:
            # it's a list of scipy sparse adj matrices (WARNING: IT DOES NOT CHECK THE TYPE INSIDE THE LIST)
            self._lA = X
            self._N = X[0].shape[0]
            self._T = len(X)
            assert weighted is not None, 'If you give me a list instead of a tnet object, you have to specify the keyword "weighted".'
            self._weighted = weighted
            if attributes is not None:
                self._attributes = attributes
        else:
            assert False, 'Input network not supported.'
            
        self._dtype = type(self._lA[0][0,0])
        
        
        self._args = (eval_max, tol, store)
            
        if convergence_on_eigenvector:
            self._on = 'eigenvector'
        else:
            self._on = 'eigenvalue'
        
        # Choose function to use (_f)
        self._f = self._df[(self._on,self._weighted)]
                
                
    # additional kwargs (like xtol or rtol) will be passed directly to the root finding routine. See scipy documentation
    def compute(self, mu2, vmin=0.001, vmax=1., maxiter=50, root_finder='brentq', **kwargs):
        
        # recovery rate(s)
        if type(mu2) == dict:
            # heterogeneous mu
            assert hasattr(self,'_attributes'), 'No attributes given.'
            mu = np.full((self._N,), mu2['default'], dtype=self._dtype)
            for i in range(self._N):
                if self._attributes[i] is not None:
                    mu[i] = mu2[self._attributes[i]]
            
        else:
            # homogeneous mu
            mu = mu2
        
        if root_finder.upper() == 'BRENTQ':
            findroot = brentq
        elif root_finder.upper() == 'BISECT':
            findroot = bisect
        else:
            raise ThresholdError,'method for root finding '+root_finder+' is not supported.'
            
        # compute threshold, and try all possible errors
        try:
            result, rr = findroot(self._f, vmin, vmax, args=(mu,self._lA,self._N,self._T)+self._args, maxiter=maxiter, full_output=True, disp=False, **kwargs)
            
            if not rr.converged:
                raise ThresholdError, 'Optimization did not converge.'
            
        # Error coming from spectral radius computation, or no convergence in root finding
        except ThresholdError, err_string:
            print err_string
            result = np.nan
            
        # Error: interval is likely not to contain zeros
        except ValueError:
            sr_min = self._f(vmin,mu,self._lA,self._N,self._T,*self._args)
            sr_max = self._f(vmax,mu,self._lA,self._N,self._T,*self._args)
            spuz = 'ValueError: Interval may not contain zeros (or other ValueError). f({:.5f})={:.5f}; f({:.5f})={:.5f}'.format(vmin, sr_min,vmax, sr_max)
            print spuz 
            result = np.nan

        finally:        
            return result
            
    
    # average degree (useful for boundaries)
    @property
    def avg_k(self):
        if not hasattr(self,'_avg_k'):
            self._avg_k = np.mean([np.mean(A.sum(axis=1)) for A in self._lA])
        return self._avg_k
        
    @avg_k.setter
    def avg_k(self,x):
        print 'You cannnot manually set the average degree.'
        
    @avg_k.deleter
    def avg_k(self):
        if hasattr(self,'_avg_k'):
            del self._avg_k
    
            
            
    # PROPERTIES for choosing weights and functions
    @property
    def weighted(self):
        return self._weighted
    
    @weighted.setter
    def weighted(self,w): # can be true/false
        
        self._weighted = w
        self._f = self._df[(self._on,self._weighted)]
        
    @property
    def convergence_on(self):
        return self._on
        
    @convergence_on.setter
    def convergence_on(self,s):
        
        assert s in ('eigenvalue','eigenvector'), 'Must be "eigenvalue" or "eigenvector".'
        
        self._on = s
        self._f = self._df[(self._on,self._weighted)]
        
    # PROPERTIES for eval_max, tol, store
    @property
    def eval_max(self):
        return self._args[0]
    
    @eval_max.setter
    def eval_max(self,x):
        self._args[0] = x
        
    @property
    def tol(self):
        return self._args[1]
    
    @tol.setter
    def tol(self,x):
        self._args[1] = x
        
    @property
    def store(self):
        return self._args[2]
    
    @store.setter
    def store(self,x):
        self._args[2] = x
        
    # PROPERTY FOR RESETTING ADJACENCY MATRICES
    @property
    def lA(self):
        return self._lA
        
    @lA.setter
    def lA(self,la):
        self._lA = la
        self._N = la[0].shape[0]
        self._T = len(la)
        del self.avg_k
        
        
    
    # functions for printing    
    def __str__ (self):
        spoutp = 'N = %d; T = %d\n' % (self._N,self._T)
        spoutp += 'data type : %s\n' % str(self._dtype)
        
        # Weighted.
        if self.weighted:
            spu = 'True'
        else:
            spu = 'False'
        spoutp += 'weighted : ' + spu + '\n'
        
        # convergence eigenvector/eigenvalue
        spoutp += 'convergence on: {}\n'.format(self.convergence_on)
        
        return spoutp
        
    def __repr__ (self):
        return self.__str__()
        
    
    

 

# check your system
def test_system():
    
    def compare_versions(a,b):
        a2 = tuple( map(int,a.split('.')) )
        b2 = tuple( map(int,b.split('.')) )
        if a2==b2:
            return '* Same version *'
        elif a2>b2:
            return '*** Your version is more recent ***'
        else:
            return '*** Your version older ***'
    
    dc0 = {}
    dc0['python'] = '2.7.11' # 2.7.11
    dc0['numpy'] = '1.11.0'
    dc0['networkx'] = '1.10'
    dc0['scipy'] = '0.17.0'
    
    lerr = []
    ferr = True    
    
    from sys import version_info   
    
    try:    
        from scipy import __version__ as vers_scipy
    except ImportError:
        lerr.append('scipy')
        ferr = False
    
    try:
        import networkx as nx
    except ImportError:
        lerr.append('networkx')
        ferr = False
        
    spu = 'ERROR! MISSING MODULES: '
    for e in lerr:
        spu += e+', '
    spu = spu.strip().strip(',')+'.'

    assert ferr, spu
    
    dc = {}
    dc['python'] = '%s.%s.%s' % version_info[:3]
    dc['numpy'] = np.__version__
    dc['networkx'] = nx.__version__
    dc['scipy'] = vers_scipy

    
    print '\n---------'
    print '---------'
    print 'All required modules are present'
    print '---------'
    print '---------'
    print '{:16}{:16}{:16}'.format('MODULE','TESTED FOR','YOURS')
    for x,v0 in dc0.iteritems():
        print '{:16}{:16}{:16} {}'.format(x,v0,dc[x],compare_versions(dc[x],v0))
    print '--------'
    print '--------'
    print '--------'
    print '--------\n'

    
if __name__ == '__main__':
    
    test_system()
