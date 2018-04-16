# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse import csr_matrix
import networkx as nx
from scipy.optimize import brentq,bisect
#from sys import version_info
#from scipy import __version__ as vers_scipy
from warnings import warn
import traceback, logging


# functions for computing the spectral radius
from utilp import psr1uw, psr1w, psr2uw, psr2w, psr2uw_agg, ThresholdError
try:
    from utilc import psr2uw as psr2uw_c
except ImportError:
    warn('CYTHON CANNOT BE IMPORTED. Every cython function will be silently replaced with a pure python one.')
    psr2uw_c = None
    # When called, it is silently substituted by a pure python one


def lA_to_ars(lA):
    """
    time t:
    lA[t].indptr == l_indptr[t*(N+1):(t+1)*(N+1)]
    lA[t].indices == l_indices[l_place[t]:l_place[t+1]]
    lA[t].data == l_data[l_place[t]:l_place[t+1]]
    """
    l_indptr = np.concatenate([ A.indptr for A in lA ])
    l_indices = np.concatenate([ A.indices for A in lA ])
    l_data = np.concatenate([ A.data for A in lA ])
    l_place = np.cumsum( np.array([0]+[A.indices.shape[0] for A in lA]), dtype=np.int32 )
    return (l_indptr, l_indices, l_data, l_place)




class NetworkFormatError(Exception):
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
            lG[t].add_edge(x, y, weight = float(azz[2]))
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
            
            # check if you specify manually a different period            
            if period is None:
                self._T = buT
            else:
                self._T = period
                # if shorter, cut last time steps
                if self._T < buT:
                    self._lG = self._lG[:self._T]
                # if longer, add empty time steps
                elif self._T > buT:
                    H = self._lG[0].copy()
                    H.remove_edges_from(H.edges())
                    self._lG = self._lG+[H.copy() for i in range(self._T-buT)]
        
        # else: list of graphs
        else:
            if not ( str(type(myn[0])) == "<class 'networkx.classes.graph.Graph'>" or str(type(myn[0])) == "<class 'networkx.classes.digraph.DiGraph'>" ): # networkx graph
                raise NetworkFormatError('Unsupported format: could not find neither networkx Graph nor DiGraph objects.')
            self._lG = myn
            
            if period is not None:
                assert period <= len(self._lG), 'Specified period is longer than dataset.'
                if period < len(self._lG):
                    self._lG = self._lG[:period]
            self._T = len(self._lG) 
            
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
            if ct==self.T:
                assert False, 'Empty network: no edges at any time.'
        dweight = nx.get_edge_attributes(self._lG[ct],'weight')
        if len(dweight)>0 :
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
        return [A.copy() for A in self._lA]
        
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
            return self._attributes
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
                
                
    ##
    ## PROPERTY nodelist
    ##
    @property
    def nodelist(self):
        return self._nodelist
        
    @nodelist.setter
    def nodelist(self,x):
        assert False, 'You cannot manually edit list of nodes.'
                
    
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





    
    
class threshold(object):
    """
    Class for computing the epidemic threshold on a temporal network.
    """
    
    _df = {('eigenvalue',False):psr1uw, ('eigenvalue',True):psr1w, ('eigenvector',False):psr2uw, ('eigenvector',True):psr2w}
    
    # weighted can be None, True, False
    # attributes has to be given only if X is not a tnet
    def __init__(self, X, eval_max=20000, tol=1e-6, store=10, additional_time=0, weighted=None, convergence_on_eigenvector=True, attributes=None, cython=False):
        
        # check the type of input, and store accordingly
        if str(type(X)) == "<class 'threshold.threshold.tnet'>":
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
        
        
        self._args = [eval_max, tol, store]
            
        if convergence_on_eigenvector:
            self._on = 'eigenvector'
        else:
            self._on = 'eigenvalue'

        # cython. REMEMBER 1) cython overrides the choice of weighted and 1/2 convergence functions. 2) if cython module could not be loaded, ERROR
        self.cython = cython

        ###
        if self.cython and psr2uw_c is not None:
            self._f = lambda ladda, mu, sr_target : psr2uw_c(ladda, mu, self.l_indptr, self.l_indices, self.l_data,
                                                             self.l_place, self.N, self.T, self.eval_max, self.tol, self.store, sr_target)
        else:
            self._f = lambda ladda, mu, sr_target : self._df[(self._on,self._weighted)](ladda, mu, self.lA, self.N, self.T, self.eval_max, self.tol, self.store, sr_target)
        ###
        
        # Additional time
        self._addtime = additional_time
                
                
    # additional kwargs (like xtol or rtol) will be passed directly to the root finding routine. See scipy documentation
    # additional_time is the number of empty time steps you want to give. 
    def compute(self, mu2, vmin=0.001, vmax=1., maxiter=50, root_finder='brentq', **kwargs):
        
        # recovery rate(s)
        mu = np.zeros((self._N,), dtype=self._dtype)
        if type(mu2) == dict:
            # heterogeneous mu
            assert hasattr(self,'_attributes'), 'No attributes given.'
            mu[:] = mu2['default']
            #mu = np.full((self._N,), mu2['default'], dtype=self._dtype)
            #mu[:] = mu2['default']
            for i in range(self._N):
                if self._attributes[i] is not None:
                    mu[i] = mu2[self._attributes[i]]
                    
            assert self._addtime==0, 'You cannot give additional time when mu is heterogeneous'
            
        else:
            # homogeneous mu
            #mu = mu2
            mu[:] = mu2
        
        if root_finder.upper() == 'BRENTQ':
            findroot = brentq
        elif root_finder.upper() == 'BISECT':
            findroot = bisect
        else:
            raise ThresholdError,'method for root finding '+root_finder+' is not supported.'
            
        if self._addtime == 0:
            sr_target = 1.
        else:
            # Here mu must be homogeneous (see previous assertion), so I take just the first value.
            sr_target = np.power( 1.-mu[0], -float(self._addtime)/float(self.T) )
            
        # compute threshold, and try all possible errors
        try:
            result, rr = findroot(self._f, vmin, vmax, args=(mu, sr_target), maxiter=maxiter, full_output=True, disp=False, **kwargs)
            
            if not rr.converged:
                raise ThresholdError, 'Optimization did not converge.'
            
        # Error coming from spectral radius computation, or no convergence in root finding
        except ThresholdError, err_string:
            print err_string
            result = np.nan
            
        # Error: interval is likely not to contain zeros
        except ValueError:
            sr_min = self._f(vmin,mu,sr_target)
            sr_max = self._f(vmax,mu,sr_target)
            spuz = 'ValueError: Interval may not contain zeros (or other ValueError). f({:.5f})={:.5f}; f({:.5f})={:.5f}'.format(vmin, sr_min,vmax, sr_max)
            print spuz 
            result = np.nan
            
        # catch any other exception
        except Exception as ecc:
            print ecc
            logging.error(traceback.format_exc())
            result = np.nan

        #finally:
        #    return result
        return result
            
            
    # Compute spectral radius in specific point
    # additional kwargs (like xtol or rtol) will be passed directly to the root finding routine. See scipy documentation
    def sr_point(self, ladda, mu2, maxiter=50, **kwargs):

        mu = np.zeros((self._N,), dtype=self._dtype)
        if type(mu2) == dict:
            # heterogeneous mu
            assert hasattr(self, '_attributes'), 'No attributes given.'
            mu[:] = mu2['default']
            # mu = np.full((self._N,), mu2['default'], dtype=self._dtype)
            # mu[:] = mu2['default']
            for i in range(self._N):
                if self._attributes[i] is not None:
                    mu[i] = mu2[self._attributes[i]]

            assert self._addtime == 0, 'You cannot give additional time when mu is heterogeneous'

        else:
            # homogeneous mu
            # mu = mu2
            mu[:] = mu2

        """
        # recovery rate(s)
        if type(mu2) == dict:
            # heterogeneous mu
            assert hasattr(self,'_attributes'), 'No attributes given.'
            mu = np.full((self._N,), mu2['default'], dtype=self._dtype)
            for i in range(self._N):
                if self._attributes[i] is not None:
                    mu[i] = mu2[self._attributes[i]]
                    
            assert self._addtime==0, 'You cannot give additional time when mu is heterogeneous'
        else:
            # homogeneous mu
            mu = mu2
        """
            
        if self._addtime == 0:
            sr_target = 1.
        else:
            # here mu must be homogeneous (see previous assertion), so I take just the first value
            sr_target = np.power( 1.-mu[0], -float(self._addtime)/float(self.T) )

           
        try:
            result = self._f(ladda,mu,sr_target)
        except ThresholdError, err_string:
            result = np.nan
            print err_string
        finally:
            return result
            
            
            
            
    # BASIC PROPERTIES
    @property
    def N (self):
        return self._N
    @N.setter
    def N (self,x):
        print 'You cannnot manually set the number of nodes.'
    @N.deleter
    def N (self):
        print 'You cannnot manually delete the number of nodes.'
        
    @property
    def T (self):
        return self._T
    @T.setter
    def T (self,x):
        print 'You cannnot manually set the period.'
    @T.deleter
    def T (self):
        print 'You cannnot manually delete the period.'
    
    
    # PROPERTY for average degree (useful for boundaries)
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

        if not self.cython:
            self._weighted = w
            self._f = lambda ladda, mu, sr_target: self._df[(self._on, self._weighted)](ladda, mu, self.lA, self.N,
                                                                                        self.T, self.eval_max, self.tol, self.store, sr_target)
            # delete variables that depend on this
            del self.avg_sr
        else:
            warn('Cython is on, so I am ignoring this.')
        
        
        
    @property
    def convergence_on(self):
        return self._on
        
    @convergence_on.setter
    def convergence_on(self, s):
        
        assert s in ('eigenvalue','eigenvector'), 'Must be "eigenvalue" or "eigenvector".'

        if not self.cython:
            self._on = s
            self._f = lambda ladda, mu, sr_target: self._df[(self._on, self._weighted)](ladda, mu, self.lA, self.N, self.T,
                                                                                        self.eval_max, self.tol, self.store, sr_target)
            # delete variables that depend on this
            del self.avg_sr
        else:
            warn('Cython is on, so I am ignoring this.')


    # PROPERTIES for eval_max, tol, store, sr_target
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
       
       
    @property
    def additional_time(self):
        return self._addtime
        
    @additional_time.setter
    def additional_time(self,x):
        self._addtime = x
        
    @additional_time.deleter
    def additional_time(self):
        print 'Resetting additional time back to zero.'
        self._addtime = 0
        
    # PROPERTY FOR RESETTING ADJACENCY MATRICES
    @property
    def lA(self):
        return self._lA
        
    @lA.setter
    def lA(self,la):
        self._lA = la
        self._N = la[0].shape[0]
        self._T = len(la)
        # delete variables that depend on lA
        del self.avg_k
        del self.avg_A
        del self.avg_sr


    ### USED BY CYTHON
    @property
    def l_indptr(self):
        if not hasattr(self, '_l_indptr'):
            self._l_indptr, self._l_indices, self._l_data, self._l_place = lA_to_ars(self.lA)
        return self._l_indptr

    @property
    def l_indices(self):
        if not hasattr(self, '_l_indices'):
            self._l_indptr, self._l_indices, self._l_data, self._l_place = lA_to_ars(self.lA)
        return self._l_indices

    @property
    def l_data(self):
        if not hasattr(self, '_l_data'):
            self._l_indptr, self._l_indices, self._l_data, self._l_place = lA_to_ars(self.lA)
        return self._l_data

    @property
    def l_place(self):
        if not hasattr(self, '_l_place'):
            self._l_indptr, self._l_indices, self._l_data, self._l_place = lA_to_ars(self.lA)
        return self._l_place



        
    
    # PROPERTY FOR AGGREGATED MATRIX
    @property
    def avg_A(self):
        
        if not hasattr(self,'_avg_A'):
            self._avg_A = csr_matrix(self._lA[0])
            for A in self._lA[1:]:
                self._avg_A = self._avg_A + A
                
        return self._avg_A/float(self.T)
        
    @avg_A.setter
    def avg_A(self,x):
        print 'You cannnot manually set the average adjacency matrix.'
        
    @avg_A.deleter
    def avg_A(self):
        if hasattr(self,'_avg_A'):
            del self._avg_A
        
    
    # PROPERTY FOR THE AGGREGATED SPECTRAL RADIUS
    @property
    def avg_sr(self):
        
        if not hasattr(self,'_avg_sr'):
            thisargs = list(self._args)
            thisargs[0] = 100*thisargs[0] # increase evaluations
            thisargs[1] = 100*thisargs[1] # decrease precision
            self._avg_sr = psr2uw_agg(self.avg_A,self._N,*thisargs )
        
        return self._avg_sr
        
    @avg_sr.setter
    def avg_sr(self,x):
        print 'You cannnot manually set the spectral radius of average adjacency matrix.'
        
    @avg_sr.deleter
    def avg_sr(self):
        if hasattr(self,'_avg_sr'):
            del self._avg_sr
        
        
    
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
        
        # additional time
        spoutp += 'additional time steps: {}\n'.format(self.additional_time)
        
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
