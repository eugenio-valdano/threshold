# -*- coding: utf-8 -*-
#
### written with
### python 2.7.8
### numpy 1.9.0
### scipy 0.14.0
### networkx 1.9.1
#
import numpy as np
from scipy.sparse import csr_matrix
import networkx as nx
from scipy.optimize import brentq,bisect
#from scipy.linalg import eigvals

class NetworkFormatError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
##############
##############
##############
class ThresholdError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
        
        
                      

def text_to_net (fname,**kwargs):
    if 'separator' in kwargs:
        separator = kwargs['separator']
    else:
        separator = '\t'
    #
    if 'directed' in kwargs:
        directed = kwargs['directed']
    else:
        directed = False
    #
    raw = open(fname).readlines() # read file
    #
    if len( raw[0].strip().split(separator) )>3: # is it weighted?
        isw = True
    else:
        isw = False
    #
    snodes = set()
    stimes = set()
    for line in raw:
        azz = line.strip().split(separator)
        stimes.add( int(azz[-1]) )
        snodes.add( azz[0] )
        snodes.add( azz[1] )
    N = len(snodes)
    T = max(stimes)+1
    #
    if directed:
        lG = [ nx.DiGraph() for t in range(T) ]
    else:
        lG = [ nx.Graph() for t in range(T) ]
    #
    # fill with edges
    if isw:
        for line in raw:
            azz = line.strip().split(separator)
            x,y,t = azz[0],azz[1],int(azz[-1])
            lG[t].add_edge(x,y,weight=int(azz[2]))
    else:
        for line in raw:
            azz = line.strip().split(separator)
            x,y,t = azz[0],azz[1],int(azz[-1])
            lG[t].add_edge(x,y)
    #
    # fill with all nodes
    snodes = list(snodes)
    for G in lG:
        lG.add_nodes_from( snodes )
    #
    return lG,N,T # returns list of graphs and number of nodes
    

def graph_to_csr (lG,dtype):
    lAs = []
    for G in lG:
        A = nx.adjacency_matrix(G)
        lAs.append( csr_matrix( A,dtype=dtype ))
    return lAs
    

        
class tnet:
    lG = None
    T = None
    N = None
    lA = None
    dtype = np.float64
    #
    def __init__ (self, myn, period=None, dtype='float64', **kwargs ): # directed=False, separator='\t'
        #
        if dtype != 'float64': # if different from this, set longer float
            self.dtype = np.float128
        #
        if type(myn) == 'str': # PATH TO FILE
            self.lG,self.N,self.T = text_to_net(myn,**kwargs )
        #
        #
        else:
            if not ( str( type(myn[0]) ) == "<class 'networkx.classes.graph.Graph'>" or str( type(myn[0]) ) == "<class 'networkx.classes.digraph.DiGraph'>" ): # networkx graph
                raise NetworkFormatError('Unsupported format: could not find neither networkx Graph nor DiGraph objects.')
            self.lG = myn
            #
            if self.T == None:
                self.T = len(self.lG)
            #
            # fill all graphs with all nodes
            snodes = set()
            if self.N == None:
                self.N = len(snodes)
            for G in self.lG:
                snodes |= set( G.nodes() )
            snodes = list(snodes)
            for G in self.lG:
                G.add_nodes_from( snodes )
        #    
    #
    #
    def getMatrices (self):
        if self.lA == None:
            self.lA = []
            for G in self.lG:
                A = nx.adjacency_matrix(G)
                self.lA.append( csr_matrix( A,dtype=self.dtype ))
        return self.lA
    #
    #
    def __str__ (self):
        spoutp = 'N = %d; T = %d\n' % (self.N,self.T)
        spoutp += 'data type : %s\n' % str( self.dtype )
        #
        if str( type(self.lG[0]) ) == "<class 'networkx.classes.graph.Graph'>": # WEIGHTED
            spu = 'False'
        else:
            spu = 'True'
        spoutp += 'weighted : ' + str(self.weight) + '\n'
        #
        t = 0
        while len( self.lG[t].edges() ) == 0:
            t += 1
        if 'weight' in self.lG[t].edges(data=True)[0][2] : # DIRECTED
            spu = 'True'
        else:
            spu = 'False'
        spoutp += 'directed : ' + str(spu) + '\n'
        #
        if self.lA == None: # MATRICES LOADED
            spu = 'not loaded'
        else:
            spu = 'loaded'
        spoutp += 'adjacency matrices : ' + str(spu) + '\n'
        return spoutp
    #
    def __repr__ (self):
        return self.__str__()


############################
############################
############################
############################
########################################################
############################
############################
############################
########################################################
############################
############################
############################
############################

def power_spectral_radius(ladda,mu,lA,N,T,valumax=1000,stabint=10,tolerance=0.0001): #lA = sparse csr matrices 
    """
    NOTE: matrices in lA must be objects \'scipy.sparse.csr.csr_matrix\'
    """
    rootT=1.0/float(T)
    # initialize
    leval = []
    v0 = 0.9*np.random.random(N)+0.1
    v = v0.copy()
    vold = v.copy()
    interrupt = False # when there is convergence, becomes True
    #
    itmax = T*valumax
    for k in range(itmax):
        v = ladda*lA[k%T].dot(v) + (1.-mu)*v # iteration
        if k%T == T-1: # period completed
            autoval = np.dot(vold,v)
            leval.append( autoval**rootT )
            #
            if len(leval)>=stabint: # check convergence
                fluct = ( max( leval[-stabint:] ) - min( leval[-stabint:] ) ) / np.mean( leval[-stabint:] )
            else:
                fluct = 1.+tolerance
            if fluct < tolerance:
                interrupt = True
                break
            mnorm = np.linalg.norm(v)
            v = v/mnorm
            vold = v.copy()
    #
    if not interrupt: # if never interrupted, check now convergence
        fluct = ( max( leval[-stabint:] ) - min( leval[-stabint:] ) ) / np.mean( leval[-stabint:] )
        if fluct >= tolerance:
            raise ThresholdError,'Power method did not converge.'
    return leval[-1]-1.
############################
############################
############################
############################
############################
def power_spectral_radius_weighted(ladda,mu,lA,N,T,valumax=1000,stabint=10,tolerance=0.0001):
    """
    NOTE: matrices in lA must be objects \'scipy.sparse.csr.csr_matrix\'
    """
    loglad = np.log(1.-ladda)
    rootT=1.0/float(T)
    # initialize
    leval = []
    v0 = 0.9*np.random.random(N)+0.1
    v = v0.copy()
    vold = v.copy()
    interrupt = False # when there is convergence, becomes True
    #
    itmax = T*valumax
    for k in range(itmax):
        # use of function expm1: -(loglad*lA[k%T]).expm1() = 1-(1-ladda)^Aij
        v = -(loglad*lA[k%T]).expm1().dot(v) + (1.-mu)*v # iteration
        if k%T == T-1: # period completed
            autoval = np.dot(vold,v)
            leval.append( autoval**rootT )
            #
            if len(leval)>=stabint: # check convergence
                fluct = ( max( leval[-stabint:] ) - min( leval[-stabint:] ) ) / np.mean( leval[-stabint:] )
            else:
                fluct = 1.+tolerance
            if fluct < tolerance:
                interrupt = True
                break
            mnorm = np.linalg.norm(v)
            v = v/mnorm
            vold = v.copy()
    #
    if not interrupt: # if never interrupted, check now convergence
        fluct = ( max( leval[-stabint:] ) - min( leval[-stabint:] ) ) / np.mean( leval[-stabint:] )
        if fluct >= tolerance:
            raise ThresholdError,'Power method did not converge.'
    return leval[-1]-1.
############################
############################
############################
############################
############################
def find_threshold (mu,R,vmin=0.001,vmax=0.999,weighted=False,findroot='brentq',maxiter=50,xtol=0.0001):
    #
    if findroot == 'brentq':
        rootFinder = brentq
    elif findroot == 'bisect':
        rootFinder = bisect
    else:
        raise ThresholdError,'method for root finding '+findroot+' is not supported.'
    #
    try:
        if weighted:
            result,rr = rootFinder( power_spectral_radius_weighted,vmin,vmax,args=(mu,R.getMatrices(),R.N,R.T),xtol=xtol,maxiter=maxiter,full_output=True,disp=False )
        else:
            result,rr = rootFinder( power_spectral_radius,         vmin,vmax,args=(mu,R.getMatrices(),R.N,R.T),xtol=xtol,maxiter=maxiter,full_output=True,disp=False )
    except ThresholdError,err_string:
        print err_string
        return np.nan
    except ValueError:
        print  'ValueError: Interval may not contain zeros (or other ValueError).',power_spectral_radius_weighted(vmax,mu,R.getSparse(),R.N,R.T)
        return np.nan
    else:
        if not rr.converged:
            print 'Optimization did not converge.'
            return np.nan
    return result
##############
##############
##############