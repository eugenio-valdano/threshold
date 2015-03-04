# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 14:53:30 2014

@author: eugenio
"""
import numpy as np
from scipy.sparse import csr_matrix
import networkx as nx
from scipy.optimize import brentq,bisect
from scipy.linalg import eigvals

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
##############
##############
##############

def text_to_matrix_sparse (fname,N,dtype,weighted,directed): # transforms an edgelist into a list of csr_matrices
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    G.add_nodes_from( range(N) )
    if weighted:
        for line in open(fname):
            azz = line.strip().split()
            azz = map(int,azz)
            G.add_edge(*azz[:2],weight=azz[3])
        return csr_matrix( nx.adjacency_matrix(G,weight='weight') , dtype=dtype )
    else:
        for line in open(fname):
            azz = line.strip().split()
            azz = map(int,azz)
            G.add_edge(*azz[:2])
        return csr_matrix( nx.adjacency_matrix(G,weight=None) , dtype=dtype )
        
def text_to_matrix_dense (fname,N,dtype,weighted,directed): # transforms an edgelist into a list of dense arrays
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    G.add_nodes_from( range(N) )
    if weighted:
        for line in open(fname):
            azz = line.strip().split()
            azz = map(int,azz)
            G.add_edge(*azz[:2],weight=azz[3])
        return np.array( nx.to_numpy_matrix(G,weight='weight',dtype=dtype) )
    else:
        for line in open(fname):
            azz = line.strip().split()
            azz = map(int,azz)
            G.add_edge(*azz[:2])
        return np.array( nx.to_numpy_matrix(G,weight=None,dtype=dtype) )


### built for
### python 2.7.8
### numpy 1.9.0
### scipy 0.14.0
### networkx 1.9.1
###
### supported formats:
### '<dirtoname>/<name>_t_' and extension 'npy'
### '<dirtoname>/<name>' and extension 'npz'
### '<dirtoname>/<name>_t_' and extension 'txt' : edge list with columns separated by whitespace/tab, and id_out id_in weight
### list of networkx graphs
### list of dense arrays
### list of sparse arrays
###
### directed=True is useful only if I give edge list and want to treat is as directed. Otherwise assumes matrices as they are, not looking if they are sym or not

class tnet:
    lA = None
    lA_sparse = None
    T = None
    N = None
    dtype = np.float64
    #
    def __init__ (self, gimme, extension = 'npy', period = None,
                  nodes = None, weighted = False, directed = False,
                  priority = 'sparse', dtype=np.float64 , nodense = True ):
        #
        self.weight = weighted
        #          
        if period != None:
            self.T = period
        #
        if nodes != None:
            self.N = nodes
        #
        if nodense: # if true prevents from computing dense matrices. useful for large networks
            self.nodense = True
            if priority == 'dense':
                raise NetworkFormatError("Cannot set priority='dense' when nodense=True.")
        else:
            self.nodense = False
        #
        self.dtype = dtype
        #
        if type(gimme) == str:
            # it's a dir to a file
            #
            # extension npy
            if extension == 'npy': 
                assert self.T != None, 'Format npy requires giving period length as period.'
                if priority == 'dense':
                    self.lA = []
                    for t in range(self.T):
                        self.lA.append( np.array( np.load( gimme+str(t)+'.'+extension ),dtype=self.dtype ) )
                    if self.N == None:
                        self.N = self.lA[0].shape[0]
                else: #priority sparse
                    self.lA_sparse = []
                    for t in range(self.T):
                        self.lA_sparse.append( csr_matrix( np.load( gimme+str(t)+'.'+extension ),dtype=self.dtype  ) )
                    if self.N == None:
                        self.N = self.lA_sparse[0].shape[0]
                #
            #
            # extension npz
            elif extension == 'npz':
                with np.load( gimme+'.'+extension ) as archivio:
                    if self.T == None:
                        self.T = len(archivio.files)
                    #
                    if priority == 'dense':
                        self.lA = []
                        for t in range(self.T):
                            self.lA.append( np.array( archivio['arr_'+str(t)],dtype=self.dtype ) )
                        if self.N == None:
                            self.N = self.lA[0].shape[0]
                    else: # priority sparse
                        self.lA_sparse = []
                        for t in range(self.T):
                            self.lA_sparse.append( csr_matrix(archivio['arr_'+str(t)],dtype=self.dtype) )
                        if self.N == None:
                            self.N = self.lA_sparse[0].shape[0]
            #
            # extension txt
            elif extension == 'txt':
                if self.T == None:
                    raise NetworkFormatError('Format txt requires giving period length as period=')
                if self.N == None:
                    raise NetworkFormatError('Format txt requires giving number of nodes as nodes=')
                #
                if priority == 'dense':
                    self.lA = []
                    for t in range(self.T):
                        self.lA.append( text_to_matrix_dense( gimme+str(t)+'.'+extension, self.N, self.dtype , weighted , directed ) )
                    #if self.N == None:
                    #   self.N = self.lA[0].shape[0]
                else: # priority sparse
                    self.lA_sparse = []
                    for t in range(self.T):
                        self.lA_sparse.append( csr_matrix( text_to_matrix_sparse( gimme+str(t)+'.'+extension, self.N, self.dtype , weighted , directed )  ) )
                    #if self.N == None:
                    #   self.N = self.lA_sparse[0].shape[0]
        #
        elif type(gimme) == list:
            if self.T == None:
                self.T = len(gimme)
            #
            # networkx graphs
            if str( type(gimme[0]) ) == "<class 'networkx.classes.graph.Graph'>" or str( type(gimme[0]) ) == "<class 'networkx.classes.digraph.DiGraph'>":
                if self.N == None:
                    self.N = len( gimme[0].nodes() )
                if priority == 'dense':
                    self.lA = []
                    for t in range(self.T):
                        self.lA.append( np.array( nx.to_numpy_matrix(gimme[t],dtype=self.dtype) ) )
                else: # priority sparse
                    self.lA_sparse = []
                    for t in range(self.T):
                        self.lA_sparse.append( csr_matrix( nx.adjacency_matrix(gimme[t]),dtype=self.dtype ) )
            #
            # dense arrays
            elif str( type(gimme[0]) ) == "<type 'numpy.ndarray'>":
                print 'Setting nodense=False'
                self.nodense = False
                if type(gimme[0][0,0]) != dtype:
                    raise NetworkFormatError('Input data type and the one of given arrays do not match.')
                self.lA = list(gimme)
                if self.N == None:
                    self.N = gimme[0].shape[0]
            #
            # sparse arrays
            elif str( type(gimme[0]) ) == "<class 'scipy.sparse.csr.csr_matrix'>":
                if type(gimme[0][0,0]) != dtype:
                    raise NetworkFormatError('Input data type and the one of given arrays do not match.')
                self.lA_sparse = list(gimme)
                if self.N == None:
                    self.N = gimme[0].shape[0]
            #
        else:
            raise NetworkFormatError('The format is not recognized')
        #
        #
    #
    #
    def setDense (self):
        if self.nodense == True:
            raise NetworkFormatError('Trying to produce dense matrices with todense=True.')
        if self.lA == None:
            self.lA = []
            F = np.empty( shape=(self.N,self.N),dtype=self.dtype )
            for A in self.lA_sparse:
                A.todense(out=F)
                self.lA.append( F.copy() )
    #
    #
    def setSparse (self):
        if self.lA_sparse == None:
            self.lA_sparse = []
            for A in self.lA:
                self.lA_sparse.append( csr_matrix(A) )
    #
    #
    def getDense (self):
        if self.lA == None:
            self.setDense()
        return self.lA
    #
    #
    def getSparse (self):
        if self.lA_sparse == None:
            self.setSparse()
        return self.lA_sparse
    #
    #
    def __str__ (self):
        sputo = 'N = %d; T = %d\n' % (self.N,self.T)
        spu = ['loaded','loaded']
        if self.lA == None:
            spu[0] = 'None'
        if self.lA_sparse == None:
            spu[1] = None
        sputo += 'lA : %s; lA_sparse : %s\n' % tuple(spu)
        sputo += 'data type : %s\n' % str( self.dtype )
        sputo += 'weighted : ' + str(self.weight) + '\n'
        return sputo
    #
    def __repr__ (self):
        return self.__str__()

############################
############################
############################
############################
############################
def direct_spectral_radius (ladda,mu,lA,N,T):
    rootT=1.0/float(T)
    identi = np.identity(N)
    P = ladda*lA[0] + (1.-mu)*identi
    for t in range(1,T):
        P = np.dot(ladda*lA[t] + (1.-mu)*identi,P)
    spec = np.abs( eigvals(P) )
    radP = np.max(spec)
    return radP**rootT - 1.
############################
############################
############################
############################
############################
def direct_spectral_radius_weighted (ladda,mu,lA,N,T):
    rootT=1.0/float(T)
    identi = np.identity(N)
    P = 1. - np.power(1.-ladda,lA[0]) + (1.-mu)*identi
    for t in range(1,T):
        P = np.dot( 1. - np.power(1.-ladda,lA[t]) + (1.-mu)*identi,P)
    spec = np.abs( eigvals(P) )
    radP = np.max(spec)
    return radP**rootT - 1.
############################
############################
############################
############################
############################

def power_spectral_radius(ladda,mu,lA,N,T,valumax=1000,stabint=10,tolerance=0.0001): #lA = MATRICI SPARSE CSR
    """
    NOTA: le matrici di lA devono essere oggetti \'scipy.sparse.csr.csr_matrix\' ( creali con la funzione scipy.sparse.csr_matrix)
    """
    rootT=1.0/float(T)
    # inizializzo le cose iniziali
    leval = []
    v0 = 0.9*np.random.random(N)+0.1
    v = v0.copy()
    vold = v.copy()
    interrotto = False # se viene interrotto per convergenza, diventa vero
    #
    itmax = T*valumax
    for k in range(itmax):
        v = ladda*lA[k%T].dot(v) + (1.-mu)*v # iterazione
        if k%T == T-1: # completato il periodo
            autoval = np.dot(vold,v)
            leval.append( autoval**rootT )
            #
            if len(leval)>=stabint: # check della convergenza
                fluct = ( max( leval[-stabint:] ) - min( leval[-stabint:] ) ) / np.mean( leval[-stabint:] )
            else:
                fluct = 1.+tolerance
            if fluct < tolerance:
                interrotto = True
                break
            norma = np.linalg.norm(v)
            v = v/norma
            vold = v.copy()
    #
    if not interrotto: # se non e' stato mai interrotto, controllo la convergenza
        fluct = ( max( leval[-stabint:] ) - min( leval[-stabint:] ) ) / np.mean( leval[-stabint:] )
        if fluct >= tolerance:
            raise ThresholdError,'Power method did not converge.'
    return leval[-1]-1.
############################
############################
############################
############################
############################
def power_spectral_radius_weighted(ladda,mu,lA,N,T,valumax=1000,stabint=10,tolerance=0.0001): #lA = MATRICI SPARSE CSR
    """
    NOTA: le matrici di lA devono essere oggetti \'scipy.sparse.csr.csr_matrix\' ( creali con la funzione scipy.sparse.csr_matrix)
    """
    loglad = np.log(1.-ladda)
    rootT=1.0/float(T)
    # inizializzo le cose iniziali
    leval = []
    v0 = 0.9*np.random.random(N)+0.1
    v = v0.copy()
    vold = v.copy()
    interrotto = False # se viene interrotto per convergenza, diventa vero
    #
    itmax = T*valumax
    for k in range(itmax):
        # -(loglad*lA[k%T]).expm1() = 1-(1-ladda)**Aij
        v = -(loglad*lA[k%T]).expm1().dot(v) + (1.-mu)*v # iterazione     #### CAMBIARE!
        if k%T == T-1: # completato il periodo
            autoval = np.dot(vold,v)
            leval.append( autoval**rootT )
            #
            if len(leval)>=stabint: # check della convergenza
                fluct = ( max( leval[-stabint:] ) - min( leval[-stabint:] ) ) / np.mean( leval[-stabint:] )
            else:
                fluct = 1.+tolerance
            if fluct < tolerance:
                interrotto = True
                break
            norma = np.linalg.norm(v)
            v = v/norma
            vold = v.copy()
    #
    if not interrotto: # se non e' stato mai interrotto, controllo la convergenza
        fluct = ( max( leval[-stabint:] ) - min( leval[-stabint:] ) ) / np.mean( leval[-stabint:] )
        if fluct >= tolerance:
            raise ThresholdError,'Power method did not converge.'
    return leval[-1]-1.
############################
############################
############################
############################
############################
def find_threshold (mu,R,method='power',weighted=False,findroot='brentq',vmin=0.001,vmax=0.999,maxiter=50,xtol=0.0001): # f=funzione che usa per calcolare auto
    #
    if findroot == 'brentq':
        rootFinder = brentq
    elif findroot == 'bisect':
        rootFinder = bisect
    else:
        raise ThresholdError,'method for root finding '+findroot+' is not supported.'
    #
    if method=='direct':
        try:
            if weighted:
                result,rr = rootFinder( direct_spectral_radius_weighted,vmin,vmax,args=(mu,R.getDense(),R.N,R.T),xtol=xtol,maxiter=maxiter,full_output=True,disp=False )
            else:
                result,rr = rootFinder( direct_spectral_radius,vmin,vmax,args=(mu,R.getDense(),R.N,R.T),xtol=xtol,maxiter=maxiter,full_output=True,disp=False )
        except np.linalg.LinAlgError:
            print 'Computation of spectral radius does not converge.'
        except ValueError:
            print  'ValueError: Interval may not contain zeros (or other ValueError).'
            return np.nan
        else:
            if not rr.converged:
                print 'Optimization did not converge.'
                return np.nan
        return result
    elif method=='power':
        try:
            if weighted:
                result,rr = rootFinder( power_spectral_radius_weighted,vmin,vmax,args=(mu,R.getSparse(),R.N,R.T),xtol=xtol,maxiter=maxiter,full_output=True,disp=False )
            else:
                result,rr = rootFinder( power_spectral_radius,vmin,vmax,args=(mu,R.getSparse(),R.N,R.T),xtol=xtol,maxiter=maxiter,full_output=True,disp=False )
        except ThresholdError,err_string:
            print err_string
            return np.nan
        except ValueError:
            print  'ValueError: Interval may not contain zeros (or other ValueError).'
            return np.nan
        else:
            if not rr.converged:
                print 'Optimization did not converge.'
                return np.nan
        return result
    else:
        raise ThresholdError,'method '+method+' is not supported.'
##############
##############
##############