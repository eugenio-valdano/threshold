cimport cython
from libc.stdlib cimport malloc, free
from libc.math cimport fabs as c_abs

ctypedef long double mydouble
ctypedef long myint

# @cython.boundscheck(False)
cpdef mydouble psr2uw(mydouble ladda, mydouble [:] mu, int [:] lip, int [:] li, mydouble [:] ld, int [:] lp, myint N, myint T, myint valumax, mydouble tolerance, myint store, mydouble sr_target):
    """
    lip: indptr
    li: indices
    ld: data
    lp: places
    """

    cdef mydouble rootT = T
    rootT = 1./rootT

    # counters and indices and util
    cdef int i,k,a,tau
    cdef mydouble sr,norm,delta

    # eigenvector register. c-th vector is MV[c*N:(c+1)*N]. Vector to use
    cdef mydouble *MV
    cdef mydouble *v
    cdef mydouble *v2
    MV = <mydouble *>malloc(N*store*cython.sizeof(mydouble))
    v  = <mydouble *>malloc(N*cython.sizeof(mydouble))
    v2 = <mydouble *>malloc(N*cython.sizeof(mydouble))
    cdef int c = 0
    i = 0
    while i<N:
        MV[i] = 1./(N ** 0.5)
        v[i] = MV[i]
        i += 1

    cdef int flag = 0
    cdef mydouble ecco;

    with nogil:
        k = 0
        while flag != 1 and k < T*valumax:

            # time step
            tau = k%T

            i = 0
            while i<N:
                # diagonal
                v2[i] = (1.-mu[i])*v[i]

                # multiply
                a = lip[i+tau*(N+1)]
                while a<lip[i+1+tau*(N+1)]:
                    v2[i] += ladda*ld[a]*v[li[a]]
                    a += 1
                i += 1

            # copy back to v
            i = 0
            while i<N:
                v[i] = v2[i]
                i += 1

            # if period is complete
            if tau == T-1:

                # compute sr approximation, and norm of v
                sr = 0.
                norm = 0.
                i = 0
                while i<N:
                    sr += v[i]*MV[c*N+i]
                    norm += v[i]*v[i]
                    i += 1
                norm = norm ** 0.5

                # normalize v, and compute L1-norm
                i = 0
                delta = 0.
                while i<N:
                    v[i] = v[i]/norm
                    delta += c_abs(MV[c*N+i]-v[i])
                    i += 1

                if delta<tolerance:
                    flag = 1
                    ecco = sr**rootT - sr_target

                if flag!=1:
                    # put in store
                    c = (c+1)%store
                    i = 0
                    while i<N:
                        MV[c*N+i] = v[i]
                        i += 1

            k += 1

        free(MV)
        free(v)
        free(v2)


    if flag != 1:
        raise ValueError

    return ecco
