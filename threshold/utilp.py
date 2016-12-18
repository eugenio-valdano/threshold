# Now functions related to threshold computation.

import numpy as np

class ThresholdError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

# There are two kinds of algorithm: psr1 and psr2 (psr=power spectral radius). They're both weighted and unweighted, so psr1uw, psr1w, psr2uw, psr2w
# psr1 is more crude: it checks convergence on the value of the spectral radius itself.
# psr2 is checks convergence on the principal eigenvector

# PSR1

# unweighted
def psr1uw(ladda, mu, lA, N, T, valumax, tolerance, store, sr_target):
    # parameters
    # valumax = kwargs['valumax'] # 1000
    # tolerance = kwargs['tolerance'] # 1e-5
    # store = kwargs['store'] # 10

    rootT = 1.0 / float(T)
    # same data type as the adjacency matrices
    dtype = type(lA[0][0, 0])

    # Initialize
    leval = np.empty(shape=(store,), dtype=dtype)
    ceval = 0
    v0 = 0.9 * np.random.random(N) + 0.1
    v0 = np.array(v0, dtype=dtype)
    v = v0.copy()
    vold = v.copy()
    interrupt = False  # When convergence is reached, it becomes True.

    itmax = T * valumax
    for k in range(itmax):
        # Perform iteration:
        v = ladda * lA[k % T].dot(v) + (1. - mu) * v

        # Whether period is completed:
        if k % T == T - 1:
            # autoval = np.dot(vold,v)
            leval[ceval % store] = np.dot(vold, v) ** rootT
            ceval += 1
            # leval.append(autoval**rootT)

            # Check convergence
            if ceval >= store:
                fluct = (np.max(leval) - np.min(leval)) / np.mean(leval)
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
        fluct = (np.max(leval) - np.min(leval)) / np.mean(leval)
        if fluct >= tolerance:
            raise ThresholdError, 'Power method did not converge.'

    return leval[-1] - sr_target


# weighted
def psr1w(ladda, mu, lA, N, T, valumax, tolerance, store, sr_target):
    # parameters
    # valumax = kwargs['valumax'] # 1000
    # tolerance = kwargs['tolerance'] # 1e-5
    # store = kwargs['store'] # 10

    rootT = 1.0 / float(T)
    # same data type as the adjacency matrices
    dtype = type(lA[0][0, 0])
    loglad = np.log(1. - ladda)

    # Initialize
    leval = np.empty(shape=(store,), dtype=dtype)
    ceval = 0
    v0 = 0.9 * np.random.random(N) + 0.1
    v0 = np.array(v0, dtype=dtype)
    v = v0.copy()
    vold = v.copy()
    interrupt = False  # When convergence is reached, it becomes True.

    itmax = T * valumax
    for k in range(itmax):

        # Perform iteration. Meaning of function expm1: -(loglad*lA[k%T]).expm1() = 1-(1-ladda)^Aij
        v = -(loglad * lA[k % T]).expm1().dot(v) + (1. - mu) * v

        # Whether period is completed
        if k % T == T - 1:
            leval[ceval % store] = np.dot(vold, v) ** rootT
            ceval += 1

            # Check convergence
            if ceval >= store:
                fluct = (np.max(leval) - np.min(leval)) / np.mean(leval)
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
        fluct = (np.max(leval) - np.min(leval)) / np.mean(leval)
        if fluct >= tolerance:
            raise ThresholdError, 'Power method did not converge.'
    return leval[-1] - sr_target


# PSR2

# unweighted
def psr2uw(ladda, mu, lA, N, T, valumax, tolerance, store, sr_target):
    # stop when the L1 norm of (v_{a}-v_{a-1}) is below tolerance, where a is the a-th loop on the period

    # parameters
    # valumax = kwargs['valumax'] # 20000
    # tolerance = kwargs['tolerance'] # 1e-6
    # store = kwargs['store'] # 10
    # sr_target is usually=1. It's the target value for the spectral radius
    # Unless I'm discarding some empty timestep, in that case I want it to be (1-mu)^{-tau} where tau is how many I discard

    rootT = 1.0 / float(T)
    # same data type as the adjacency matrices
    dtype = type(lA[0][0, 0])

    # Initialize eigenvector register
    MV = np.empty(shape=(N, store), dtype=dtype)
    # first vector is a versor with the equal components:
    MV[:, 0] = np.array([1. / np.sqrt(N)] * N, dtype=dtype)
    # register counter
    c = 0
    # vector to work on:
    v = MV[:, c].copy()

    for k in range(T * valumax):
        # Perform iteration:
        v = ladda * lA[k % T].dot(v) + (1. - mu) * v

        # Whether period is completed:
        if k % T == T - 1:

            # spectral radius
            sr = np.dot(MV[:, c % store], v)

            # normalize
            v = v / np.linalg.norm(v)

            # Compute tolerance, and return if reached:
            delta = np.sum(np.abs(MV[:, c % store] - v))
            if delta < tolerance:
                # return spectral radius^(1/T) - 1, as usual.
                return sr ** rootT - sr_target  # , v/np.linalg.norm(v)

            # increment index, and update storage
            c += 1
            MV[:, c % store] = v.copy()

    # if it goes out of the loop without returning the sr
    raise ThresholdError, 'Power method did not converge.'


# weighted
def psr2w(ladda, mu, lA, N, T, valumax, tolerance, store, sr_target):
    # stop when the L1 norm of (v_{a}-v_{a-1}) is below tolerance, where a is the a-th loop on the period

    # parameters
    # valumax = kwargs['valumax'] # 20000
    # tolerance = kwargs['tolerance'] # 1e-6
    # store = kwargs['store'] # 10

    loglad = np.log(1. - ladda)
    rootT = 1.0 / float(T)
    # same data type as the adjacency matrices
    dtype = type(lA[0][0, 0])

    # Initialize eigenvector register
    MV = np.empty(shape=(N, store), dtype=dtype)
    # first vector is a versor with the equal components:
    MV[:, 0] = np.array([1. / np.sqrt(N)] * N, dtype=dtype)
    # register counter
    c = 0
    # vector to work on:
    v = MV[:, c].copy()

    for k in range(T * valumax):
        # Perform iteration:
        v = -(loglad * lA[k % T]).expm1().dot(v) + (1. - mu) * v

        # Whether period is completed:
        if k % T == T - 1:

            # spectral radius
            sr = np.dot(MV[:, c % store], v)

            # normalize
            v = v / np.linalg.norm(v)

            # Compute tolerance, and return if reached:
            delta = np.sum(np.abs(MV[:, c % store] - v))
            if delta < tolerance:
                # return spectral radius^(1/T) - 1, as usual.
                return sr ** rootT - sr_target  # , v/np.linalg.norm(v)

            # increment index, and update storage
            c += 1
            MV[:, c % store] = v.copy()

    # if it goes out of the loop without returning the sr
    raise ThresholdError, 'Power method did not converge.'


# for the aggregated spectral radius
def psr2uw_agg(A, N, valumax, tolerance, store):
    # stop when the L1 norm of (v_{a}-v_{a-1}) is below tolerance, where a is the a-th loop on the period

    # same data type as the adjacency matrices
    dtype = type(A[0, 0])

    # Initialize eigenvector register
    MV = np.empty(shape=(N, store), dtype=dtype)
    # first vector is a versor with the equal components:
    MV[:, 0] = np.array([1. / np.sqrt(N)] * N, dtype=dtype)
    # register counter
    c = 0
    # vector to work on:
    v = MV[:, 0].copy()

    while c < valumax:
        # Perform iteration:
        v = A.dot(v)

        # spectral radius
        sr = np.dot(MV[:, c % store], v)

        # normalize
        v = v / np.linalg.norm(v)

        # Compute tolerance, and return if reached:
        delta = np.sum(np.abs(MV[:, c % store] - v))
        if delta < tolerance:
            # return spectral radius^(1/T) - 1, as usual.
            return sr

        # increment index, and update storage
        c += 1
        MV[:, c % store] = v.copy()

    # if it goes out of the loop without returning the sr
    raise ThresholdError, 'Power method did not converge.'

