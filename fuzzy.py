import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def cmeans0(data, u_old, c, m):

    u_old /= np.ones((c, 1)).dot(np.atleast_2d(u_old.sum(axis=0)))
    u_old = np.fmax(u_old, np.finfo(np.float64).eps)

    um = u_old ** m
    # Calculate cluster centers
    data = data.T
    cntr = um.dot(data) / (np.ones((data.shape[1],
                                    1)).dot(np.atleast_2d(um.sum(axis=1))).T)

    d = distance(data, cntr)
    d = np.fmax(d, np.finfo(np.float64).eps)

    jm = (um * d ** 2).sum()

    u = d ** (- 2. / (m - 1))
    u /= np.ones((c, 1)).dot(np.atleast_2d(u.sum(axis=0)))

    return cntr, u, jm, d


def distance(data, centers):
    return cdist(data, centers).T


def fp_coeff(u):
    n = u.shape[1]
    return np.trace(u.dot(u.T)) / float(n)


def cmeans(data, c, m, error, maxiter, init=None, seed=None):
    if init is None:
        if seed is not None:
            np.random.seed(seed=seed)
        n = data.shape[1]
        u0 = np.random.rand(c, n)
        u0 /= np.ones(
            (c, 1)).dot(np.atleast_2d(u0.sum(axis=0))).astype(np.float64)
        init = u0.copy()
    u0 = init
    u = np.fmax(u0, np.finfo(np.float64).eps)
    # Initialize loop parameters
    jm = np.zeros(0)
    p = 0
    cntr=[]

    # Main cmeans loop
    while p <= maxiter - 1:
        u2 = u.copy()
        [cntr, u, Jjm, d] = cmeans0(data, u2, c, m)
        jm = np.hstack((jm, Jjm))
        p += 1

        # Stopping rule
        if np.linalg.norm(u - u2) < error:
            break

    # Final calculations
    error = np.linalg.norm(u - u2)
    fpc = fp_coeff(u)

    return cntr, u, u0, d, jm, p, fpc


def cmeans_predict(test_data, cntr_trained, m, error, maxiter, init=None,seed=None):
    c = cntr_trained.shape[0]
    # Setup u0
    if init is None:
        if seed is not None:
            np.random.seed(seed=seed)
        n = test_data.shape[1]
        u0 = np.random.rand(c, n)
        u0 /= np.ones(
            (c, 1)).dot(np.atleast_2d(u0.sum(axis=0))).astype(np.float64)
        init = u0.copy()
    u0 = init
    u = np.fmax(u0, np.finfo(np.float64).eps)

    # Initialize loop parameters
    jm = np.zeros(0)
    p = 0

    # Main cmeans loop
    while p < maxiter - 1:
        u2 = u.copy()
        [u, Jjm, d] = cmeans_predict0(test_data, cntr_trained, u2, c, m)
        jm = np.hstack((jm, Jjm))
        p += 1
        # Stopping rule
        if np.linalg.norm(u - u2) < error:
            break

    # Final calculations
    error = np.linalg.norm(u - u2)
    fpc = fp_coeff(u)
    return u, u0, d, jm, p, fpc


def cmeans_predict0(test_data, cntr, u_old, c, m):
    # Normalizing, then eliminating any potential zero values.
    u_old /= np.ones((c, 1)).dot(np.atleast_2d(u_old.sum(axis=0)))
    u_old = np.fmax(u_old, np.finfo(np.float64).eps)

    um = u_old ** m
    test_data = test_data.T

    # For prediction, we do not recalculate cluster centers. The test_data is
    # forced to confirm to the prior clustering.

    d = distance(test_data, cntr)
    d = np.fmax(d, np.finfo(np.float64).eps)

    jm = (um * d ** 2).sum()

    u = d ** (- 2. / (m - 1))
    u /= np.ones((c, 1)).dot(np.atleast_2d(u.sum(axis=0)))

    return u, jm, d

def fuzzy():
    fp = open("encodedata.csv")
    data = []
    for row in range (3000):
        new = []
        line = fp.readline()
        line = line.split(",")
        for i in line:
              new.append(i)
        data.append(new)
    data=zip(*data)
    data = np.asarray(data,dtype='float64')

    cntr, u, u0, d, jm, p, fpc=cmeans(data,2,2,0.00001,50)




    data = []
    for row in range(3000,3780):
        new = []
        line = fp.readline()
        line = line.split(",")
        for i in line:
              new.append(i)
        data.append(new)
    data = zip(*data)
    fp.close()
    data = np.asarray(data,dtype='float64')

    u, u0, d, jm, p, fpc=cmeans_predict(data,cntr,2,0.00001,50)