# -*- coding: utf-8 -*-

import typing
import numpy

#%% Helpers


def makeH(dim: int) -> numpy.ndarray:
    """
    Make H matrix for estimating Fredholm equations as in (Twomey 1963).

    Parameters
    ----------
    dim : int
        The dimension of the H matrix.

    Returns
    -------
    Hmat : 2-D array
        The H matrix 
    """
    # create (symmetric) diagonal vectors
    d2 = numpy.ones(dim - 2)
    d1 = numpy.concatenate(([-2], numpy.repeat(-4, dim - 3), [-2]))
    d0 = numpy.concatenate(([1, 5], numpy.repeat(6, dim - 4), [5, 1]))
    # make matrix with diagonals
    Hmat = numpy.diag(d1, 1) + numpy.diag(d2, 2)
    Hmat = Hmat + Hmat.T
    numpy.fill_diagonal(Hmat, d0)
    return Hmat


#%%


def solve(
    k: typing.Callable[[numpy.ndarray, numpy.ndarray], numpy.ndarray],
    f: typing.Callable[[numpy.ndarray], numpy.ndarray] = lambda x: x,
    a: float = -1.0,
    b: float = 1.0,
    gamma: float = 1e-3,
    num: int = 40,
) -> numpy.ndarray:
    """
    Approximate the solution, g(x), to the Fredholm Integral Equation of the first kind:
    
    $$
    f(s) = \\int_0^b K(s,y) g(y) dy
    $$

    This may fail if the solution is not unique.

    Parameters
    ----------
    k : function
        The kernel function that takes two arguments.
    f : function 
        The left hand side (free) function that takes one argument.
    a : float
        Lower bound of the of the Fredholm definite integral, defaults to -1.
    b : float
        Upper bound of the of the Fredholm definite integral, defaults to 1.
    num : int
        Number of estimation points between zero and `b`.

    Returns
    -------
    grid : 2-D array
        Input values are in the first row and output values are in the second row.
    """
    if not isinstance(num, int):
        num = int(num)
    # get grid for s
    sgrid = numpy.linspace(a, b, num)
    # get quadrature points and weights
    ygrid, weights = numpy.polynomial.legendre.leggauss(num)
    # change of variables to [-1,1]
    if (a != -1) or (b != 1):
        ygrid = ((b - a) / 2) * ygrid + ((a + b) / 2)
        weights = ((b - a) / 2) * weights
    # create the quadrature matrix
    ksqur = weights * k(sgrid[:, numpy.newaxis], ygrid)
    # Make H matrix as in (Twomey 1963)
    if gamma != 0:
        Hmat = makeH(num)
        AAgH = (ksqur.T @ ksqur) + (gamma * Hmat)
    else:
        AAgH = ksqur.T @ ksqur
    # find the gvalues (/num) by solving the system of equations
    ggrid = numpy.linalg.solve(AAgH, ksqur.T @ f(sgrid) )
    # combine the s grid and the g grid
    return numpy.array([ygrid, ggrid])


# %%

