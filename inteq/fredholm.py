# -*- coding: utf-8 -*-

import typing
import numpy
from .helpers import makeH, simpson

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

    using the method described in Twomey (1963). It will return a smooth curve that is an approximate solution. However, it may not be a good approximate to the true solution.

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
    # need num to be odd to apply Simpson's rule
    if (num % 2) == 0:
        num += 1
    # get grid for s
    sgrid = numpy.linspace(a, b, num)
    # get quadrature weights
    weights = simpson(num)
    # create the quadrature matrix
    ksqur = weights * k(sgrid[:, numpy.newaxis], sgrid)
    # Make H matrix as in (Twomey 1963)
    if gamma != 0:
        Hmat = makeH(num)
        AAgH = (ksqur.T @ ksqur) + (gamma * Hmat)
    else:
        AAgH = ksqur.T @ ksqur
    # find the gvalues (/num) by solving the system of equations
    ggrid = numpy.linalg.solve(AAgH, ksqur.T @ f(sgrid))
    # combine the s grid and the g grid
    return numpy.array([sgrid, (ggrid * num / (b - a))])


# %%
