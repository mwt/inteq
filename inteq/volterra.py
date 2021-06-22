# -*- coding: utf-8 -*-

import typing
import numpy
import scipy.linalg

from .helpers import makeH

#%%


def solve(
    k: typing.Callable[[numpy.ndarray, numpy.ndarray], numpy.ndarray],
    f: typing.Callable[[numpy.ndarray], numpy.ndarray] = lambda x: x,
    a: float = 0.0,
    b: float = 1.0,
    num: int = 1000,
    method: str = "midpoint",
) -> numpy.ndarray:
    """
    Approximate the solution, g(x), to the Volterra Integral Equation of the first kind:

    .. math::
        f(s) = \\int_a^s K(s,y) g(y) dy

    using the method in Betto and Thomas (2021).

    Parameters
    ----------
    k : function
        The kernel function that takes two arguments.
    f : function
        The left hand side (free) function with f(a) = 0.
    a : float
        Lower bound of the integral, defaults to 0.
    b : float
        Upper bound of the estimate, defaults to 1.
    num : int
        Number of estimation points between zero and `b`.
    method : string
        Use either the 'midpoint' (default) or 'trapezoid' rule.

    Returns
    -------
    grid : 2-D array
        Input values are in the first row and output values are in the second row.
    """
    if not isinstance(num, int):
        num = int(num)
    # make a grid of `num` points from (eps > 0) to `b`
    sgrid = numpy.linspace(a + (b - a) / num, b, num)
    # create a lower triangular matrix of kernel values
    ktril = numpy.tril(k(sgrid[:, numpy.newaxis], sgrid))
    if method is "midpoint":
        pass
    elif method is "trapezoid":
        # apply trapezoid rule by halving the endpoints
        numpy.fill_diagonal(ktril, numpy.diag(ktril) / 2)
        # remember that 0,0 was already halved in the diagonal
        ktril[:, 0] = ktril[:, 0] + k(sgrid, 0) / 2
    else:
        raise Exception("method must be one of 'midpoint', 'trapezoid'")
    # find the gvalues (/num) by solving the system of equations
    ggrid = scipy.linalg.solve_triangular(
        ktril, f(sgrid), lower=True, check_finite=False
    )
    # combine the s grid and the g grid
    return numpy.array([sgrid, (ggrid * num / (b - a))])


# %%


def solve2(
    k: typing.Callable[[numpy.ndarray, numpy.ndarray], numpy.ndarray],
    f: typing.Callable[[numpy.ndarray], numpy.ndarray] = lambda x: x,
    a: float = 0.0,
    b: float = 1.0,
    num: int = 1000,
    method: str = "midpoint",
) -> numpy.ndarray:
    """
    Approximate the solution, g(x), to the Volterra Integral Equation of the second kind:

    .. math::
        g(s) = f(s) + \\int_a^s K(s,y) g(y) dy

    using the methods in Linz (1969).

    Parameters
    ----------
    k : function
        The kernel function that takes two arguments.
    f : function
        The free function.
    a : float
        Lower bound of the integral, defaults to 0.
    b : float
        Upper bound of the estimate, defaults to 1.
    num : int
        Number of estimation points between zero and `b`.
    method : string
        Use either the 'midpoint' (default) or 'trapezoid' rule.

    Returns
    -------
    grid : 2-D array
        Input values are in the first row and output values are in the second row.
    """
    if not isinstance(num, int):
        num = int(num)
    # make a grid of `num` points from (eps > 0) to `b`
    sgrid = numpy.linspace(0, b, num)
    # create a lower triangular matrix of kernel values
    ktril = numpy.tril(k(sgrid[:, numpy.newaxis], sgrid))
    # create matrix to invert
    ktril = numpy.diag(num) - ktril
    if method is "midpoint":
        pass
    elif method is "trapezoid":
        # apply trapezoid rule by halving the endpoints
        numpy.fill_diagonal(ktril, numpy.diag(ktril) / 2)
        # remember that 0,0 was already halved in the diagonal
        ktril[:, 0] = ktril[:, 0] + k(sgrid, 0) / 2
    else:
        raise Exception("method must be one of 'midpoint', 'trapezoid'")
    # find the gvalues (/num) by solving the system of equations
    ggrid = scipy.linalg.solve_triangular(
        ktril, f(sgrid), lower=True, check_finite=False
    )
    # combine the s grid and the g grid
    return numpy.array([sgrid, (ggrid * num / (b - a))])
