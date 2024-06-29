"""
This module provides functions for solving Volterra integral equations of the first and second kind.
"""

import typing

import numpy
import scipy.linalg
from .quad import simpson_quad, gregory_weight_matrix


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
    if method == "midpoint":
        pass
    elif method == "trapezoid":
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


def solve2(
    k: typing.Callable[[numpy.ndarray, numpy.ndarray], numpy.ndarray],
    f: typing.Callable[[numpy.ndarray], numpy.ndarray] = lambda x: x,
    a: float = 0.0,
    b: float = 1.0,
    num: int = 1000,
    method: str = "midpoint",
    greg_order: int = 3,
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
        Quadrature method: 'midpoint' (default), 'trapezoid', 'simpson', or 'gregory'.
    greg_order: int
        Order of the Gregory quadrature rule. Used only if `method` is 'gregory'.

    Returns
    -------
    grid : 2-D array
        Input values are in the first row and output values are in the second row.
    """
    if not isinstance(num, int):
        num = int(num)
    # make a grid of `num` points from (eps > 0) to `b`
    sgrid = numpy.linspace(a, b, num)
    h = (b - a) / (num - 1)
    # create a lower triangular matrix of kernel values
    kmat = k(sgrid, sgrid[:, numpy.newaxis])
    fgrid = f(sgrid)

    if method == "midpoint":
        kmat = numpy.tril(kmat)
        # first entry is exactly g(a) = f(a)
        kmat[0, 0] = 0
        ggrid = scipy.linalg.solve_triangular(
            numpy.eye(num) - h * kmat, fgrid, lower=True, check_finite=False
        )
    elif method == "trapezoid":
        # apply trapezoid rule by halving the left endpoints
        kmat = numpy.tril(kmat)
        kmat[:, 0] *= 1/2
        # apply trapezoid rule by halving the right endpoints (0,0 gets fixed later)
        numpy.fill_diagonal(kmat, numpy.diag(kmat) / 2)
        kmat[0, 0] = 0
        ggrid = scipy.linalg.solve_triangular(
            numpy.eye(num) - h * kmat, fgrid, lower=True, check_finite=False
        )
    elif method == 'simpson':
        kmat *= simpson_quad(num - 1)
        ggrid = scipy.linalg.solve(-h*kmat+numpy.eye(num), fgrid)
    elif method == 'gregory':
        kmat *= gregory_weight_matrix(greg_order, num - 1)
        ggrid = scipy.linalg.solve(-h*kmat+numpy.eye(num), fgrid)
    else:
        msg = f"Invalid method: {method}."
        raise ValueError(msg)

    # find the gvalues (/num) by solving the system of equations
    # combine the s grid and the g grid
    return numpy.array([sgrid, ggrid])
