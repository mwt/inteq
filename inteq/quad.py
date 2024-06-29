#  SPDX-FileCopyrightText: 2024 Christopher Hillenbrand

#  SPDX-License-Identifier: MIT

# MIT License

# Copyright (c) 2024 Christopher Hillenbrand

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from fractions import Fraction
from functools import cache

import numpy as np
import scipy.linalg as sla


@cache
def gregory_coef(k):
    r"""Gregory coefficients.

    See https://en.wikipedia.org/wiki/Gregory_coefficients.

    Starting from :math:`G_0`, the first few coefficients are:
    :math:`-\frac{1}{2}, \frac{1}{12}, -\frac{1}{24}, \frac{19}{720}, \ldots`

    .. math::
        G_k = \frac{-1}{(k+1)!} \left[ \frac{d^{k+1}}{dz^{k+1}}\frac{z}{\ln(1+z)}\right]_{z=0}

    Parameters
    ----------
    k : int
        (starts from zero)

    Returns
    -------
    :class:`fraction.Fraction`
        kth Gregory coefficient
    """
    assert k >= 0
    if k == 0:
        Fraction(-1, 2)
    grn = Fraction(1, k + 2)
    for r in range(k):
        grn -= abs(gregory_coef(r)) * Fraction(1, k + 1 - r)
    return grn * (-1) ** (k + 1)


@cache
def gregory_weights(k):
    r"""Gregory quadrature weights.

    See equation 14 in
    Fornberg, B., Reeger, J.A. An improved Gregory-like method for 1-D quadrature.
    Numer. Math. 141, 1–19 (2019). https://doi-org.yale.idm.oclc.org/10.1007/s00211-018-0992-0

    .. math::
        \mathbf{w} =
        \begin{bmatrix}
        1 & 0 & 0 & 0 & \cdots \\
        -1 & 1 & 0 & 0 & \cdots \\
        1 & -2 & 1 & 0 & \cdots \\
        -1 & 3 & -3 & 1 & \cdots \\
        \vdots & \vdots & \vdots & \vdots & \ddots \\
        \end{bmatrix}
        \begin{bmatrix}
        G_0 \\
        G_1 \\
        G_2 \\
        G_3 \\
        \vdots
        \end{bmatrix}
        + \begin{bmatrix}
        1 \\
        1 \\
        1 \\
        1 \\
        \vdots
        \end{bmatrix}

    Parameters
    ----------
    k : int
        Starts from 0; gives degree of polynomial approx.

    Returns
    -------
    w : (k+1,) ndarray
        Order k Gregory quadrature weights, exact.
    """
    gr_coef = np.array([gregory_coef(r) for r in range(k + 1)])
    wts_exact = sla.invpascal(k + 1, kind="upper", exact=True) @ gr_coef + 1
    wts = np.asarray(wts_exact, dtype=np.float64)
    wts.setflags(write=False)
    return wts


@cache
def sigma_weights(k):
    """Sigma quadrature weights (exact).
    
    See equation 98 in
    
    Schüler, M.; Golež, D.; Murakami, Y.; Bittner, N.; Herrmann, A.;
    Strand, H. U. R.; Werner, P.; Eckstein, M. NESSi:
    The Non-Equilibrium Systems Simulation Package.
    Computer Physics Communications, 2020, 257, 107484.
    https://doi.org/10.1016/j.cpc.2020.107484.
    
    See also
    
    P. H. M. Wolkenfelt, The Construction of Reducible Quadrature Rules for
    Volterra Integral and Integro-differential Equations, IMA Journal of
    Numerical Analysis, Volume 2, Issue 2, April 1982, Pages 131–152,
    https://doi.org/10.1093/imanum/2.2.131.

    Parameters
    ----------
    k : int
        Starts from 0; gives degree of polynomial approx.

    Returns
    -------
    sigma : (k+1, 2k+2) ndarray
        Order k sigma quadrature weights, exact.
    """

    omega = gregory_weights(k)
    sigma = np.zeros((k + 1, 2 * k + 2), dtype=object)
    for i in range(1, k + 2):
        sigma[i - 1, : k + 1] += omega
        sigma[i - 1, i : k + 1] += np.flip(omega[i:]) - 1
        sigma[i - 1, k + 1 : k + 1 + i] = np.flip(omega[:i])
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma.setflags(write=False)
    return sigma


@cache
def poly_integration_weights(k):
    r"""Polynomial integration weights.

    .. math::
        I^{(k)} = \begin{bmatrix}
            0 & 0 & 0 & \cdots & 0 \\
            \frac{1}{1} & \frac{1}{2} & \frac{1}{3} & \cdots & \frac{1}{k+1} \\
            \frac{2^1}{1} & \frac{2^2}{2} & \frac{2^3}{3} & \cdots & \frac{2^{k+1}}{k+1} \\
            \vdots & \vdots & \vdots & \ddots & \vdots \\
            \frac{k^1}{1} & \frac{k^2}{2} & \frac{k^3}{3} & \cdots & \frac{k^{k+1}}{k+1} \\
        \end{bmatrix} (V^{(k)})^{-1}

    Parameters
    ----------
    k : int
        Polynomial degree.

    Returns
    -------
    I : (k+1,k+1) ndarray
        Polynomial integration weights.
    """
    grid = np.arange(k + 1)
    aux = np.vander(np.arange(k+1), increasing=True) / (grid+1) * grid[:, np.newaxis]
    wts = aux @ np.linalg.inv(np.vander(np.arange(k+1), increasing=True))
    wts.setflags(write=False)
    return wts


@cache
def gregory_weight_matrix(k, n):
    """Weight matrix for Gregory quadrature.
    
    See equation 98 in

    Schüler, M.; Golež, D.; Murakami, Y.; Bittner, N.; Herrmann, A.;
    Strand, H. U. R.; Werner, P.; Eckstein, M. NESSi:
    The Non-Equilibrium Systems Simulation Package.
    Computer Physics Communications, 2020, 257, 107484.
    https://doi.org/10.1016/j.cpc.2020.107484.

    Parameters
    ----------
    k : int
        Polynomial degree.
    n : int
        Row (or upper limit of integration).

    Returns
    -------
    w : (n+1,n+1) ndarray
        Quadrature weights.
    """
    nrow_max = max(n + 1, 2 * k + 2)
    ncol_min = min(n + 1, 2 * k + 2)
    assert n >= k
    mat = np.zeros((nrow_max, n + 1), dtype=np.float64)
    mat[:k + 1, :k + 1] = poly_integration_weights(k)
    mat[k + 1 : 2 * k + 2, :ncol_min] = sigma_weights(k)[:, :ncol_min]
    for r in range(2 * k + 2, n + 1):
        mat[r, :r + 1] = 1
        mat[r, :k + 1] = gregory_weights(k)
        mat[r, r - k : r + 1] = gregory_weights(k)[::-1]
    mat = mat[:n + 1]
    mat.setflags(write=False)
    return mat


@cache
def simpson_quad(n):
    """Repeated Simpson + Simpson's 3/8 rule
    
    Example 3.2.1 in

    H. Brunner & P. J. Van der Houwen, The Numerical Solution of
    Volterra Equations, CWI Monographs, vol. 3, North-Holland, Amsterdam, 1986.

    Parameters
    ----------
    n : int
        Row (or upper limit of integration).

    Returns
    -------
    w : (n+1,n+1) ndarray
        Quadrature weights.
    """
    arr = np.zeros((n + 1, n + 1), dtype=np.float64)
    arr[1, 0] = 5/12
    arr[1, 1] = 8/12
    arr[1, 2] = -1/12
    array_3993 = np.array([3/8, 9/8, 9/8, 3/8], dtype=np.float64)
    arr[2:, 0] = 1/3
    for row in range(2, n + 1, 2):
        arr[row, 1:(2*row//2+1)] = np.tile([4/3, 2/3], row//2)
        arr[row, row] = 1/3
    for row in range(3, n + 1, 2):
        np.copyto(arr[row], arr[row - 3])
        arr[row, row-3:row+1] += array_3993
    arr.setflags(write=False)
    return arr
