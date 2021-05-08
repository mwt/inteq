import numpy


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


def smooth(v):
    """
    Smooth a vector that is oscillating.

    Parameters
    ----------
    v : 1-D array
        The oscillating vector you want to smooth.

    Returns
    -------
    sv : 1-D array
        The smoothed vector.
    """
    dim = len(v)
    smat = numpy.diag(
        numpy.concatenate(([1], numpy.repeat(0.5, dim - 1)))
    ) + numpy.diag(numpy.repeat(0.5, dim - 1), -1)
    return smat @ v
