Fredholm Integral Equations
===========================

First Kind
----------

This package provides the function :func:`inteq.SolveFredholm` which approximates the solution, g(x), to the Fredholm Integral Equation of the first kind using the method described in `Twomey (1963)`_. It will return a smooth curve that is an approximate solution. However, it may not be a good approximate to the true solution.

.. autofunction:: inteq.SolveFredholm


Example
^^^^^^^

|example of first kind fie|

.. literalinclude:: ../../example/fredholm-example.py
    :language: python


Second Kind
-----------

Implementation forthcoming.
    

.. _`Twomey (1963)`: https://doi.org/10.1145/321150.321157
.. |example of first kind fie| image:: fredholm-example.svg

