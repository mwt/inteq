Volterra Integral Equations
===========================


First Kind
----------

This package provides the function :func:`inteq.SolveVolterra` which approximates the solution to the Volterra Integral Equation of the first kind using the method in `Betto and Thomas (2021)`_. 

.. autofunction:: inteq.SolveVolterra


Example
^^^^^^^

|example of first kind vie|

.. literalinclude:: ../../example/volterra-example.py
    :language: python

|example of second kind vie|

.. literalinclude:: ../../example/volterra-example2.py
    :language: python


Trapezoid vs Midpoint Rule
^^^^^^^^^^^^^^^^^^^^^^^^^^

Volterra integral equations are typically solved using the midpoint
rule. However, the trapezoid rule often converges faster. See below an
example of the trapezoid rule performing well with just six grid points.

|example of trapezoid rule converging faster|

Thus, the trapezoid rule typically performs better. However, the
trapezoid rule is less stable than the midpoint rule. An example where
this this instability is an issue is provided below.

|example of trapezoid rule having issues|

This can be remedied by smoothing the function. For example, with
:func:`inteq.helpers.smooth`.


Second Kind
-----------

The function :func:`inteq.SolveVolterra2` approximates the solution to the Volterra Integral Equation of the second kind using either the midpoint or trapezoid rule.

.. autofunction:: inteq.SolveVolterra


.. _`Betto and Thomas (2021)`: https://mattwthomas.com/papers/asymmetric-all-pay-contests-with-spillovers/
.. |example of first kind vie| image:: volterra-example.svg
.. |example of second kind vie| image:: volterra-example2.svg
.. |example of trapezoid rule converging faster| image:: trap-vs-mid1.svg
.. |example of trapezoid rule having issues| image:: trap-vs-mid2.svg