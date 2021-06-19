---
title: 'Inteq: Solve Volterra and Fredholm integral equations in Python'
tags:
  - Python
  - mathematics
  - economics
  - numerical
authors:
  - name: Matthew W. Thomas^[corresponding author]
    affiliation: 1
affiliations:
 - name: Matthew Wildrick Thomas, Economics PhD Candidate, Northwestern University
   index: 1
date: 18 June 2021
bibliography: paper.bib
---

# Summary

Volterra and Fredholm integral equations describe important processes in
fluid dynamics, signal processing, physics, computer graphics, actuarial science,
and economics. Most of these equations do not admit closed form solutions. As a
result, these equations must be approximated numerically. Due to the importance
of these equations in a variety of fields, there is a requirement for efficient
numerical tools which can be used across fields.

# Statement of need

`Inteq` is a Python package that efficiently constructs numerical approximations
of Fredholm and Volterra integral equations. Despite the wide interdisciplinary
applicability of these integral forms, there is currently no pre-built solution
for solving them. Existing solutions are typically purpose built by practitioners
for specific problems and are not public.

None of the algorithms implemented in `Inteq` are entirely novel. However, there
are several enhancements of existing methods. 


Fredholm integral equations of the first kind take the form:
$$
f(s)=\int_a^b K(s,y)g(y)dy
$$
where you want to solve for $g$. For Fredholm integral equations of the first kind,
we apply the methods of @Twomey:1963 which is an improved version of @Phillips:1962.
The method can be applied using Simpsons rule, the trapezoid rule, the midpoint rule,
or Gaussian quadrature depending on the properties of the functions. Volterra integral
equations of the first kind take the form:
$$
f(s)=\int_a^s K(s,y)g(y)dy
$$
where we want to solve for $g$. We solve Fredholm integral equations using the methods
in @Betto:2021 and @Linz:1969.

`Inteq` is a dependency of the `allpy` package which uses Volterra integral equations
to find the equilibria of all-pay auctions. Both packages are used in @Betto:2021.

# References
