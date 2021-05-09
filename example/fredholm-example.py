# -*- coding: utf-8 -*-

from inteq import SolveFredholm
import numpy as np
import matplotlib.pyplot as plt

# define kernel
def k(s, t):
    return (np.abs(t - s) <= 3) * (1 + np.cos(np.pi * (t - s) / 3))


def f(s):
    sp = np.abs(s)
    sp3 = sp * np.pi / 3
    return ((6 - sp) * (2 + np.cos(sp3)) + (9 / np.pi) * np.sin(sp3)) / 2


# true value
def trueg(s):
    return k(0, s)


s, g = SolveFredholm(k, f, a=-3, b=3, num=1000, smin=-6, smax=6)

# plot functions
figure = plt.figure()

plt.plot(s, g)
plt.plot(s, trueg(s))

plt.legend(["Estimate", "True Value"])

plt.xlabel("s")
plt.ylabel("g(s)")

figure.set_dpi(100)
figure.set_size_inches(8, 5)

figure.savefig("..\\assets\\fredholm-example.svg", bbox_inches="tight")

