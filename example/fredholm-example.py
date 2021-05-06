# -*- coding: utf-8 -*-

from inteq import SolveFredholm
import matplotlib.pyplot as plt

# define kernel
def k(s, t):
    return s * t


def f(s):
    return s


# true value
def trueg(s):
    return s


s, g = SolveFredholm(k, f, a=0, b=1, num=60, gamma=1e-10)

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

