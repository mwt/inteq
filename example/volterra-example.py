# -*- coding: utf-8 -*-

from inteq import SolveVolterra
import numpy as np
import matplotlib.pyplot as plt

# define kernel
def k(s, t):
    return np.cos(s - t)


# true value
def trueg(s):
    return (2 + s ** 2) / 2


s, g = SolveVolterra(k)

# plot functions
plt.plot(s, g)
plt.plot(s, trueg(s))

plt.legend(["Estimate", "True Value"])

plt.xlabel("s")
plt.ylabel("g(s)")

plt.savefig("..\\assets\\volterra-example.png")

