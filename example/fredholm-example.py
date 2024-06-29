import os

import matplotlib.pyplot as plt
import numpy as np

from inteq import SolveFredholm


# define kernel
def k(s, t):
    return (np.abs(t - s) <= 3) * (1 + np.cos(np.pi * (t - s) / 3))


# define free function
def f(s):
    sp = np.abs(s)
    sp3 = sp * np.pi / 3
    return ((6 - sp) * (2 + np.cos(sp3)) + (9 / np.pi) * np.sin(sp3)) / 2


# define true solution
def trueg(s):
    return k(0, s)


# apply the solver
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

save_path = os.path.join(os.path.dirname(__file__),
                            "..", "docs", "fredholm", "fredholm-example.svg")
figure.savefig(save_path, bbox_inches="tight")
