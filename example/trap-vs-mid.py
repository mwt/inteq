import matplotlib.pyplot as plt
import numpy as np

from inteq import SolveVolterra

## trapezoid does well


# define kernel
def k(s, t):
    return np.cos(s - t)


# true value
def trueg(s):
    return (2 + s**2) / 2


st, gt = SolveVolterra(k, method="trapezoid", num=6)
sm, gm = SolveVolterra(k, method="midpoint", num=6)

# plot functions
figure = plt.figure()

plt.plot(st, gt)
plt.plot(sm, gm)
plt.plot(sm, trueg(sm))

plt.legend(["Trapezoid", "Midpoint", "True Value"])

plt.xlabel("s")
plt.ylabel("g(s)")

figure.set_dpi(100)
figure.set_size_inches(8, 5)

figure.savefig("..\\docs\\volterra\\trap-vs-mid1.svg", bbox_inches="tight")

## trapezoid does poorly


# define kernel
def k(s, t):
    return 1 / (s + t)


st, gt = SolveVolterra(k, method="trapezoid", num=100)
sm, gm = SolveVolterra(k, method="midpoint", num=100)

# plot functions
figure = plt.figure()

plt.plot(st, gt)
plt.plot(sm, gm)

plt.legend(["Trapezoid", "Midpoint"])

plt.xlabel("s")
plt.ylabel("g(s)")

figure.set_dpi(100)
figure.set_size_inches(8, 5)

figure.savefig("..\\docs\\volterra\\trap-vs-mid2.svg", bbox_inches="tight")
