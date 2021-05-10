from inteq import SolveVolterra
import numpy as np
import matplotlib.pyplot as plt

# define kernel
def k(s, t):
    return np.cos(s - t)


# define true solution
def trueg(s):
    return (2 + s ** 2) / 2


# look decent with just 8 grid points but we'll use 25
s, g = SolveVolterra(k, num=25, method="trapezoid")

# plot functions
figure = plt.figure()

plt.plot(s, g)
plt.plot(s, trueg(s))

plt.legend(["Estimate", "True Value"])

plt.xlabel("s")
plt.ylabel("g(s)")

figure.set_dpi(100)
figure.set_size_inches(8, 5)

figure.savefig("..\\docs\\volterra\\volterra-example.svg", bbox_inches="tight")
