import os

import matplotlib.pyplot as plt
import numpy as np

from inteq import SolveVolterra2

def k2(s, t):
    return 0.5 * (t-s)**2 * np.exp(s-t)

def free(t):
    return 0.5 * t**2 * np.exp(-t)

def true(t):
    return 1/3*(1-np.exp(-3*t/2)*(np.cos(np.sqrt(3)/2 * t) + np.sqrt(3) * np.sin(np.sqrt(3)/2 * t)))

s, gmid = SolveVolterra2(k2, free, a=0.0, b=6.0, num=25, method="midpoint")
s, gtrap = SolveVolterra2(k2, free, a=0.0, b=6.0, num=25, method="trapezoid")


figure = plt.figure()
plt.plot(s, gmid)
plt.plot(s, gtrap)
plt.plot(s, true(s))

plt.legend(["Midpoint", "Trapezoid", "Exact"])
plt.xlabel("t")
plt.ylabel("f(t)")

figure.set_dpi(100)
figure.set_size_inches(8, 5)

save_path = os.path.join(os.path.dirname(__file__),
                         "..", "docs", "volterra", "volterra-example2.svg")
figure.savefig(save_path, bbox_inches="tight")