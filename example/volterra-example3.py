import os

import matplotlib.pyplot as plt
import numpy as np

from inteq import SolveVolterra2

def k(s, t):
    return 0.5 * (t-s)**2 * np.exp(s-t)
def free(t):
    return -0.5 * t**2 * np.sin(t) * np.exp(-t/2)

def true(t):
    return 1/291762750 * np.exp( -3/2 * t ) * ( -3 * np.exp( t ) \
        * ( 16 * ( 2390928 + 365 * t * ( -9936 + 365 * t ) ) * np.cos( t ) \
        + 3 * ( 6965888 + 365 * t * ( 24544 + 25915 * t ) ) * np.sin( t ) \
        ) + 32 * ( 389017 * ( np.e )**( 3/2 * t ) + ( 3197375 * np.cos( \
        1/2 * ( 3 )**( 1/2 ) * t ) + -318625 * ( 3 )**( 1/2 ) * np.sin( \
        1/2 * ( 3 )**( 1/2 ) * t ) ) ) )

s, gmid = SolveVolterra2(k, free, a=0.0, b=20.0, num=100, method="midpoint")
s, gtrap = SolveVolterra2(k, free, a=0.0, b=20.0, num=100, method="trapezoid")
s, gsimp = SolveVolterra2(k, free, a=0.0, b=20.0, num=100, method="simpson")
s, ggreg = SolveVolterra2(k, free, a=0.0, b=20.0, num=100,
                          method="gregory", greg_order=7)

figure, axs = plt.subplots(2, 1)
axs[0].plot(s, true(s))

axs[1].semilogy(s, np.abs(gmid-true(s)))
axs[1].semilogy(s, np.abs(gtrap-true(s)))
axs[1].semilogy(s, np.abs(gsimp-true(s)))
axs[1].semilogy(s, np.abs(ggreg-true(s)))

axs[1].legend(["Midpoint", "Trapezoid", "Simpson", "Gregory(7)", "Exact"], loc="upper right")
axs[0].set_xlabel("t")
axs[0].set_ylabel("f(t)")
axs[1].set_xlabel("t")
axs[1].set_ylabel("Absolute error")

figure.set_dpi(100)
figure.set_size_inches(8, 5)

save_path = os.path.join(os.path.dirname(__file__),
                         "..", "docs", "volterra", "volterra-example3.svg")
figure.savefig(save_path, bbox_inches="tight")