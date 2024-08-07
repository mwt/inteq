"""
Solve various integral equations using numerical methods.
"""

from .fredholm import solve as SolveFredholm
from .volterra import solve as SolveVolterra
from .volterra import solve2 as SolveVolterra2