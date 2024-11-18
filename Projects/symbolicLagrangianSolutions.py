import sympy as smp
import vpython as vp
import numpy as np
import matplotlib.pyplot as plt


# Defining variables
t, A, g, R, omega, m = smp.symbols("t A g R \omega m")
theta = smp.symbols(r"\theta", cls=smp.Function)
theta = theta(t)
thetadot = smp.diff(theta, t)
thetaddot = smp.diff(thetadot, t)

x = R * smp.sin(theta)
y = -R * smp.cos(theta)
T = smp.Rational(1, 2) * m * (smp.diff(x, t) ** 2 + smp.diff(y, t) ** 2)
V = m * g * y

L = T-V
L.simplify()
LE = smp.diff(L,theta)-smp.diff(smp.diff(L,thetadot),t)
sols = smp.solve(LE, thetaddot)
sols[0].simplify()

