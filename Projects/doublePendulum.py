# Lagrangian method for determining the equation of motion for a double pendulum

import sympy as smp
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np

m_1, m_2, R_1, R_2, g, t = smp.symbols("m_{1} m_{2} R_{1} R_{2} g t")

theta_1 = smp.symbols(r"\theta_{1}", cls=smp.Function)
theta_1 = theta_1(t)
theta_1dot = smp.diff(theta_1, t)
theta_1ddot = smp.diff(theta_1dot, t)

theta_2 = smp.symbols(r"\theta_{2}", cls=smp.Function)
theta_2 = theta_2(t)
theta_2dot = smp.diff(theta_2, t)
theta_2ddot = smp.diff(theta_2dot, t)

x_1 = R_1 * smp.sin(theta_1)
y_1 = -R_1 * smp.cos(theta_1)

x_2 = x_1 + R_2 * smp.sin(theta_2)
y_2 = y_1 - R_2 * smp.cos(theta_2)

x_1dot = smp.diff(x_1, t)
x_2dot = smp.diff(x_2, t).simplify()

y_1dot = smp.diff(y_1, t)
y_2dot = smp.diff(y_2, t).simplify()

T_1 = smp.Rational(1, 2) * m_1 * (x_1dot**2 + y_1dot**2).simplify()
T_2 = smp.Rational(1, 2) * m_2 * (x_2dot**2 + y_2dot**2).simplify()

V_1 = m_1 * g * y_1
V_2 = m_2 * g * y_2

T = T_1 + T_2
V = V_1 + V_2

L = T - V

LE1 = smp.diff(L, theta_1) - smp.diff(smp.diff(L, theta_1dot), t).simplify()
LE2 = smp.diff(L, theta_2) - smp.diff(smp.diff(L, theta_2dot), t).simplify()

sols = smp.solve([LE1, LE2], [theta_1ddot, theta_2ddot], simplify=False, rational=False)

# creating functions refer to numericalSolutionsNotes.ipynb for explanation and better notes

dz1dt_f = smp.lambdify(
    (t, g, m_1, m_2, R_1, R_2, theta_1, theta_2, theta_1dot, theta_2dot),
    sols[theta_1ddot],
)
dz2dt_f = smp.lambdify(
    (t, g, m_1, m_2, R_1, R_2, theta_1, theta_2, theta_1dot, theta_2dot),
    sols[theta_2ddot],
)
# Next two are just identif functions, but they are part of making python use 4 first order ODEs
dtheta_1dt_f = smp.lambdify(theta_1dot, theta_1dot)
dtheta_2dt_f = smp.lambdify(theta_2dot, theta_2dot)


#
def dSdt(S, t, g, m1, m2, R1, R2):
    theta_1, z1, theta_2, z2 = S
    return [
        dtheta_1dt_f(z1),
        dz1dt_f(t, g, m1, m2, R1, R2, theta_1, theta_2, z1, z2),
        dtheta_2dt_f(z2),
        dz2dt_f(t, g, m1, m2, R1, R2, theta_1, theta_2, z1, z2),
    ]


t = np.linspace(0, 40, 1001)
g = 9.81
m1 = 2
m2 = 3
R1 = 3
R2 = 4
# y0 = initial conditions (theta, theta1dot theta2 theta2dot) t=t tells it to use the t generated above and args are args
ans = odeint(dSdt, y0=[1,-3, 0, 1], t=t, args=(g,m1,m2,R1,R2))

# transposing the array allows you to break up each block into one variable 

plt.plot(t,ans.T[2])

# Once we have this information we can switch to using vpython for the fun stuff