# Lagrangian method for determining the equation of motion for a double pendulum


import sympy as smp
import matplotlib.pyplot as plt

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

sol_theta1 = smp.diff(L, theta_1) - smp.diff(smp.diff(L, theta_1dot), t).simplify()
sol_theta2 = smp.diff(L, theta_2) - smp.diff(smp.diff(L, theta_2dot), t).simplify()

sol_theta_1ddot = smp.solve(sol_theta1, theta_1ddot)[0].simplify()
sol_theta_2ddot = smp.solve(sol_theta2, theta_2ddot)[0].simplify()


