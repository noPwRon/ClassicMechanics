import sympy as smp

m_cart, m_bob, L, g, k, t = smp.symbols("m_{cart} m_{bob} L g k t")

theta = smp.symbols(r"\theta", cls=smp.Function)
theta = theta(t)
thetadot = smp.diff(theta, t)
thetaddot = smp.diff(thetadot, t)

x_cart = smp.symbols(r"x_{cart}", cls=smp.Function)
x_cart = x_cart(t)
x_cartdot = smp.diff(x_cart, t)
x_cartddot = smp.diff(x_cartdot, t)

x_pend = x_cart + L * smp.cos(theta)
y_pend = L * smp.sin(theta)

T_cart = smp.Rational(1, 2) * m_cart * x_cartdot**2
T_pend = (
    smp.Rational(1, 2) * m_bob * (smp.diff(x_pend, t) ** 2 + smp.diff(y_pend, t) ** 2)
)
T = T_cart + T_pend

V_cart = smp.Rational(1, 2) * k * x_cart**2
V_pend = m_bob * g * y_pend
V = V_cart + V_pend

L = T - V
LE = smp.diff(L, theta) - smp.diff(smp.diff(L, thetadot), t)
sols = smp.solve(LE, thetaddot)
sols[0].simplify()
