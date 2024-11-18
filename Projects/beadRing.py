import sympy as smp

omega, R, t, g, m = smp.symbols("\omega R t g m")
theta = smp.symbols(r"\theta", cls=smp.Function)
theta = theta(t)
thetadot = smp.diff(theta, t)
thetaddot = smp.diff(thetadot, t)


z = -R * smp.cos(theta)
x = R * smp.sin(theta) * smp.sin(omega * t)
y = R * smp.sin(theta) * smp.sin(omega * t)

T = smp.Rational(1, 2) * m * (smp.diff(x, t) ** 2 + smp.diff(y, t) ** 2)
V = m * g * z
L = T - V

LE = smp.diff(L,t) - smp.diff(smp.diff(L,thetadot),t)
sols = smp.solve(LE,thetaddot)[0].simplify()
