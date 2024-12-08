import sympy as smp

m1, m2, g, l, s, t = smp.symbols(f"m_1 m_2 g l s t")

x1 = smp.symbols(f"x_1", cls=smp.Function)
x1 = x1(t)
x1Dot = smp.diff(x1, t)
x1DDot = smp.diff(x1Dot, t)

x2 = smp.symbols(f"x_2", cls=smp.Function)
x2 = x2(t)
x2Dot = smp.diff(x2, t)
x2DDot = smp.diff(x2Dot, t)

y1 = smp.symbols(f"y_1", cls=smp.Function)
y1 = y1(t)
y1Dot = smp.diff(y1, t)
y1DDot = smp.diff(y1Dot, t)

y2 = smp.symbols(f"y_2", cls=smp.Function)
y2 = y2(t)
y2Dot = smp.diff(y2, t)
y2DDot = smp.diff(y2Dot, t)

t1 = smp.Rational(1, 2) * m1 * x1Dot**2
t2 = smp.Rational(1, 2) * m2 * y1Dot**2
T = t1 + t2

v1 = m1 * g * y1
v2 = m2 * g * y2
V = v1 + v2

L=T-V