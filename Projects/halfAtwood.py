import sympy as smp

m1, m2, g, l, s = smp.symbols(f'm_1 m_2 g l s')

x1 = smp.symbols(f'x_1', cls=Function)
x1 = x1(t)
x1Dot = smp.diff(x1,t)
x1DDot = smp.diff(x1Dot,t)

x2 = smp.symbols(f'x_2', cls=Function)
x2 = x2(t)
x2Dot = smp.diff(x2,t)
x2DDot = smp.diff(x2Dot,t)

y1 = smp.symbols(f'y_1', cls=Function)
y1 = y1(t)
y1Dot = smp.diff(y1,t)
y1DDot = smp.diff(y1Dot,t)

y2 = smp.symbols(f'y_2', cls=Function)
y2 = y2(t)
y2Dot = smp.diff(y2,t)
y2DDot = smp.diff(y2Dot,t)

t1 = smp.rational(1,2) * m1 * x1Dot**2
t2 = smp.rational(1,2) * m2 * yDot**2

v1 = m1 * g * y1 
v2 = m2 * g * y2
