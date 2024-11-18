import sympy as smp

m_cart, m_bob, L, g, k,t = smp.symbols('m_{cart} m_{bob} L g k t')

theta = smp.symbols(r'\theta', cls=smp.function)
theta=theta(t)
thetadot = smp.diff(theta,t)
thetaddot = smp.diff(thetadot,t)

x_cart = smp.symbols(r'x_{cart}', cls=smp.function)
x_cart = x_cart(t)
x_cartdot = smp.diff(x_cart,t)
x_cartddot = smp.diff(x_cartdot,t)

x_pend = x_cart + L*smp.cos(theta)
y_pend = L*smp.sin(theta)
 
T_cart = smp.rational(1,2)*m_cart*x_cartdot**2
T_pend = smp.rational(1,2)*m_bob*(diff(x_pend,t)**2 + diff(y_pend,t)**2)
T = T_cart + T_pend

V_cart = smp.rational(1,2)*k*x_cart**2
V_pend = m*g*y_pend
V = V_cart+V_pend

L=T-V
LE = diff(L,theta) - diff(diff(L,thetadot),t)
sols = smp.solve(LE,thetaddot).simplify()



