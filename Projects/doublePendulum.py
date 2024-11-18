import sympy as smp

m_1, m_2, L_1, L_2, g, t = smp.symbols('m_{1} m_{2} L g k t')

theta_1 = smp.symbols(r'\theta_{1}', cls=smp.function)
theta_1=theta_1(t)
theta_1dot = smp.diff(theta_1,t)
theta_1ddot = smp.diff(theta_1dot,t)

theta_2 = smp.symbols(r'theta_{2}', cls=smp.function)
theta_2 = theta_2(t)
theta_2dot = smp.diff(theta_2,t)
theta_2ddot = smp.diff(theta_2dot,t)

x_1 = L_1*smp.cos(theta_1)
y_1 = L_1*smp.sin(theta_1)

x_2 = x_1 + L_2*smp.cos(theta_2)
y_2 = y_1 + L_2*smp.sin(theta_2)

T_1 = smp.rational(1,2)*m_1*(smp,diff(x_1.t)**2 + smp.diff(y_1,t)**2)
T_2 = smp.rational(1,2)*m_2*(smp.diff(x_2,t)**2 + smp.diff(y_2,t)**2)

V_1 = m_1*g*y_1
V_2 = m_2*g*y_2

T = T_1 + T_2
V = V_1 + V_2

L = T - V

LE = diff(L,t) + diff(diff(L,thetadot),t)

sols = smp.solve(LE,thetaddot).simplify()



