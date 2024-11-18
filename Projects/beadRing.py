import sympy as smp

omega, R, t, g, m = smp.symbols('\omega R t g')
theta = smp.symbols(r'\theta',cls=smp.function)
theta = theta(t)
thetadot = diff(theta,t)
thetaddot = diff(thetadot,t)


z = -R*smp.cos(theta)
x = R*smp.sin(theta)*smp.sin(omega*t)
y = R*smp.sin(theta)*smp.sin(omega*t)

T = smp.rational(1,2)*m*(diff(x,t)**2+diff(y,t)**2)
V = m*g*z
