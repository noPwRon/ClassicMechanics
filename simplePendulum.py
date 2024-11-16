import sympy as smp

#  Variables

# x = L*np.cos(theta)
# y = L*np.sin(theta)

# trying to figure out how to get T which is 1/2mv^2
# our v^2 wiil be broken down into x and y components

# L = 1/2*m*v^2


m = smp.Symbol('m') # this one is easy and definintely needed
L = smp.Symbol('L') # also another easy value because well... it's a value, but wwait this is where we want to use smp....
x = smp.Symbol('x')
y = smp.Symbol('y')
g = smp.Symbol('g') #gravity yall

# but wait. we know more physics than that so lets gowith this much nicer choice

# L = T- V

# T = L*\Dot(Theta)
# V = 1/2 m g h
# h = L*cos(theta)
theta = smp.Symbol(r'\theta')



T = 0.5*m*L*thetaDot**2

V = L*g*H

L = T- V

# need to find Dl = P


# s

