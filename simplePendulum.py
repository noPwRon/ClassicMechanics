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
