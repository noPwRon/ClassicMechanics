import numpy as np
import matplotlib.pyplot as plt
# main imports that we will need to make the pendulumn work

# defining typical variables: length, gravity, time step. Note that mass has nothing to do with the equation of motion for the pendulumn.

L = 0.3
g = 9.8
dt = 0.1

# intitial conditions need to be initialized.
t = 0
theta = 45 * np.pi / 180  # this is equivulent to 45 degrees because radians
thetadot = 0  # initial circular velocity is zero (this means the ball has been held at 45 degrees and then released with no additional energy)
thetap = []
tp = []

while t < 2:
    thetaddot = -g * np.sin(theta) / L
    thetadot = thetadot + thetaddot * dt
    theta = theta + thetadot * dt
    t = t + dt
    thetap = thetap + [theta]
    tp = tp + [t]


plt.title("pendlumn")
plt.xlabel("t")
plt.ylabel("theta")
plt.plot(tp, thetap)
plt.show()
