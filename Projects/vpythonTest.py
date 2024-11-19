import vpython as vp

g = 9.8
M1 = 0.1
M2 = 0.2
R1 = 0.1
R2 = 0.1

theta1 = 34 * vp.pi / 100
theta1dot = 0
theta2 = 20 * vp.pi / 100
theta2dot = 0

pivot = vp.sphere(pos=vp.vector(0, R1, 0), radius=R1 / 20)
m1 = vp.sphere(
    pos=vp.vector(R1 * vp.sin(theta1), -R1 * vp.cos(theta1), 0), radius=R1 / 10
)
m2 = vp.sphere(
    pos=vp.vector(
        R1 * vp.sin(theta1) + R2 * vp.sin(theta2),
        -R1 * vp.cos(theta1) - R2 * vp.cos(theta2),
        0,
    ),
    radius=R2 / 10,
)


while True:
    vp.rate(60)
