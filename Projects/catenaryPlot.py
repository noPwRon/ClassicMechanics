import vpython as vp

#  Defining length, mass, and spring constant
L = 0.2
M=0.1
K=1

# defining the number of nodes
N=20

# breaking the spring constant into individual nodes
k = K*(N-1)
balls = []
leftend = vp.vector(0,0,0)
ds = vector(1,0,0)*L/(N-1)
s = L/(N-1)
R = L/(3*N)


for i in range(N):
    # Defining a set of balls that are a sphere which start at posisition leftend, of radius R and have a mass of M (total mass of the string) divided by the amount of units that make up the string
    balls=balls+[vp.sphere(pos=leftend+i*ds, radius=R, m=M/N, p=vp.vector(0,0,0), F = vp.vector(0,0,0))]
    
springs=[]

for i in range(N-1):
    # defining the connection between the balls
    springs=springs+[vp.cylinder(pos=leftend=i*ds, axis = ds, radius = R/5)]
    
t=0
dt=0.001

while t<10:
    rate(1000)
    r1l=-springs[0].axis
    # balls[0].F=-k*(mag(r1l)-s)*norm(r1l)+vector(-0.01,.002,0)
    balls[0].F=vector(0,0,0)
    for i in range(1,N-1):
        balls[i].F=balls[i].m*g-C*balls[i].p-k*(mag(springs[i-1].axis)-s)*norm(springs[i-1].axis)+k*(mag(springs[i].axis)-s)*norm(springs[i].axis)
        
    balls[-1].F=vector(0,0,0)
    for ball in balls:
        ball.p = ball.p+ball.F*dt
        ball.pos=ball.pos+ball.p*dt/ball.m
    for i in range(1,N):
        springs[i-1].axis=balls[i]pos-balls[i-1].pos
        springs[i-1].pos=balls[i-1].pos
        
    t=t+dt
    
Lnew=0

for i in range(1,N):
    Dlnew=mag(balls[i].pos-ballls[i-1].pos)
    Lnew=Lnew+dt.new
    
print("New Length = ", Lnew," m")



    
    
    