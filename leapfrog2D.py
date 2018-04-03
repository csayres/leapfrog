import numpy

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


############## some constants ############

G = 6.674e-11 # grav constant
earth_mass = 5.972e24 # kgs
sat_mass = 100 # kgs
earth_radius = 6371000
m1 = earth_mass
m2 = sat_mass

##########################################


def twoBody(t, dt, pos1_o, pos2_o, v1_o, v2_o, m1, m2):
    """Leapfrog solver for 2 body problem

    t: total duration of time in seconds
    dt: time step in seconds
    pos1_o: [x,y] initial position of object 1 in meters
    pos2_o: [x,y] initial position of object 2 in meters
    v1_o: [vx,vy] initial velocity of object 1 in meters/sec
    v2_o: [vx,vy] initial velocity of object 2 in meters/sec
    m1: mass of object 1 in kg
    m2: mass of object 2 in kg

    returns:
    tVec: vector of timesteps in seconds (of length N, t/dt)
    pos1: N x [x,y], array of positions of object 1 at each timestep
    pos2: N x [x,y], array of positions of object 2 at each timestep
    v1: N x [vx,vy], array of positions of object 1 at each timestep
    v2: N x [vx,vy], array of positions of object 2 at each timestep
    """
    # initialize outputs to zero
    # they will be populated in the
    # leapfrog loop.
    tVec = numpy.arange(0,t,dt)
    n = len(tVec)
    pos1 = numpy.zeros((n,2))
    pos2 = numpy.zeros((n,2))
    v1 = numpy.zeros((n,2))
    v2 = numpy.zeros((n,2))

    # initial conditions
    pos1[0,:] = pos1_o
    pos2[0,:] = pos2_o
    v1[0,:] = v1_o
    v2[0,:] = v2_o

    for i in range(n-1):
        pos1[i+1,:] = pos1[i,:] + dt*v1[i,:]
        pos2[i+1,:] = pos2[i,:] + dt*v2[i,:]
        rVec = pos2[i+1,:]-pos1[i+1,:]
        r = numpy.linalg.norm(rVec)
        v1[i+1,:] = v1[i,:] + dt*(G*m2/r**2)*(rVec/r)
        v2[i+1,:] = v2[i,:] + dt*(G*m1/r**2)*(-1*rVec/r)

    return tVec, pos1,pos2,v1,v2


def stableOrbit():
    # return initial conditions for a stable orbit about earth
    # you can plug these into the leapfrog solver
    sat_r = 6780000 # radius satellite is at in m
    sat_p = 2*numpy.pi*numpy.sqrt(sat_r**3/(G*earth_mass)) # satellite period
    sat_v = numpy.sqrt(G*earth_mass/sat_r) # orbital velocity at sat_r v = sqrt(G*earth_mass/sat_r)
    t = sat_p # one orbit
    dt = t/10000 # 1000 timesteps
    v1 = numpy.array([0,0]) # xy
    v2 = numpy.array([0, sat_v])
    pos1 = numpy.array([0,0])
    pos2 = numpy.array([sat_r,0])
    return t, dt, pos1, pos2, v1, v2

def launch():
    # return initialconditions for a launched satellite
    # you can plug these into leapfrog solver

    esc_vel = 11200 # m/s escape velocity from earth surface
    # time steps
    sec_per_day = 60*60*60*24
    t = sec_per_day

    t = 60*60*24*200 # 200 days
    dt = 60*2 # seconds
    v1 = numpy.array([0,0]) # xy earth at rest
    v2 = numpy.array([0, esc_vel*.99])
    pos1 = numpy.array([0,0])
    pos2 = numpy.array([earth_radius,0])
    return t, dt, pos1, pos2, v1, v2


# t, dt, pos1, pos2, v1, v2 = stableOrbit()
t, dt, pos1, pos2, v1, v2 = launch()

# solve for trajectories
tVec, pos1,pos2,v1,v2 = twoBody(t,dt,pos1,pos2,v1,v2,m1,m2)

# scale pos by earth radius
pos2 = pos2/float(earth_radius)

# set up and show matplotlib animation of trajectory
lim = numpy.max(numpy.abs(pos2))*1.3

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-lim, lim), ylim=(-lim, lim))
ax.set_aspect('equal')
ax.set_xlabel("earth radius")
ax.set_ylabel("earth radius")

ax.plot([0], [0], '.k', markersize=10)
line, = ax.plot([], [], '.b', markersize=10)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text

def animate(i):
    thisx = pos2[i,0]
    thisy = pos2[i,1]
    # thisx = [0, x1[i], x2[i]]
    # thisy = [0, y1[i], y2[i]]

    line.set_data(thisx, thisy)
    time_text.set_text(time_template % (i*dt))
    return line, time_text

ani = FuncAnimation(fig, animate, numpy.arange(1, len(pos2)),
                              interval=1, blit=True, init_func=init)

plt.show()

