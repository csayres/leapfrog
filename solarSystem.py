import numpy

import matplotlib.pyplot as plt

from astropy.time import Time
from astropy.coordinates import get_body_barycentric


################
# various constants
G = 6.674e-11 # grav constant
esc_vel = 11200 # m/s escape velocity from earth surface
metersPerAU = 149597870700.0 # meters per astronomical unit (AU == dist from sun to earth)
secondsPerDay = 86400.0 # seconds in a day
secondsPerMinute = 60.0 # seconds per minute
secondsPerHour = secondsPerMinute * 60
# masses of planets in kg according to google
mSun = 1.989e30
mMercury = 3.285e23
mVenus = 4.867e24
mEarth = 5.972e24
mMars = 6.39e23
mJupiter = 1.898e27
mSaturn = 5.683e26
mUranus = 8.681e25
mNeptune = 1.024e26
##################


###################
# generate list of positions for solar system objects
# at each time step
# precomputing these saves time/computation, but it's a bit of a cheat
# because for a true n-body simulation we should be updating
# these each iteration of the leapfrog solver
dt = secondsPerHour # time step used for solver, 1 hour in seconds

# work in astropy's Time class which
# will allow us to easily use calculations
# provided by astropy
launchDate = Time("1990-04-18 00:00")
endDate = Time("2025-12-7 00:00")
# mjd time is decimal days, which is an easy
# time scale to use to build an array of times
# spaced evenly by dt.
launchMJD = launchDate.mjd
endMJD = endDate.mjd
# convert our time step (dt) to mjd
dtMJD = dt / secondsPerDay
# create an array of mjds with time step from launchDate to endDate
mjdArray = numpy.arange(launchMJD, endMJD, dtMJD)
astroTimeArray = Time(mjdArray, format="mjd")
# determine xyz arrays ([x,y,z] x N) for all solar system bodies at each
# point in time in AU
sunArr = get_body_barycentric("sun", astroTimeArray).xyz.value
mercuryArr = get_body_barycentric("mercury", astroTimeArray).xyz.value
venusArr = get_body_barycentric("venus", astroTimeArray).xyz.value
earthArr = get_body_barycentric("earth", astroTimeArray).xyz.value
marsArr = get_body_barycentric("mars", astroTimeArray).xyz.value
jupiterArr = get_body_barycentric("jupiter", astroTimeArray).xyz.value
saturnArr = get_body_barycentric("saturn", astroTimeArray).xyz.value
uranusArr = get_body_barycentric("uranus", astroTimeArray).xyz.value
neptuneArr = get_body_barycentric("neptune", astroTimeArray).xyz.value

# throw away the z values in each of the above arrays
# we only care about xy (2D)
# convert from AU (returned by astropy) to meteres
# transpose arrays so time is along first axis, position along second
# resulting array shape is n x [x,y]
# overwrite original arrays
sunArr = numpy.transpose(sunArr[:2,:]*metersPerAU)
mercuryArr = numpy.transpose(mercuryArr[:2,:]*metersPerAU)
venusArr = numpy.transpose(venusArr[:2,:]*metersPerAU)
earthArr = numpy.transpose(earthArr[:2,:]*metersPerAU)
marsArr = numpy.transpose(marsArr[:2,:]*metersPerAU)
jupiterArr = numpy.transpose(jupiterArr[:2,:]*metersPerAU)
saturnArr = numpy.transpose(saturnArr[:2,:]*metersPerAU)
uranusArr = numpy.transpose(uranusArr[:2,:]*metersPerAU)
neptuneArr = numpy.transpose(neptuneArr[:2,:]*metersPerAU)
#########################################

#### solar system stuff into dicts so we can iterate over elements.
ssDict = {
    "sun": {"mass": mSun, "posArr": sunArr, "color": "gold", "size": 30},
    "mercury": {"mass": mMercury, "posArr": mercuryArr, "color": "tan", "size": 12},
    "venus": {"mass": mVenus, "posArr": venusArr, "color": "sienna", "size": 15},
    "earth": {"mass": mEarth, "posArr": earthArr, "color": "blue", "size": 15},
    "mars": {"mass": mMars, "posArr": marsArr, "color": "red", "size": 14},
    "jupiter": {"mass": mJupiter, "posArr": jupiterArr, "color": "orange", "size":20},
    "saturn": {"mass": mSaturn, "posArr": saturnArr, "color": "sandybrown", "size": 18},
    "uranus": {"mass": mUranus, "posArr": uranusArr, "color": "blue", "size": 17},
    "neptune": {"mass": mNeptune, "posArr": neptuneArr, "color": "navy", "size": 17},
}


########################
# set up initial parameters for launch
# set up inital parameters for launch
mProbe = 1000 # mass of probe, kg

# initial position of probe is at earth xy
probeXY_o = earthArr[0]

# estimate xy velocity of earth from position array
dxyEarth_o = earthArr[1] - earthArr[0]
vxyEarth_o = dxyEarth_o / dt

# initial velocity of probe is the same as earth
probeVXY_o = vxyEarth_o


n = len(astroTimeArray) # number of time steps
probeXY = numpy.zeros((n,2)) # initialize probe's position array
probeVXY = numpy.zeros((n,2)) # initialize probe's velocity array
probeXY[0,:] = probeXY_o
probeVXY[0,:] = probeVXY_o

# generate thrust array, this corresponds to change in velocity
# due to rocket thrusters
thrustArr = numpy.zeros((n,2))
# at first timestep give the probe a delta thrust
# equal to the escape velocity from earth in the +y direction
thrustArr[0,:] = numpy.array([0, esc_vel*1.15])

##############################

############
# begin leapfrog iterations!
for i in range(n-1):
    print(i/float(n))
    # update probe's next position based on current velocities
    nextPos = probeXY[i,:] + dt*probeVXY[i,:]
    probeXY[i+1,:] = nextPos
    # iterate over solar system forces to update next velocity
    updatedVelocity = probeVXY[i,:]
    # iterate over all solar system bodies
    # and compute acceleration due to gravity
    for value in ssDict.itervalues():
        objPos = value["posArr"][i]
        objMass = value["mass"]
        # get distance to object
        rVec = objPos - nextPos
        rMag = numpy.linalg.norm(rVec)
        # add this component to the updatedVelocity variable
        updatedVelocity += dt*(G*objMass/rMag**2)*(rVec/rMag)
    # add in any additional delta velocity from thrust array
    # thrust array is how you perform flight adjustments
    updatedVelocity += thrustArr[i,:]
    probeVXY[i+1,:] = updatedVelocity


# save image to disk for each 200th timestep
lim = numpy.max(numpy.abs(jupiterArr*1.5))
downsample = numpy.arange(0,n,200)
imgNum=0
for i in downsample:
    imgNum += 1
    fig = plt.figure(figsize=(10,10))
    # plot position of each solar system body
    for ssItem in ssDict.itervalues():
        x = ssItem["posArr"][i][0]
        y = ssItem["posArr"][i][1]
        color = ssItem["color"]
        size = ssItem["size"]
        plt.plot(x,y,".", markersize=size, color=color)

    # plot position of probe
    x = probeXY[i][0]
    y = probeXY[i][1]
    color = 'black'
    plt.plot(x,y, marker="1", markersize=10, mew=3, color=color)

    # set the xy limits of the plot to be a little
    # bigger than Jupiter's orbit
    plt.xlim([-lim, lim])
    plt.ylim([-lim, lim])
    # construct a filename to save this image
    # we want it to be fig<zeropaddedInt>.png
    zfillStr = ("%i"%imgNum).zfill(8)
    filename = "fig%s.png"%zfillStr
    plt.savefig(filename)
    plt.close()

# if ffmpeg is installed....
# when images are saved to disk create a movie with ffmpeg like this:
# ffmpeg -r 10 -f image2 -i fig%08d.png out.mp4

