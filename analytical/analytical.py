"""
Plot the analytical solutions of 1D transient heat conduction in a solid
sphere, cylinder, and slab with convection at the surface and no heat
generation within the solid.

Requirements:
Python 3, NumPy, SciPy, and Matplotlib

Functions:
funcTheta.py returns theta (dimensionless temp) for sphere, cylinder, or slab
funcRoots.py returns the positive roots of the zeta, Bi equation
funcZeta.py functions for zeta, Bi equation for sphere, cylinder, and slab
funcTheta <- funcRoots <- funcZeta

References: 
1) Recktenwald 2006
2) Bergman, Lavine, Incropera, Dewitt 2011 from Ch. 5, pg.299-304
3) Papadikis 2010a
"""

# Modules
#------------------------------------------------------------------------------
import plotly
import numpy as np
import numpy.matlib
import matplotlib.pyplot as py
from funcTheta import theta
import pltSphere
import plotly.graph_objects as go
from plotly.subplots import make_subplots
py.close('all')

# Parameters from Papadikis 2010a Table 1
#------------------------------------------------------------------------------

rhow = 700      # density of biomass, 700 kg/m3
d = 0.07e-2    # particle diameter for 350 um size, m
cpw = 3500      # specific heat capacity biomass, J/(kg K)
kw = 0.105      # thermal conductivity biomass, W/(m K)
Ti = 300        # uniform initial temp of sphere, K
Tinf = 773      # surrounding fluid or gas temp, K
tmax = 1.0      # max time, s
h = 375         # heat transfer coefficent, W/(m2 K)

# Initial Calculations
#------------------------------------------------------------------------------

ro = (d/2)                          # radius of sphere (a.k.a outer radius), m
rs = ro/ro                          # dimensionless surface radius, (-)
rc = 1e-12/ro                       # dimensionless center radius, (-)
nDiscrR=25                          # number of discretization radius
rr=np.linspace(1.0e-9,ro,nDiscrR)       # discretization radius
alpha = kw/(rhow*cpw)               # thermal diffusivity biomass, m^2/s
t = np.arange(0, tmax+0.002, 0.002) # time range for simulation, s
z = np.arange(0, 1250, 0.1)         # range to evaluate the zeta, Bi equation
z[0] = 1e-12                        # prevent divide by zero warning

Bi = (h*ro)/kw                      # Biot number, (-)
Fo = (alpha * t) / (ro**2)          # Fourier number, (-)

# Sphere Temperature Profiles
#------------------------------------------------------------------------------

thetaIn=Ti-Tinf

b = 2   # shape factor where 2 sphere, 1 cylinder, 0 slab

# surface temperature where ro for outer surface
thetaRo = theta(rs, b, z, Bi, Fo)   # dimensionless temperature profile
T_o = Tinf + thetaRo*(Ti-Tinf)      # convert theta to temperature in Kelvin, K

# center temperature where r for center
thetaR = theta(rc, b, z, Bi, Fo)    # dimensionless temperature profile
T_r = Tinf + thetaR*(Ti-Tinf)       # convert theta to temperature in Kelvin, K

# Discretized sphere temperature
ii=0

X,Y,Z=pltSphere.sphereCoord(radius=ro, resolution=nDiscrR)
rr2=(X**2+Z**2)**0.5
rrArray=np.zeros(nDiscrR)
thetaRTime=np.zeros((nDiscrR, nDiscrR, len(t)))
for ii in range(rr2.shape[1]):
    for jj in range(rr2.shape[0]):
        if rr2[jj,ii]==0:
            rr2[jj,ii]=1e-12   
        currRadius=rr2[jj,ii]        
        # surface temperature where ro for outer surface
        thetaRTime[jj,ii,:] = theta(currRadius/ro, b, z, Bi, Fo)   # dimensionless temperature profile    

val, ind=np.unique(currRadius, return_index=True)
thetaRadius=np.zeros((len(rr), len(t)))
for ii in range(len(rr)):
     thetaRadius[ii,:] = theta(rr[ii]/ro, b, z, Bi, Fo) 



timeSel=250

ccMin=np.min((thetaRTime[:,:,:])*thetaIn+Tinf)
ccMax=np.max((thetaRTime[:,:,:])*thetaIn+Tinf)

tt=pltSphere.plotSphere(X,Y*0.0,Z, clrMatrix=thetaRTime[:,:,timeSel]*thetaIn+Tinf, 
                         ccLim=(ccMin, ccMax))
tt2=pltSphere.plotSphere(X,Y,Z, clrMatrix=np.ones(X.shape)*thetaRo[timeSel]*thetaIn+Tinf, ccLim=(ccMin, ccMax))
fig = go.Figure(data = tt + tt2)
#fig.add_trace(tt+tt2)
 
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Sphere Cross Section', 'Temperature Profile', 'Additional View', 'Time Evolution'),
    specs=[[{'type': 'surface'}, {'type': 'scatter'}],
           [{'type': 'surface'}, {'type': 'scatter'}]]
)

# Add first sphere surface to subplot 1
for trace in tt:
    fig.add_trace(trace, row=1, col=1)

# Add second sphere surface to subplot 3
for trace in tt2:
    fig.add_trace(trace, row=1, col=1)

# Add temperature profile plot
fig.add_trace(
    go.Scatter(x=rr, y=thetaRadius[:,timeSel]*thetaIn+Tinf, 
               mode='lines', name='Temperature Profile'),
    row=1, col=2
)

# Add time evolution plot
fig.add_trace(
    go.Scatter(x=t, y=thetaRo*thetaIn+Tinf, 
               mode='lines', name='Surface Temperature'),
    row=2, col=2
)


#slider_steps=pltSphere.createSliderSteps(t, thetaRTime, thetaRo,thetaRadius)


# Create slider steps for all subplots
slider_steps = []
for j in range(len(t)):
    slider_steps.append({
        "method": "update",
        "label": str(j),
        "args": [{
            # Update surface colors for both spheres
            "surfacecolor": [
                thetaRTime[:,:,j]*thetaIn+Tinf,  # For the sphere plot
                np.ones(X.shape)*thetaRo[j]*thetaIn+Tinf  # Keep time evolution plot static
            ],
            # Update temperature profile scatter plot
            "y": [Y*0.0, Y, thetaRadius[:,j]*thetaIn+Tinf,  # For the temperature profile plot
                 thetaRo*thetaIn+Tinf]  # Keep time evolution plot static
        }, {
            "title": f"Temperature Distribution - Time Step: {j}"
        }]
    })

# Update layout with slider
fig.update_layout(
    height=800,
    title_text="Temperature Distribution in Sphere",
    sliders=[{
        'currentvalue': {"prefix": "Time Step: "},
        'steps': slider_steps,
        "active": timeSel,
        "pad": {"t": 50}
    }],
    # Update axes for temperature profile subplot (1,2)
    xaxis2=dict(
        title="Time",
        range=[t[0], t[-1]],
        tickmode='linear',
        dtick=0.2
    ),
    yaxis2=dict(
        title="Temperature [°C]",
        range=[ccMin, ccMax],  # Or use [ccMin, ccMax] for data-specific range
    ),
    
    # Keep existing axes settings for time evolution subplot
    yaxis1=dict(title="Temperature [°C]", range=[ccMin, ccMax])
)


fig.show()

# Cylinder Temperature Profiles
#------------------------------------------------------------------------------

b = 1   # shape factor where 2 sphere, 1 cylinder, 0 slab

# surface temperature where ro for outer surface
thetaRo = theta(rs, b, z, Bi, Fo)   # dimensionless temperature profile
To_cyl = Tinf + thetaRo*(Ti-Tinf)   # convert theta to temperature in Kelvin, K

# center temperature where r for center
thetaR = theta(rc, b, z, Bi, Fo)    # dimensionless temperature profile
Tr_cyl = Tinf + thetaR*(Ti-Tinf)    # convert theta to temperature in Kelvin, K

# Slab Temperature Profile
#------------------------------------------------------------------------------

b = 0   # shape factor where 2 sphere, 1 cylinder, 0 slab

# surface temperature where ro for outer surface
thetaRo = theta(rs, b, z, Bi, Fo)   # dimensionless temperature profile
To_slab = Tinf + thetaRo*(Ti-Tinf)  # convert theta to temperature in Kelvin, K

# center temperature where r for center
thetaR = theta(rc, b, z, Bi, Fo)    # dimensionless temperature profile
Tr_slab = Tinf + thetaR*(Ti-Tinf)   # convert theta to temperature in Kelvin, K

# Plot Results
#------------------------------------------------------------------------------

# configure y-axis based on cooling or heating simulation
if Ti > Tinf:
    # for a cooling process where Ti=773K and Tinf=300K
    Th = Ti
    ylim =[Ti+20, Tinf-20]
else:
    # for a heating process where Ti=300K and Tinf=773K
    Th = Tinf
    ylimRange = [Ti-20, Tinf+20]
    
py.figure(1)
py.plot(t, T_o, '-r', lw=2, label='surface')
py.plot(t, T_r, '--r', lw=2, label='center')
py.title('Sphere')
py.ylabel('Temperature (K)')
py.xlabel('Time (s)')
py.ylim(ylimRange)
py.xlim([0, tmax])
py.axhline(y=Th, color='k', linestyle='--', label=r'T$_\infty$')
py.rcParams['xtick.major.pad'] = 6
py.rcParams['ytick.major.pad'] = 6
py.legend(loc='best', numpoints=1)
py.grid()
py.show()

py.figure(2)
py.plot(t, To_cyl, '-b', lw=2, label='surface')
py.plot(t, Tr_cyl, '--b', lw=2, label='center')
py.title('Cylinder')
py.ylabel('Temperature (K)')
py.xlabel('Time (s)')
py.ylim(ylimRange)
py.xlim([0, tmax])
py.axhline(y=Th, color='k', linestyle='--', label=r'T$_\infty$')
py.rcParams['xtick.major.pad'] = 6
py.rcParams['ytick.major.pad'] = 6
py.legend(loc='best', numpoints=1)
py.grid()
py.show()

py.figure(3)
py.plot(t, To_slab, '-g', lw=2, label='surface')
py.plot(t, Tr_slab, '--g', lw=2, label='center')
py.title('Slab')
py.ylabel('Temperature (K)')
py.xlabel('Time (s)')
py.ylim(ylimRange)
py.xlim([0, tmax])
py.axhline(y=Th, color='k', linestyle='--', label=r'T$_\infty$')
py.rcParams['xtick.major.pad'] = 6
py.rcParams['ytick.major.pad'] = 6
py.legend(loc='best', numpoints=1)
py.grid()
py.show()

