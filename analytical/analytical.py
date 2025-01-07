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
from funcTheta import theta, thetaLumped, energyTransient, energyTransient2,energyTransientLumped
import pltSphere
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from funcRoots import roots
py.close('all')

# Parameters from Papadikis 2010a Table 1
#------------------------------------------------------------------------------

rhow = 2458      # density of biomass, 700 kg/m3
d = 0.02    # particle diameter for 350 um size, m
cpw = 835      # specific heat capacity biomass, J/(kg K)
kw = 0.75      # thermal conductivity biomass, W/(m K)
Ti = 500        # uniform initial temp of sphere, °C
Tinf = 20      # surrounding fluid or gas temp, °C
tmax = 100       # max time, s
h = 300         # heat transfer coefficent, W/(m2 K)

# Initial Calculations
#------------------------------------------------------------------------------

ro = (d/2)                          # radius of sphere (a.k.a outer radius), m
rs = ro/ro                          # dimensionless surface radius, (-)
rc = 1e-12/ro                       # dimensionless center radius, (-)
nDiscrR=100                          # number of discretization radius
rr=np.linspace(1.0e-9,ro,nDiscrR)       # discretization radius
alpha = kw/(rhow*cpw)               # thermal diffusivity biomass, m^2/s
dt=0.1
t = np.arange(0, tmax+dt, dt) # time range for simulation, s
z = np.arange(0, 2500, 0.1)         # range to evaluate the zeta, Bi equation
z[0] = 1e-12                        # prevent divide by zero warning
dt_slider=1
cMap="Hot"

nSliderSteps=int(dt_slider/dt)    
arraySliderSteps=range(0,len(t),nSliderSteps)
timeSel=0
Bi = (h*ro)/kw                      # Biot number, (-)
Fo = (alpha * t) / (ro**2)          # Fourier number, (-)

Vol=(4.0/3.0)*np.pi*ro**3
Sup=4.0*np.pi*ro**2
print("Biot Number: ", Bi)
# Sphere Temperature Profiles
#------------------------------------------------------------------------------

thetaIn=Ti-Tinf

b = 2   # shape factor where 2 sphere, 1 cylinder, 0 slab

rootsVal=roots(z, b, Bi)


# surface temperature where ro for outer surface
thetaRo = theta(rs, b, rootsVal, Bi, Fo)   # dimensionless temperature profile
T_o = Tinf + thetaRo*(Ti-Tinf)      # convert theta to temperature in Kelvin, K

# center temperature where r for center
thetaR = theta(rc, b, rootsVal, Bi, Fo)    # dimensionless temperature profile
T_r = Tinf + thetaR*(Ti-Tinf)       # convert theta to temperature in Kelvin, K

thetaMid =theta(0.5, b, rootsVal, Bi, Fo)    # dimensionless temperature profile
T_rMid = Tinf + thetaR*(Ti-Tinf)       # convert theta to temperature in Kelvin, K
theta_profiles = np.zeros((len(thetaRo), 3))
theta_profiles[:,0] = thetaRo  # Surface temperature
theta_profiles[:,1] = thetaR   # Center temperature
theta_profiles[:,2] = thetaMid # Middle point temperature

# Discretized sphere temperature
ii=0

X,Y,Z=pltSphere.sphereCoord(radius=ro, resolution=nDiscrR)
X1,Y1,Z1=pltSphere.circleCoord(rr)

rr2=(X**2.+Z**2.)**0.5
rrArray=np.zeros(nDiscrR)
thetaRTime=np.zeros((nDiscrR, nDiscrR, len(t)))
 

thetaRadius=np.zeros((len(rr), len(t)))
for ii in range(len(rr)):
     thetaRadius[ii,:] = theta(rr[ii]/ro, b, rootsVal, Bi, Fo) 

for ii in range(len(rr)):
    thetaRTime[:,ii,:]=thetaRadius[:,:]

thetaLump, BiLump, FoLump=thetaLumped(ro, b, h,kw,alpha,t)
power=np.zeros(len(t))
energy=np.zeros(len(t))
energyRatio=np.zeros(len(t))
energyLumped=np.zeros(len(t))    
for ii in range(len(t)):
    #energy[ii]=energyTransient(rhow*cpw,alpha,t[ii],rootsVal,ro,Ti,Tinf)
    energy[ii]=energyTransient2(h,alpha,t[ii],rootsVal,ro,Ti,Tinf)
    energyLumped[ii]=energyTransientLumped(rhow*cpw*Vol,Ti,thetaLump[ii]*thetaIn+Tinf)
    energyRatio[ii]=energy[ii]/energyLumped[ii]

ccMin=np.min((thetaRTime[:,:,:])*thetaIn+Tinf)
ccMax=np.max((thetaRTime[:,:,:])*thetaIn+Tinf)

ccMinLump=np.min((thetaLump[:])*thetaIn+Tinf)
ccMaxLump=np.max((thetaLump[:])*thetaIn+Tinf)


tt=pltSphere.plotCircle(X1,Y1,Z1, clrMatrix=thetaRTime[:,:,timeSel]*thetaIn+Tinf, 
                         ccLim=(ccMin, ccMax), colorMap=cMap)

tt2=pltSphere.plotSphere(X,Y,Z, clrMatrix=np.ones(X.shape)*thetaRo[timeSel]*thetaIn+Tinf, ccLim=(ccMin, ccMax),colorMap=cMap)

tt3=pltSphere.plotSphere(X,Y*0.0,Z, clrMatrix=np.ones(X.shape)*thetaLump[timeSel]*thetaIn+Tinf, 
                         ccLim=(ccMinLump, ccMaxLump),colorMap=cMap)
tt4=pltSphere.plotSphere(X,Y,Z, clrMatrix=np.ones(X.shape)*thetaLump[timeSel]*thetaIn+Tinf, ccLim=(ccMinLump, ccMaxLump),colorMap=cMap)

fig = go.Figure(data = tt + tt2 + tt3 + tt4)
#fig.add_trace(tt+tt2)

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('', '', '', ''),
    specs=[[{'type': 'surface'}, {'type': 'scatter'}],
           [{'type': 'surface'}, {'type': 'scatter'}]],
           horizontal_spacing=0.02, # in range 0 to 1/(cols-1)
           vertical_spacing=0.1,
)

# Add first sphere surface to subplot 1
for trace in tt:
    trace.update(colorbar=dict(
        x=0.35,  # Position colorbar
        y=0.8,
        len=0.4,  # Length of colorbar
        thickness=10,  # Thickness of colorbar
        title="Temperature [°C]",
        titleside="right"
    ))
    fig.add_trace(trace, row=1, col=1)
# Add second sphere surface to subplot 3
for trace in tt2:
    trace.update(showscale=False)
    fig.add_trace(trace, row=1, col=1)

for trace in tt3:
    trace.update(showscale=False)
    fig.add_trace(trace, row=2, col=1)
# Add second sphere surface to subplot 3
for trace in tt4:
    trace.update(showscale=False)
    fig.add_trace(trace, row=2, col=1)

# Add temperature profile plot

rrTot=np.concatenate((np.flipud(-rr) ,rr))
thetaRadiusTot=np.concatenate((np.flipud(thetaRadius),thetaRadius))

fig.add_trace(
    go.Scatter(x=rrTot, y=thetaRadiusTot[:,timeSel]*thetaIn+Tinf, 
               mode='lines', name='T (r,timeSel)'),
    row=1, col=2
)

fig.add_trace(
    go.Scatter(x=np.array([-ro,ro]), y=np.repeat(thetaLump[timeSel],2)*thetaIn+Tinf, 
               mode='lines', name='Lumped Cap.'),
    row=1, col=2
)

fig.add_trace(
    go.Scatter(x=np.array([-ro,ro]), y=np.repeat(Tinf,2), 
               mode='lines', line=dict(color='black',dash='dash'),
               name='Ambient Temperature'),
    row=1, col=2
)

# Add time evolution plot at different radius
nameLocation=['Surface','Center','Middle']
for ii in range(3):  
    fig.add_trace(
        go.Scatter(x=t, y=theta_profiles[:,ii]*thetaIn+Tinf, 
                mode='lines', name=nameLocation[ii]),
        row=2, col=2
    )

fig.add_trace(
        go.Scatter(x=t, y=thetaLump*thetaIn+Tinf, 
                mode='lines', name='Lumped Cap.'),
        row=2, col=2
    )

fig.add_trace(
        go.Scatter(x=np.array([0, tmax]), y=np.repeat(Tinf,2), 
                mode='lines', line=dict(color='black',dash='dash'),
                name='Ambient Temperature', showlegend=False),
        row=2, col=2
    )

#slider_steps=pltSphere.createSliderSteps(t, thetaRTime, thetaRo,thetaRadius)


# Create slider steps for all subplots
slider_steps = []
for j in arraySliderSteps:
    slider_steps.append({
        "method": "update",
        "label": str(t[j]) + " sec.",
        "args": [{
            # Update surface colors for both spheres
            "surfacecolor": [
                thetaRTime[:,:,j]*thetaIn+Tinf,  # For the sphere plot
                np.ones(X.shape)*thetaRo[j]*thetaIn+Tinf,
                np.ones(X.shape)*thetaLump[j]*thetaIn+Tinf,  # For the sphere plot
                np.ones(X.shape)*thetaLump[j]*thetaIn+Tinf  # Keep time evolution plot static  # Keep time evolution plot static
            ],
            # Update temperature profile scatter plot
            "y": [Y*0.0, Y, Y*0.0, Y,thetaRadiusTot[:,j]*thetaIn+Tinf,np.repeat(thetaLump[j],2)*thetaIn+Tinf,np.repeat(Tinf,2),  # For the temperature profile plot
                 thetaRo*thetaIn+Tinf,thetaR*thetaIn+Tinf ,thetaMid*thetaIn+Tinf, thetaLump*thetaIn+Tinf, np.repeat(Tinf,2)]   # Keep time evolution plot static
        }, {
            "title": f"Temperature Distribution - Time Step: {t[j]} - Energy [J] : {energy[j]:.2f} E/E_Lumped: {energyRatio[j]:.3f}"
        }]
    })

# Update layout with slider
fig.update_layout(
 title=dict(
        text="Temperature Distribution in Sphere",
        x=0.5,
        y=0.95,
        xanchor='center',
        yanchor='top',
        font=dict(size=14),
        pad=dict(t=0, b=0)  # Minimize padding
    ),
    sliders=[{
        'currentvalue': {"prefix": "Time: "},
        'steps': slider_steps,
        "active": timeSel,
        "pad": {"t": 25, "b": 5},
    }],
    # Update axes for temperature profile subplot (1,2)
    xaxis2=dict(
        title="Time [sec.]",
        range=[t[0], t[-1]],
    ),
    yaxis2=dict(
        title="Temperature [°C]",
        range=[np.min((ccMin,Tinf)), np.max((ccMax,Tinf))],  # Or use [ccMin, ccMax] for data-specific range
    ),
    # Keep existing axes settings for time evolution subplot
    yaxis1=dict(title="Temperature [°C]", range=[np.min((ccMin,Tinf)), np.max((ccMax,Tinf))]),
    xaxis1=dict(title="Radius [m]", range = [-ro, ro]),
    margin=dict(l=0, r=0, t=70, b=0)  # Minimize margins
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

