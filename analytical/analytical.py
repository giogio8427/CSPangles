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
from  numpy import linspace, zeros, arange, pi, array, ones, flipud, concatenate, max, min, repeat
from funcTheta import theta, thetaLumped, energyTransient2,energyTransientLumped
import pltSphere
from plotly.graph_objects import Figure,Scatter
from plotly.subplots import make_subplots
from funcRoots import roots
from tabulate import tabulate

# Parameters from Papadikis 2010a Table 1
#------------------------------------------------------------------------------

def get_input(prompt, default, change=False):
    if change:
        user_input = input(f"{prompt} [{default}]: ")
        try:
            return float(user_input) if user_input else default
        except ValueError:
            return user_input if user_input else default
    else:
        return default

defScreen = """
+-------------------------------------------------------------+
|                       DEFAULT VALUES                        |
+-------------------------------------------------------------+
| 1. Geometry  (sphere, cylinder, slab):         sphere       |
| 1. Density [kg/m3]:                             2458        |
| 2. Particle diameter/thickness [m]:             0.02        |
| 3. Specific heat capacity [J/(kg K)]:            835        |
| 4. Thermal conductivity            [W/(m K)]:   0.75        |
| 5. Uniform initial temperature           [°C]:  500         |
| 6. Surrounding fluid or gas temperature [°C]:   20          |
| 7. Maximum time [s]:                            100         |
| 8. Heat transfer coefficient [W/(m2 K)]:        300         |
| 9. Number of discretization radius:             100         |
| 10. Slider step time [s]:                       1           |
+-------------------------------------------------------------+
"""
inputsDefault = {
    1: ("Geometry", "sphere"),
    2: ("Density [kg/m3]", 2458.),
    3: ("Particle diameter [m]", 0.02),
    4: ("Specific heat capacity [J/(kg K)]", 835.),
    5: ("Thermal conductivity [W/(m K)]", 0.75),
    6: ("Uniform initial temperature [°C]", 500.),
    7: ("Surrounding fluid or gas temperature [°C]", 20.),
    8: ("Maximum time [s]", 100),
    9: ("Heat transfer coefficient [W/(m2 K)]", 300.),
    10: ("Number of discretization radius", 100),
    11: ("Slider step time [s]", 1)
    }

print(defScreen)
inputs=inputsDefault
change_input=100
newSim='y'
while (newSim=='y'):
    change_input=100
    while (change_input!=0):
        geom=inputs[1][1]
        rhow = inputs[2][1]
        d = inputs[3][1]
        cpw = inputs[4][1]
        kw = inputs[5][1]
        Ti = inputs[6][1]
        Tinf = inputs[7][1]
        tmax = inputs[8][1]
        h = inputs[9][1]
        nDiscrR = int(inputs[10][1])
        dt_slider = inputs[11][1]

        for key, value in inputs.items():
            print(f"{key}: {value[0]} [{value[1]}]")

        change_input = input("Enter the number of the input you want to change (or 0 to keep all defaults): ").strip()
        if change_input == "":
            change_input = 999
        else:
            change_input = int(change_input)


        if change_input in inputs:
            prompt, default = inputs[change_input]
            new_value = get_input(prompt, default, True)
            inputs[change_input] = (prompt, new_value)
        elif change_input:
            print("Invalid input number. Please try again.")

    print("Running simulation with the following parameters:")
    print("""
    +-------------------------------------------------------------+
    |                       CURRENT PARAMETERS                    |
    +-------------------------------------------------------------+
        """)
    for key, value in inputs.items():
        print(f"{key}. {value[0]}: {value[1]}")

    # Initial Calculations
    #------------------------------------------------------------------------------

    ro = (d/2)                          # radius of sphere (a.k.a outer radius), m
    rs = ro/ro                          # dimensionless surface radius, (-)
    rc = 1.e-12/ro                      # dimensionless center radius, (-)

    rr=linspace(1.0e-9,ro,nDiscrR)       # discretization radius
    alpha = kw/(rhow*cpw)               # thermal diffusivity biomass, m^2/s
    dt=0.1
    t = arange(0, tmax+dt, dt) # time range for simulation, s
    z = arange(0, 2500, 0.1)         # range to evaluate the zeta, Bi equation
    z[0] = 1e-12                        # prevent divide by zero warning

    nSliderSteps=int(dt_slider/dt)    
    arraySliderSteps=range(0,len(t),nSliderSteps)
    timeSel=0
    Bi = (h*ro)/kw                      # Biot number, (-)
    Fo = (alpha * t) / (ro**2)          # Fourier number, (-)

    L=1.0
    cMap="Hot"                          # color map for plotly
    
    # Temperature Profiles
    #------------------------------------------------------------------------------

    thetaIn=Ti-Tinf

    if geom=="cylinder":
        b = 1   # shape factor where 2 sphere, 1 cylinder, 0 slab
        Vol=pi*ro**2.0*L
        Sup=2.*pi*ro*L   
        intStr="Temperature distribution of Cylinder"
        energyStr="[J/m]"
        thetaRTime=zeros((nDiscrR, nDiscrR, len(t)))
    elif geom=="slab":
        b = 0
        Vol=ro*L*L
        Sup=2.*L*L
        intStr="Temperature distribution of Slab"
        energyStr="[J/m2]"
        thetaRTime=zeros((2*nDiscrR-1, nDiscrR, len(t)))
    else:
        geom="sphere"
        Vol=(4.0/3.0)*pi*ro**3
        Sup=4.0*pi*ro**2
        intStr="Temperature distribution of Sphere"
        energyStr="[J]"
        thetaRTime=zeros((nDiscrR, nDiscrR, len(t)))
        b=2
        if geom!="sphere": print("Invalid geometry- Default to sphere")


    rootsVal=roots(z, b, Bi)

    # surface temperature where ro for outer surface
    thetaRo, temp = theta(rs, b, rootsVal, Bi, Fo)   # dimensionless temperature profile
    thetaRo[0]=1.
    T_o = Tinf + thetaRo*(Ti-Tinf)      # convert theta to temperature in Kelvin, K

    # center temperature where r for center
    thetaR, temp = theta(rc, b, rootsVal, Bi, Fo)    # dimensionless temperature profile
    thetaR[0]=1.
    T_r = Tinf + thetaR*(Ti-Tinf)       # convert theta to temperature in Kelvin, K

    thetaMid, temp =theta(0.5, b, rootsVal, Bi, Fo)    # dimensionless temperature profile
    thetaMid[0]=1.
    T_rMid = Tinf + thetaR*(Ti-Tinf)       # convert theta to temperature in Kelvin, K

    theta_profiles = zeros((len(thetaRo), 3))
    theta_profiles[:,0] = thetaRo  # Surface temperature
    theta_profiles[:,1] = thetaR   # Center temperature
    theta_profiles[:,2] = thetaMid # Middle point temperature



    thetaRadius=zeros((len(rr), len(t)))
    dThetaMin=zeros((len(rr), len(t)))
    for ii in range(0,len(rr),1):
        thetaRadius[ii,:],dThetaMin[ii,:] = theta(rr[ii]/ro, b, rootsVal, Bi, Fo) 

    for ii in range(len(rr)):
        if geom=="sphere":
            thetaRTime[:,ii,:]=thetaRadius[:,:]
        elif geom=="slab":
            thetaRTime[nDiscrR:,ii,:]=thetaRadius[1:,:]
            thetaRTime[0:nDiscrR,ii,:]=flipud(thetaRadius[:,:])
        elif geom=="cylinder":
            thetaRTime[:,ii,:]=thetaRadius[:,:]

    # Lumped Capacitance Solution
    thetaLump, BiLump, FoLump, Lc=thetaLumped(ro, b, h,kw,alpha,t)

    # Energy Calculation
    power=zeros(len(t))
    energy=zeros(len(t))
    energyRatio=zeros(len(t))
    energyLumped=zeros(len(t))  
    energyMax=cpw*rhow*Vol*thetaIn
    energy_enMax_ratio=zeros(len(t))  
    for ii in range(len(t)):
        energy[ii]=energyTransient2(h,alpha,t[ii],rootsVal,ro,Ti,Tinf,geometry=geom)
        energyLumped[ii]=energyTransientLumped(rhow*cpw*Vol,Ti,thetaLump[ii]*thetaIn+Tinf)
        energyRatio[ii]=energy[ii]/energyLumped[ii]
    energy_enMax_ratio=energy/energyMax

    print("""
    +-------------------------------------------------------------+
    |                       SIMULATION PARAMETERS                 |
    +-------------------------------------------------------------+
        """)
    simPar = {
        1: ("n° Biot, [-]", Bi),
        2: ("n° Fourier max., [-]", max(Fo)),
        3: ("n° Biot (Lumped Cap.) [-]", BiLump),
        4: ("n° Fourier max.(Lumped Cap.) [-]", max(FoLump)),
        5: ("Characteristic Length (Lumped Cap.) [m]", Lc)
        }

    for key, value in simPar.items():
        print(f"{key}. {value[0]}: {value[1]}")

    print("""
    +-------------------------------------------------------------+
    |                       OVERALL RESULTS                       |
    +-------------------------------------------------------------+
        """)
    overRes={
        1: ("Energy, [J]", energy[-1]),
        2: ("[Max,Min] Temperature @t=tmax, [°C]", max(thetaRTime[:,:,-1]*(Ti-Tinf)+Tinf), min(thetaRTime[:,:,-1]*(Ti-Tinf)+Tinf)),
        3: ("Energy (Lumped Cap.), [J]", energyLumped[-1]),
        4: ("Temperaure @t=tmax (Lumped Cap.), [°C]", (thetaLump[-1]*(Ti-Tinf)+Tinf)),
        5: ("Energy Ratio (compared to Lumped Cap.)", energyRatio[-1]),
        6: ("Energy/Energy_max [-]", energy_enMax_ratio[-1])
        }

    for key, value in overRes.items():
        if len(value) == 2:
            print(f"{key}. {value[0]}: {value[1]}")
        elif len(value) == 3:
            print(f"{key}. {value[0]}: [{value[1]}, {value[2]}]")

    # Table with results
    print("""
    +-------------------------------------------------------------+
    |                       TIME SERIES RESULTS                   |
    +-------------------------------------------------------------+
        """)

    # Print table with results
    table_data = []
    for i in range(0,len(t),nSliderSteps):
        table_data.append([t[i], Fo[i], thetaR[i], thetaRo[i], T_r[i], T_o[i], energy[i], energy_enMax_ratio[i], max(dThetaMin[:,i])])
        headers = ["time [s]", "Fou [-]", "theta* center [-]", "theta* outer [-]", "Tcenter [°C]", "Touter [°C]", "Energy [J]", "En./En.max [-]", "dThetaMax [-]"]
    print(tabulate(table_data, headers=headers, floatfmt=(".4f", ".4f", ".4f", ".4f", ".4f", ".4f", ".4f", ".4f", ".3e")))


    #region PLOT RESULTS
    # Discretized sphere temperature
    ii=0

    rrArray=zeros(nDiscrR)


    ccMin=min((thetaRTime[:,:,:])*thetaIn+Tinf)
    ccMax=max((thetaRTime[:,:,:])*thetaIn+Tinf)

    ccMinLump=min((thetaLump[:])*thetaIn+Tinf)
    ccMaxLump=max((thetaLump[:])*thetaIn+Tinf)

    vertices = array([
        [-ro, -0.5, -0.5],
        [ro, -0.5, -0.5],
        [ro, 0.5, -0.5],
        [-ro, 0.5, -0.5],
        [-ro, -0.5, 0.5],
        [ro, -0.5, 0.5],
        [ro, 0.5, 0.5],
        [-ro, 0.5, 0.5]
    ])


    if geom=="sphere":
        X,Y,Z=pltSphere.sphereCoord(radius=ro, resolution=nDiscrR)
        YA=Y*0.0
        YB=Y
        X1,Y1,Z1=pltSphere.circleCoord(rr)
        tt=pltSphere.plotCircle(X1,Y1,Z1, clrMatrix=thetaRTime[:,:,timeSel]*thetaIn+Tinf, 
                                ccLim=(ccMin, ccMax), colorMap=cMap)
        tt2=pltSphere.plotSphere(X,Y,Z, clrMatrix=ones(X.shape)*thetaRo[timeSel]*thetaIn+Tinf, ccLim=(ccMin, ccMax),colorMap=cMap)
        tt3=pltSphere.plotSphere(X,Y*0.0,Z, clrMatrix=ones(X.shape)*thetaLump[timeSel]*thetaIn+Tinf, 
                                ccLim=(ccMinLump, ccMaxLump),colorMap=cMap)
        tt4=pltSphere.plotSphere(X,Y,Z, clrMatrix=ones(X.shape)*thetaLump[timeSel]*thetaIn+Tinf, ccLim=(ccMinLump, ccMaxLump),colorMap=cMap)
    elif geom=='slab':
      X,Y,Z=pltSphere.prismCoord(ro, resolution=nDiscrR)
      X1,Y1,Z1=pltSphere.rectCoord(rr)
      YA=Y1*0.0
      tt,Xtemp,Ytemp,Ztemp=pltSphere.plotRect(X1,Y1,Z1, clrMatrix=thetaRTime[:,:,timeSel]*thetaIn+Tinf, 
                                ccLim=(ccMin, ccMax), colorMap=cMap)
      tt2,Xtemp,Ytemp,Ztemp=pltSphere.drawRectPrismFromVertices(vertices, clrMatrix=thetaRo[timeSel]*thetaIn+Tinf,ccLim=(ccMin, ccMax),colorMap=cMap)
      tt3,Xtemp,Ytemp,Ztemp=pltSphere.plotRect(X1,Y1,Z1, clrMatrix=ones(X.shape)*thetaLump[timeSel]*thetaIn+Tinf, 
                                ccLim=(ccMinLump, ccMaxLump),colorMap=cMap)
      tt4,Xtemp,Ytemp,Ztemp=pltSphere.drawRectPrismFromVertices(vertices, clrMatrix=thetaLump[timeSel]*thetaIn+Tinf,ccLim=(ccMinLump, ccMaxLump),colorMap=cMap) 
      YB=Ytemp  
    elif geom=="cylinder":
        X,Y,Z=pltSphere.sphereCoord(radius=ro, resolution=nDiscrR)
        YA=Y*0.0
        YB=Y
        X1,Y1,Z1=pltSphere.circleCoord(rr)
        tt=pltSphere.plotCircle(X1,Y1,Z1, clrMatrix=thetaRTime[:,:,timeSel]*thetaIn+Tinf, 
                                ccLim=(ccMin, ccMax), colorMap=cMap)
        tt2=pltSphere.plotSphere(X,Y,Z, clrMatrix=ones(X.shape)*thetaRo[timeSel]*thetaIn+Tinf, ccLim=(ccMin, ccMax),colorMap=cMap)
        for trace in tt2: trace.update(visible=False)
        tt3=pltSphere.plotSphere(X,Y*0.0,Z, clrMatrix=ones(X.shape)*thetaLump[timeSel]*thetaIn+Tinf, 
                                ccLim=(ccMinLump, ccMaxLump),colorMap=cMap)
        tt4=pltSphere.plotSphere(X,Y,Z, clrMatrix=ones(X.shape)*thetaLump[timeSel]*thetaIn+Tinf, ccLim=(ccMinLump, ccMaxLump),colorMap=cMap)
        for trace in tt4: trace.update(visible=False)

    fig = Figure(data = tt + tt2 + tt3 + tt4)

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
    rrTot=concatenate((flipud(-rr) ,rr))
    thetaRadiusTot=concatenate((flipud(thetaRadius),thetaRadius))

    fig.add_trace(
        Scatter(x=rrTot, y=thetaRadiusTot[:,timeSel]*thetaIn+Tinf, 
                mode='lines', name='T (r,timeSel)'),
        row=1, col=2
    )

    fig.add_trace(
        Scatter(x=array([-ro,ro]), y=repeat(thetaLump[timeSel],2)*thetaIn+Tinf, 
                mode='lines', name='Lumped Cap.'),
        row=1, col=2
    )

    fig.add_trace(
        Scatter(x=array([-ro,ro]), y=repeat(Tinf,2), 
                mode='lines', line=dict(color='black',dash='dash'),
                name='Ambient Temperature'),
        row=1, col=2
    )

    # Add time evolution plot at different radius
    nameLocation=['Surface','Center','Middle']
    for ii in range(3):  
        fig.add_trace(
            Scatter(x=t, y=theta_profiles[:,ii]*thetaIn+Tinf, 
                    mode='lines', name=nameLocation[ii]),
            row=2, col=2
        )

    fig.add_trace(
            Scatter(x=t, y=thetaLump*thetaIn+Tinf, 
                    mode='lines', name='Lumped Cap.'),
            row=2, col=2
        )

    fig.add_trace(
            Scatter(x=array([0, tmax]), y=repeat(Tinf,2), 
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
                    ones(YB.shape)*thetaRo[j]*thetaIn+Tinf,
                    ones(YA.shape)*thetaLump[j]*thetaIn+Tinf,  # For the sphere plot
                    ones(YB.shape)*thetaLump[j]*thetaIn+Tinf  # Keep time evolution plot static  # Keep time evolution plot static
                ],
                # Update temperature profile scatter plot
                "y": [YA, YB, YA, YB,thetaRadiusTot[:,j]*thetaIn+Tinf,repeat(thetaLump[j],2)*thetaIn+Tinf,repeat(Tinf,2),  # For the temperature profile plot
                    thetaRo*thetaIn+Tinf,thetaR*thetaIn+Tinf ,thetaMid*thetaIn+Tinf, thetaLump*thetaIn+Tinf, repeat(Tinf,2)]   # Keep time evolution plot static
            }, {
                "title": f"{intStr} - Time Step: {t[j]} - Energy {energyStr} : {energy[j]:.2f} E/E_Lumped: {energyRatio[j]:.3f}   Biot: {Bi:.2f} Fourier: {Fo[j]:.4f}",
            }]
        })

    # Update layout with slider
    fig.update_layout(
    title=dict(
            text=intStr,
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
            range=[min((ccMin,Tinf)), max((ccMax,Tinf))],  # Or use [ccMin, ccMax] for data-specific range
        ),
        # Keep existing axes settings for time evolution subplot
        yaxis1=dict(title="Temperature [°C]", range=[min((ccMin,Tinf)), max((ccMax,Tinf))]),
        xaxis1=dict(title="Radius [m]", range = [-ro, ro]),
        margin=dict(l=0, r=0, t=100, b=0)  # Minimize margins
    )

    fig.show()
    #endregion
    newSim = (input("Do you want to compute another simulation? (y/n): ").strip())
    
input("Press Enter to exit...")

endSim=0
