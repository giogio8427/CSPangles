import numpy as np
from funcTheta import theta, thetaLumped, energyTransient2,energyTransientLumped
from funcRoots import roots
import plotly.graph_objects as go
def computeQChart(Vol,Ti,Tinf,R,hh, thDiff,geometry,biotNumbers=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2,0.5, 1,2,5,10,20,50]):
    
    if geometry=='sphere':
        b=2
    elif geometry=='cylinder':
        b=1
    elif geometry=='slab':  
        b=0
    
    minX=1e-5
    maxX=1e4
    numPoints=1000
    z = np.arange(0, 2500, 0.1)         # range to evaluate the zeta, Bi equation
    z[0] = 1e-12                        # prevent divide by zero warning

    x=np.logspace(np.log10(minX),np.log10(maxX),numPoints)
    Q_Q0=np.zeros(numPoints)
    ii=0
    rs=R
    fig = go.Figure()
    for biot in biotNumbers:
        k=hh*R/biot
        fourierNumbers=(x/biot**2)
        rho_c=k/thDiff
        t=fourierNumbers*rs**2/thDiff
        rootsVal=roots(z, b, biot)
        Q0=rho_c*Vol*(Tinf-Ti)
        #thetaRo, temp = tc.theta(rs, b, rootsVal, Fo)   # dimensionless temperature profile
        ii=0
        for timeCurr in t:
            Q_Q0[ii]=np.abs(energyTransient2(hh,thDiff,timeCurr,rootsVal,R,Ti,Tinf,geometry)/Q0)
            ii=ii+1
        fig.add_trace(go.Scatter(x=x, y=Q_Q0, mode='lines', name=f'Biot Number: {biot}', hovertemplate='%{x:.10f}, %{y:.10f}'))

        #(hh,thDiff,t,zn,R,Ti,Tinf,geometry):

    fig.update_layout(
        title='Internal energy change for a '+geometry,
        xaxis_title='Bi<sup>2</sup> * Fo [-]',
        yaxis_title='Q/Q0 [-]',
        xaxis_type='log',
        yaxis_type='linear'
    )
    
    fig.show()
    return x,Q_Q0