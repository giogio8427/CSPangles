import numpy as np
from funcTheta import theta, thetaLumped, energyTransient2,energyTransientLumped
from funcRoots import roots
import plotly.graph_objects as go
def computeQChart(geometry,biotNumbers=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2,0.5, 1,2,5,10,20,50]):
    
    Vol, Ti, Tinf, R, hh, thDiff = 1., 2., 1., 1., 1.0, 1.
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
    
    fig.show(renderer='browser')
    return x,Q_Q0

def computeChartHeisler1 (geometry):
    Vol, Ti, Tinf, R, hh, thDiff = 1., 2., 1., 1.0e-12, 1.0, 1.
    eps=1.e-6
    a=1.5
    z = np.arange(0, 2500, 0.1)         # range to evaluate the zeta, Bi equation
    z[0] = 1e-12 
    maxTheta=1.0
    minTheta=0.0005;
    minFou=0.1
    maxFou=300.
    numPoints=1000
    fig = go.Figure()
# Update layout with four x-axes
    fig.update_layout(
        title='Heisler Chart for a '+geometry,
        # First range (0-3)
        xaxis=dict(
            title='Fourier (0-3)',
            type='linear',
            domain=[0, 0.25],
            range=[0, 3]
        ),
        # Second range (3-10)
        xaxis2=dict(
            title='Fourier (3-10)',
            type='linear',
            domain=[0.25, 0.50],
            range=[3, 10],
            anchor='y'
        ),
        # Third range (10-50)
        xaxis3=dict(
            title='Fourier (10-50)',
            type='linear',
            domain=[0.50, 0.75],
            range=[10, 50],
            anchor='y'
        ),
        # Fourth range (50-250)
        xaxis4=dict(
            title='Fourier (50-250)',
            type='linear',
            domain=[0.75, 1.0],
            range=[50, 250],
            anchor='y'
        ),
        yaxis=dict(
            title='Theta [-]',
            type='log',
            range=[np.log10(0.001),np.log10(1.0)]
        )
    )

    if geometry=='sphere':
        b=2
    elif geometry=='cylinder':
        b=1
    elif geometry=='slab':  
        b=0
       
    invBi=np.concatenate([np.arange(0.1,0.6+eps,0.1), np.arange(0.8,2.0+eps,0.2), np.arange(2.5,3.5+eps,0.5),
           np.arange(4.,10.+eps,1.), np.arange(12.,20.+eps,2.),np.arange(25.,50.+eps,5.), np.arange(60.,100.+eps,10.)])

    biotNumbers=1.0/invBi
    rs=R
    x=np.logspace(np.log10(minFou),np.log10(maxFou),numPoints)
    ii=0
    for biot in biotNumbers:
        rootsVal=roots(z, b, biot) 
        thetaRo, temp = theta(rs, b, rootsVal, x)   # dimensionless temperature profile
        ind=np.logical_and(thetaRo<=maxTheta,thetaRo>=minTheta)
            # Range 0-3
        ind1 = np.logical_and(ind, x <= 3*a)
        fig.add_trace(go.Scatter(
            x=x[ind1], y=thetaRo[ind1],
            mode='lines', name=f'1/Biot: {invBi[ii]:.1f}',hovertemplate='%{x:.10f}, %{y:.10f}',
            xaxis='x'
        ))
        
        # Range 3-10
        ind2 = np.logical_and(ind, np.logical_and(x > 3/a, x <= 10*a))
        fig.add_trace(go.Scatter(
            x=x[ind2], y=thetaRo[ind2],
            mode='lines', 
            name=f'1/Biot: {invBi[ii]:.1f}',hovertemplate='%{x:.10f}, %{y:.10f}',showlegend=False,
            xaxis='x2'
        ))
        
        # Range 10-50
        ind3 = np.logical_and(ind, np.logical_and(x > 10/a, x <= 50*a))
        fig.add_trace(go.Scatter(
            x=x[ind3], y=thetaRo[ind3],
            mode='lines', name=f'1/Biot: {invBi[ii]:.1f}',hovertemplate='%{x:.10f}, %{y:.10f}',
            showlegend=False,
            xaxis='x3'
        ))
        
        # Range 50-250
        ind4 = np.logical_and(ind, np.logical_and(x > 50/a, x <= 250*a))
        fig.add_trace(go.Scatter(
            x=x[ind4], y=thetaRo[ind4],
            mode='lines', name=f'1/Biot: {invBi[ii]:.1f}',hovertemplate='%{x:.10f}, %{y:.10f}',
            showlegend=False,
            xaxis='x4'
        ))
        ii=ii+1

    fig.show(renderer='browser')
    return x,thetaRo

def computeChartHeisler2 (geometry):    
    numPoints=100
    minInvBi=0.1
    maxInvBi=100.0
    x=np.logspace(np.log10(minInvBi),np.log10(maxInvBi),numPoints)

    rr=[0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
    z = np.arange(0, 2500, 0.1)         # range to evaluate the zeta, Bi equation
    fig=go.Figure()
    if geometry=='sphere':
        b=2
    elif geometry=='cylinder':
        b=1
    elif geometry=='slab':  
        b=0
    ratioTheta=np.zeros(numPoints)
   
    
    for r in rr:
        ii=0
        for biot in (1.0/x):
            rootsVal=roots(z, b, biot)
            thetaRo, temp = theta(1.e-12, b, rootsVal, 1.0)
            thetaR,tt=theta(r,b,rootsVal,1.0)
            ratioTheta[ii]=thetaR/thetaRo
            ii=ii+1   
        fig.add_trace(go.Scatter(
            x=x, y=ratioTheta,
            mode='lines', name=f'r/r0: {r:.1f}',hovertemplate='%{x:.10f}, %{y:.10f}',
            xaxis='x'
            ))
    
    fig.update_layout(
        title='Temperature distribution for '+geometry,
        xaxis_title='1/Bi [-]',
        yaxis_title='theta(r)/theta(ro) [-]',
        xaxis_type='log',
        yaxis_type='linear',
        xaxis=dict(range=[0.01, 100.]
    ),
        yaxis=dict(range=[0.0, 1.0])
    )

    fig.show(renderer='browser')
    flag1=0
    flag2=0
    return flag1, flag2
