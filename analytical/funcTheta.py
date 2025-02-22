"""
Function to return the theta (dimensionless temperature) profile at a certain 
dimensionless point (r) in a 1D solid sphere, cylinder, or slab shape.

References: 
1) Recktenwald 2006
2) Bergman, Lavine, Incropera, Dewitt 2011 from Ch. 5, pg.299-304
"""

# Modules and Other Required Functions
#------------------------------------------------------------------------------

from numpy import exp, sqrt, zeros, sin, cos, pi, minimum
from scipy.special import j0, j1  # Import only needed special functions

# First and Second Terms of the Theta Function
#------------------------------------------------------------------------------
    
def funcCn(root, b):
    """
    First term in the theta function
    root = root from the zeta, Bi equation
    b = shape factor where 2 sphere, 1 cylinder, 0 slab
    """
    if b == 2:
        Cn = 4*(sin(root)-root*cos(root)) / (2*root-sin(2*root))
    elif b == 1:
        Cn = (2/root) * (j1(root) / (j0(root)**2 + j1(root)**2))
    elif b == 0:
        Cn = (4*sin(root)) / (2*root + sin(2*root))
    return Cn

def funcDn(r, root, b):
    """
    Second term in the theta function
    root = root from the zeta, Bi equation
    b = shape factor where 2 sphere, 1 cylinder, 0 slab
    """
    if b == 2:
        Dn = (1/(root*r)) * sin(root*r)
    elif b == 1:
        Dn = j0(root * r)
    elif b == 0:
        Dn = cos(root * r)
    return Dn

# Theta Function
#------------------------------------------------------------------------------

def theta(r, b, rts, Bi, Fo):
    """
    Dimensionless temperature for analytical solution of 1D transient heat
    conduction for a solid sphere, cylinder, or slab.
    r = dimensionaless length term to evaluate theta, (-)
    b = shape factor where 2 sphere or 1 cylinder or 0 slab, (-)
    z = range of zeta values to evaluate zeta, Bi equation for positive roots
    Bi = Biot number h*L/k, (-)
    Fo = Fourier number alpha*t/L^2, (-)
    """
    
    #rts = roots(z, b, Bi)   # positive roots of the zeta, Bi equation
    #rts2=roots2(z,b,Bi)
    n = len(rts)            # number of positive roots
    
    # initial dimensionless temperature at first root
    theta = funcCn(rts[0], b)*exp(-rts[0]**2 * Fo)*funcDn(r, rts[0], b)
    dTheta_o_prev=1.e9
    # summation of theta for the remaining roots
    for i in range(1, n):
        dTheta_o = funcCn(rts[i], b)*exp(-rts[i]**2 * Fo)*funcDn(r, rts[i], b)
        theta = theta + dTheta_o
        dTheta_o_min=minimum(dTheta_o, dTheta_o_prev)
        dTheta_o_prev=dTheta_o
    return theta,dTheta_o_min    # theta temperature profile evaluated at r

def thetaLumped(r, b, h,k,alpha,time):
    """
    Dimensionless temperature for analytical solution of 1D transient heat
    conduction for a solid sphere, cylinder, or slab.
    r = dimensionaless length term to evaluate theta, (-)
    b = shape factor where 2 sphere or 1 cylinder or 0 slab, (-)
    z = range of zeta values to evaluate zeta, Bi equation for positive roots
    Bi = Biot number h*L/k, (-)
    Fo = Fourier number alpha*t/L^2, (-)
    """
    if b==2:
        Lc=r/3.0
    elif b==1: 
        Lc=r/2.0
    elif b==0:
        Lc=r/2.0
    Bi=h*Lc/k
    Fo=alpha*time/(Lc**2)
    thetaLumped=exp(-Bi*Fo)
    return thetaLumped, Bi, Fo,Lc    # theta temperature profile evaluated at r


def energyTransient(volThermalCapacity,thDiff,t,zn,R,Ti,Tinf):
    V=4./3.*pi*R**3.
    Fou=thDiff*t/R**2.
    C=volThermalCapacity
    A=4*(sin(zn)-zn*cos(zn))/(2.*zn-sin(2.*zn))*exp(-zn**2.*Fou)
    B=zn/R
    ee= V-sum(4.*pi*A*(-R/(B**2.)*cos(B*R)+1./(B**3.)*sin(B*R)))
    energy=C*ee*(Tinf-Ti)
    return energy

def energyTransientLumped(thermCap,TempIn,TempFin):  
    energyLumped=thermCap*(TempFin-TempIn)
    return energyLumped

def energyTransient2(hh,thDiff,t,zn,R,Ti,Tinf,geometry):
    L=1.0
    B=(-zn**2./R**2.*thDiff)
    if geometry=='sphere':
        S=4.*pi*R**2.
        C=4.*(sin(zn)-zn*cos(zn))/(2.*zn-sin(2.*zn))
        A=C*sin(zn)/zn
    elif geometry=='slab':
        S=L*L*2.
        C=4.*(sin(zn))/(2.*zn+sin(2.*zn))
        A=C*cos(zn)
    elif geometry=='cylinder':
        S=2.*pi*R*L
        C=2./zn*(j1(zn)/(j0(zn)**2+j1(zn)**2))
        A=C*j0(zn)
   
    energy= S*hh*(Ti-Tinf)*sum(A/B*(1.-exp(B*t)))
    return energy