"""
Function that returns the positive roots of the zeta, Bi equation of the
analytical solution for 1D transient heat conduction of a sphere, cylinder,
or slab shape.

References: 
1) Recktenwald 2006
2) Bergman, Lavine, Incropera, Dewitt 2011 from Ch. 5, pg.299-304
"""

# Modules and Other Required Functions
#------------------------------------------------------------------------------

import numpy as np
import scipy.optimize as op
from funcZeta import funcZetaSph, funcZetaCyl, funcZetaSlab

# Roots Function
#------------------------------------------------------------------------------

def roots(z, b, Bi):
    """
    Returns a list of positive roots from zeta, Bi equation used for the
    analytical solution of 1D transient heat conduction for a solid sphere,
    cylinder, or slab shape.
    z = range of zeta values to test for positive roots
    b = shape factor where 2 sphere, 1 cylinder, 0 slab
    Bi = Biot number h*L/k, (-)
    """
    
    if b==2:
        fz = funcZetaSph(z, Bi)     # evaluate sphere function at z-values
        func = funcZetaSph          # declare sphere function as roots function
    elif b==1:
        fz = funcZetaCyl(z, Bi)     # evaluate cylinder function at z-values
        func = funcZetaCyl          # declare cylinder function as roots function
    elif b==0:
        fz = funcZetaSlab(z, Bi)    # evaluate slab function at the z-values
        func = funcZetaSlab         # declare slab function as roots function
    
    # sign of values in fz as -/+ 1
    sign = np.sign(fz)
    
    # calculate difference between fz values where a non-zero value indicates
    # the location of a possible root of the function
    diff = np.diff(sign)
    
    # b as the shape factor where 2 sphere, 1 cylinder, 0 slab
    # use the np.where function to find array index of value change
    # note that np.where returns a tuple
    if b == 2 or b==0:
        # location of sphere or slab roots
        # find index of only positive value change thus ignoring singularities
        where = np.where(diff>0)
    elif b==1:
        # location of cylinder roots
        where = np.where(diff)
    
    roots = np.zeros(len(where[0])) # setup an empty array to store roots
    
    for i, j in enumerate(where[0]):
       roots[i] = op.brentq(func, z[j], z[j+1], args=(Bi)) # roots for a given interval
       #roots[i] = op.root_scalar(func,bracket=[z[j],z[j+1]],args=(Bi),method='brentq').root
    return roots    # return a list of the positive roots


def roots2(z, b, Bi):
    """
    Returns a list of positive roots from zeta, Bi equation used for the
    analytical solution of 1D transient heat conduction for a solid sphere,
    cylinder, or slab shape.
    z = range of zeta values to test for positive roots
    b = shape factor where 2 sphere, 1 cylinder, 0 slab
    Bi = Biot number h*L/k, (-)
    """
    
    if b==2:
        aa=1
        func = funcZetaSph          # declare sphere function as roots function
    elif b==1:
        fz = funcZetaCyl(z, Bi)     # evaluate cylinder function at z-values
        func = funcZetaCyl          # declare cylinder function as roots function
    elif b==0:
        fz = funcZetaSlab(z, Bi)    # evaluate slab function at the z-values
        func = funcZetaSlab         # declare slab function as roots function
    
    # b as the shape factor where 2 sphere, 1 cylinder, 0 slab
    # use the np.where function to find array index of value change
    # note that np.where returns a tuple
    diff=0
    if b == 2 or b==0:
        # location of sphere or slab roots
        # find index of only positive value change thus ignoring singularities
        where = np.where(diff>0)
    elif b==1:
        # location of cylinder roots
        where = np.where(diff)
    ii=range(0,800,1)
    roots = np.zeros(len(ii)) # setup an empty array to store roots
    n=0
    for i in ii:
        n=n+1
        if Bi==1.0:
            roots[i]=(2*n-1)*np.pi/2
            return roots
        if (1.0-Bi)>0.0:
            z1=(n-1)*np.pi
            z2=np.pi/2+(n-1)*np.pi
        elif (1.0-Bi)<0.0:
                z1=np.pi/2.0+(n-1.0)*np.pi
                z2=np.pi+(n-1.0)*np.pi
                zz=(z1+z2)/2.0
        roots[i] = op.root_scalar(func,x0=zz, x1=z2*0.6+z1*(1-0.6),args=(Bi)).root # roots for a given interval
        
    return roots    # return a list of the positive roots


