import numpy as np
from scipy.linalg import norm

def sunVect(azimuth, zenith):
    return np.array([np.sin(zenith) * np.cos(azimuth), np.sin(zenith) * np.sin(azimuth), np.cos(zenith)])

def parabolicCylinderSurface(apertureWidht=10,focalLength=5,length=10, nDiscr=10):
    x = np.linspace(-apertureWidht/2,apertureWidht/2,nDiscr)
    z = x**2/(4*focalLength)
    y= np.linspace(-length/2.,length/2.,nDiscr//4)
    x, y = np.meshgrid(x, y)
    z = np.tile(z, (y.shape[0], 1))
    return x, y, z

def planeSurface(trasl=[0,0,5],length=10, nDiscr=4):
    x = np.linspace(-length/2,length/2,nDiscr)
    y = np.linspace(-length/2,length/2,nDiscr)
    x, y = np.meshgrid(x, y)
    z = np.zeros(x.shape)
    x=x+trasl[0]
    y=y+trasl[1]
    z=z+trasl[2]
    return x, y, z

def sphereSurface(radius=5,coord=[0,0,0], nDiscr=30):
    theta=np.linspace(0,2*np.pi,nDiscr)
    phi=np.linspace(0,np.pi,nDiscr)
    theta, phi = np.meshgrid(theta, phi)
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    x=x+coord[0]
    y=y+coord[1]
    z=z+coord[2]
    return x, y, z

def arcCircle3D(radius,center,point1,point2, nDiscr=25):
    # Normalize vectors from center to point1 and point2
    v1 = ((point1 - center) / norm(point1 - center)).reshape(3)
    v2 = ((point2 - center) / norm(point2 - center)).reshape(3)
    normal = np.cross(v1, v2)
    normal = normal / norm(normal)
    angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
    x = np.zeros(nDiscr)
    y = np.zeros(nDiscr)
    z = np.zeros(nDiscr)
    # Generate points along the arc
    t = np.linspace(0, angle, nDiscr)
    ii = 0
    for angle in t:
        point = radius * (np.cos(angle) * v1 + np.sin(angle) * np.cross(normal, v1))
        point += center.reshape(3)  # Add center after vector operations
        x[ii], y[ii], z[ii] = point[0], point[1], point[2]
        ii += 1
    return x, y, z

def arcCircle2D(radius, center, point1, point2, nDiscr=25):
    # Normalize vectors from center to point1 and point2
    v1 = ((point1[0:2] - center[0:2]) / norm(point1[0:2] - center[0:2])).reshape(2)
    v2 = ((point2[0:2] - center[0:2]) / norm(point2[0:2] - center[0:2])).reshape(2)
    normal = np.cross(v1, v2)
    normal = normal / norm(normal)
    angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))  # Positive angle for counter-clockwise
    if np.cross(v1, v2) < 0:  # Adjust for clockwise direction
        angle = -angle
    angle=-np.arctan2(v1[0]*v2[1]-v1[1]*v2[0],v1[0]*v2[0]+v1[1]*v2[1])
    x = np.zeros(nDiscr)
    y = np.zeros(nDiscr)
    # Generate points along the arc
    t = np.linspace(0, angle, nDiscr)
    if angle<0:
        t = np.linspace(angle,0, nDiscr)
    ii = 0
    for angle in t:
        point = radius * (np.cos(angle) * v1 + np.sin(angle) * np.array([-v1[1], v1[0]]))  # Rotate clockwise
        point += center[0:2].reshape(2)  # Add center after vector operations
        x[ii], y[ii] = point[0], point[1]
        ii += 1
    return x, y

def arcCircle2DAngle(radius,center,angle,nDiscr=25):
    x = np.zeros(nDiscr)
    y = np.zeros(nDiscr)
    # Generate points along the arc
    t = np.linspace(0, angle, nDiscr)
    ii = 0
    for angle in t:
        point = radius * np.array([np.cos(angle), np.sin(angle)])  # Rotate clockwise
        point += center[0:2].reshape(2)  # Add center after vector operations
        x[ii], y[ii] = point[0], point[1]
        ii += 1
    return x, y