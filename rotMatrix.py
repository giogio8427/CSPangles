import numpy as np
from numpy.linalg import norm

def rotationMatrixAroundY(angleRad):
    return np.array([[np.cos(angleRad), 0, np.sin(angleRad)], [0, 1, 0], [-np.sin(angleRad), 0, np.cos(angleRad)]])
        
def rotationMatrixAroundZ(angleRad):
    return np.array([[np.cos(angleRad), -np.sin(angleRad), 0], [np.sin(angleRad), np.cos(angleRad), 0], [0, 0, 1]])

def rotationMatrixAroundX(angleRad):
    return np.array([[1, 0, 0], [0, np.cos(angleRad), -np.sin(angleRad)], [0, np.sin(angleRad), np.cos(angleRad)]])

def rotationMatrixAroundAxis(axis, angleRad):
    axis = axis / norm(axis)  # Normalize the axis vector
    cos_angle = np.cos(angleRad)
    sin_angle = np.sin(angleRad)
    ux, uy, uz = axis
    return np.array([[cos_angle + ux**2 * (1 - cos_angle), ux * uy * (1 - cos_angle) - uz * sin_angle, ux * uz * (1 - cos_angle) + uy * sin_angle],
                   [uy * ux * (1 - cos_angle) + uz * sin_angle, cos_angle + uy**2 * (1 - cos_angle), uy * uz * (1 - cos_angle) - ux * sin_angle],
                   [uz * ux * (1 - cos_angle) - uy * sin_angle, uz * uy * (1 - cos_angle) + ux * sin_angle, uz * uz * (1 - cos_angle) + cos_angle]])

def rotatePoints(x,y,z,rotationMatrix):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            point = np.array([x[i,j], y[i,j], z[i,j]])
            rotated = rotationMatrix @ point
            x[i,j], y[i,j], z[i,j] = rotated[0], rotated[1], rotated[2]
    return x,y,z