from geoElem import parabolicCylinderSurface, planeSurface, sphereSurface,  arcCircle3D, arcCircle2DAngle
from rotMatrix import rotationMatrixAroundY, rotationMatrixAroundZ, rotationMatrixAroundAxis, rotatePoints
from scipy.linalg import norm
import numpy as np
def comp(zenith, azimuth, axRot_az=0):
    global projSunHor, zen, az,ii, startArr, sunVector, endArr, iiSun, end2, normalVect, start, end, x, y, z, xAp, yAp, zAp, xLon, yLon, zLon, xHor, yHor, zHor, xArc, yArc, zArc, xArcZen, yArcZen, zArcZen, xArcAz, yArcAz, zArcAz, xSun, ySun, zSun, incAngle, azArray, zenArray, length,thetaPerp

    zen=np.rad2deg(zenith)  
    az=np.rad2deg(azimuth)
    length=10.
    azApparent=azimuth-np.deg2rad(axRot_az)
    x,y,z=parabolicCylinderSurface()    
    maxZ=np.max(z)
    centX=(np.max(x)+np.min(x))/2.0
    centY=(np.max(y)+np.min(y))/2.0
    start=np.array([centX,centY,maxZ]).reshape((3,1))  # Start point of the sun vector    
    xAp,yAp,zAp=planeSurface(trasl=[0,0,0])
    xLon,yLon,zLon=planeSurface(trasl=[0,0,0])
    xHor,yHor,zHor=planeSurface(trasl=[0,0,0])
    rotAngle = np.deg2rad(90)  # Rotate by 90 degrees
    rotMatr=rotationMatrixAroundY(rotAngle)
    xLon,yLon,zLon=rotatePoints(xLon,yLon,zLon,rotMatr)

    start=np.array([centX,centY,maxZ]).reshape((3,1))  # Start point of the sun vector    

    sunVector = np.array([np.sin(zenith) * np.cos(azimuth), np.sin(zenith) * np.sin(azimuth), np.cos(zenith)])
    sunVectorApparent = np.array([np.sin(zenith) * np.cos(azApparent), np.sin(zenith) * np.sin(azApparent), np.cos(zenith)])
    projPerp=np.array([0,sunVectorApparent[1],sunVectorApparent[2]])
    projPerp=projPerp/norm(projPerp)
    if sunVectorApparent[1]>=0.0:
        thetaPerp=  np.arccos(np.dot(projPerp, np.array([0, 0, 1])))
    else:
        thetaPerp= -np.arccos(np.dot(projPerp, np.array([0, 0, 1])))
    trackingAngle=np.rad2deg(thetaPerp)
    rotZ=rotationMatrixAroundZ(np.deg2rad(90.0+axRot_az))
    rotAxis=rotationMatrixAroundZ(np.deg2rad(axRot_az)) @ np.array([1,0,0])
    rotatedTrack=rotationMatrixAroundAxis(rotAxis,-thetaPerp)
  
    rr=np.matmul(rotatedTrack,rotZ)
    x,y,z=rotatePoints(x,y,z,rr)
    xAp,yAp,zAp=rotatePoints(xAp,yAp,zAp,rr)
    xLon,yLon,zLon=rotatePoints(xLon,yLon,zLon,rr)
    
    start=rr @ start
    end = start + sunVector.reshape((3,1)) * length

    normalVect=rr @ np.array([0,0,1])
    end2=start+normalVect.reshape((3,1))*length

    xArc,yArc, zArc=arcCircle3D(length,start,end,end2, nDiscr=25)

    incAngle=np.rad2deg(np.arccos(np.dot(sunVector,normalVect)/(norm(sunVector)*norm(normalVect))))

    coordSun=start+sunVector.reshape((3,1))*length
    xSun,ySun,zSun=sphereSurface(radius=0.5,coord=[coordSun[0], coordSun[1], coordSun[2]], nDiscr=8)

    if sunVector[1]>=0.0:
        tt=length/2.0
    else:
        tt=-length/2.0

    xHor,yHor,zHor=xHor+start[0],yHor+start[1]+tt,zHor+start[2]
    xLon,yLon,zLon=xLon+start[0],yLon+start[1],zLon+start[2]
    xAp,yAp,zAp=xAp+start[0],yAp+start[1],zAp+start[2]
    projSunHor=(np.array([sunVector[0],sunVector[1],0])).reshape(3,1)

    xArcZen,yArcZen, zArcZen=arcCircle3D(length/2.0,start,start+np.array([0,0,0.5*length]).reshape(3,1),end, nDiscr=25)
    xArcAz,yArcAz = arcCircle2DAngle(length/2.0,start,azimuth,nDiscr=50)
    zArcAz=np.zeros(len(xArcAz))+start[2]

    projSunHor=projSunHor/norm(projSunHor)
    projSunHor=projSunHor*length/2.0
    return incAngle, trackingAngle