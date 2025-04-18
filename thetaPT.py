# %%
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from scipy.linalg import norm
import matplotlib.pyplot as plt
import sunposition as sp
from ipywidgets import interact, IntSlider, FloatSlider, HBox, Layout, interactive_output
from IPython.display import display




#region defining functions

def sunVect(azimuth, zenith):
    return np.array([np.sin(zenith) * np.cos(azimuth), np.sin(zenith) * np.sin(azimuth), np.cos(zenith)])

def parabolicCylinderSurface(apertureWidht=10,focalLength=5,length=10, nDiscr=100):
    x = np.linspace(-apertureWidht/2,apertureWidht/2,nDiscr)
    z = x**2/(4*focalLength)
    y= np.linspace(-length/2.,length/2.,nDiscr)
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
#endregion 

#region read-input 

# Default values
lon = 14.1981  # Rome, Italy longitude
lat = 41.9028  # Rome, Italy latitude
day = 21
month = 6
year = 2023
axRot_az=0.0

# Use input with default values
lon = input(f'Enter the longitude [{lon}]: ') or lon
lat = input(f'Enter the latitude [{lat}]: ') or lat
day = input(f'Enter the day [{day}]: ') or day
month = input(f'Enter the month [{month}]: ') or month
year = input(f'Enter the year [{year}]: ') or year
axRot_az = input(f'Enter the azimuth rotation angle [{0}]°: ') or axRot_az

timeStep=1 # [h]
#endregion
firstRun=True


def initialCompute(lon, lat, year, month, day):
    global azArray, zenArray, hourArray, endArr, startArr,ii, iiSun
    # Initialize arrays for azimuth and zenith angles
    azArray=np.zeros(24)
    zenArray=np.zeros(24)
    hourArray=np.linspace(0,23,24)
    lon=float(lon)
    lat=float(lat) 
    year=str(year)
    month=str(month)
    day=str(day)
    if int(month)<10:
        month='0'+str(month)
    if int(day)<10:  
        day='0'+str(day)

    stringDay=year+'-'+month+'-'+day
    lon  = np.array([lon])
    lat = np.array([lat])

    sp.disable_jit()
    for ii in hourArray:
        addStr=''
        if ii<10:
            addStr='0'
        currTime=np.datetime64(stringDay + 'T' + addStr + str(int(ii)) + ':00:00')
        azArray[int(ii)],zenArray[int(ii)] = sp.sunpos(currTime,lat,lon,0)[:2] #discard RA, dec, H

    iiSun=np.argmax((zenArray>0) & (zenArray<=90))
    ii=iiSun
    tt=np.zeros(24)
    endArr=np.zeros((3,24))
    startArr=np.zeros((3,24))



#now = sp.time_to_datetime64('now')


#az[0]=130.
#zen[0]=70.
# Sun-path diagram
#xSunPath=np.zeros(24)
#ySunPath=np.zeros(24)
#zSunPath=np.zeros(24)
def comp(zenith, azimuth, axRot_az=0):
    global projSunHor, zen, az,ii, startArr, sunVector, endArr, iiSun, iiFin, end2, normalVect, start, end, x, y, z, xAp, yAp, zAp, xLon, yLon, zLon, xHor, yHor, zHor, xArc, yArc, zArc, xArcZen, yArcZen, zArcZen, xArcAz, yArcAz, zArcAz, xSun, ySun, zSun, incAngle, azArray, zenArray, length,thetaPerp

    zen=np.rad2deg(zenith)  
    az=np.rad2deg(azimuth)

    length=10.
    rotMatrAxRot=rotationMatrixAroundZ(np.deg2rad(axRot_az))
    azApparent=azimuth+np.deg2rad(axRot_az)
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

    #while ((zenArray[ii]<=90) & (zenArray[ii]>0)):
    start=np.array([centX,centY,maxZ]).reshape((3,1))  # Start point of the sun vector    
     #   zenith = np.deg2rad(zenArray[ii])
     #   azimuth = np.deg2rad(azArray[ii])
    sunVector = np.array([np.sin(zenith) * np.cos(azimuth), np.sin(zenith) * np.sin(azimuth), np.cos(zenith)])
    sunVectorApparent = np.array([np.sin(zenith) * np.cos(azApparent), np.sin(zenith) * np.sin(azApparent), np.cos(zenith)])
    projPerp=np.array([0,sunVectorApparent[1],sunVectorApparent[2]])
    projPerp=projPerp/norm(projPerp)
    if sunVectorApparent[1]>=0.0:
        thetaPerp=  np.arccos(np.dot(projPerp, np.array([0, 0, 1])))
    else:
        thetaPerp= -np.arccos(np.dot(projPerp, np.array([0, 0, 1])))
    rotZ=rotationMatrixAroundZ(np.deg2rad(90.0+axRot_az))
    rotAxis=rotationMatrixAroundZ(np.deg2rad(axRot_az)) @ np.array([1,0,0])
    rotatedTrack=rotationMatrixAroundAxis(rotAxis,-thetaPerp)
  
    #start=rotatedTrack @ start
#        startArr[:,ii]=start[:,0]
#        endArr[:,ii] = startArr[:,ii] + sunVector * length
    #ii=ii+1
    
    #iiFin=ii-1     

    #x,y,z=rotatePoints(x,y,z,rotZ)
    #xAp,yAp,zAp=rotatePoints(xAp,yAp,zAp,rotZ)
    #xLon,yLon,zLon=rotatePoints(xLon,yLon,zLon,rotZ)

    #start=np.array([centX,centY,maxZ]).reshape((3,1))  # Start point of the sun vector    
    #rotatedTrack=rotationMatrixAroundY(-thetaPerp)
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
    xSun,ySun,zSun=sphereSurface(radius=0.5,coord=[coordSun[0], coordSun[1], coordSun[2]], nDiscr=30)

    if sunVector[1]>=0.0:
        tt=length/2.0
    else:
        tt=-length/2.0

    xHor,yHor,zHor=xHor+start[0],yHor+start[1]+tt,zHor+start[2]
    xLon,yLon,zLon=xLon+start[0],yLon+start[1],zLon+start[2]
    xAp,yAp,zAp=xAp+start[0],yAp+start[1],zAp+start[2]
    projSunHor=(np.array([sunVector[0],sunVector[1],0])).reshape(3,1)

    xArcZen,yArcZen, zArcZen=arcCircle3D(length/2.0,start,start+np.array([0,0,0.5*length]).reshape(3,1),end, nDiscr=25)
    #xArcAz,yArcAz, zArcAz=arcCircle3D(length/2.0,start,start+projSunHor*0.5*length,start+np.array([0,-0.5*length,0]).reshape(3,1), type='az', nDiscr=25)
    #xArcAz,yArcAz = arcCircle2D(length/2.0,start,start+np.array([0,-0.5*length,0]).reshape(3,1),start+projSunHor*0.5*length, nDiscr=50)
    xArcAz,yArcAz = arcCircle2DAngle(length/2.0,start,azimuth,nDiscr=50)

    zArcAz=np.zeros(len(xArcAz))+start[2]
    #xArcAz,yArcAz, zArcAz=arcCircle3D(length/2.0,start,start+projSunHor*0.5*length, start+np.array([0,-0.5*length,0]).reshape(3,1), nDiscr=25)
    projSunHor=projSunHor/norm(projSunHor)
    projSunHor=projSunHor*length/2.0
    return

#endregion

indTest=16

initialCompute(lon, lat, year, month, day)
comp(np.deg2rad(zenArray[indTest]), np.deg2rad(azArray[indTest]))

#region Plotly section
# ============================================================================
#                               Draw figure section
# ============================================================================

def drawFig():
    maxX=np.max(x)
    fig = go.Figure()
    test=False
    fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale=[[0, 'grey'], [1, 'grey']], opacity=0.9, showscale=False, name='Parabolic Cylinder'))
    color = 'orange'
    color2='blue'
    #fig = go.Figure(data=[go.Surface(x=x, y=y, z=z)])
    fig.add_trace(go.Cone(
        x=[start[0,0]], y=[start[1,0]], z=[start[2,0]],
        u=[-sunVector[0]], v=[-sunVector[1]], w=[-sunVector[2]],
        sizemode="absolute",
        sizeref=1.,
        colorscale=[[0, color], [1, color]],
        showscale=False
    ))
    fig.add_trace(go.Scatter3d(
        x=[start[0,0], end[0,0]],
        y=[start[1,0], end[1,0]],
        z=[start[2,0], end[2,0]],
        mode='lines',
        line=dict(color=color, width=10),
        name='Sun Vector'
    ))
    fig.add_trace(go.Cone(
        x=[end2[0,0]], y=[end2[1,0]], z=[end2[2,0]],
        u=[normalVect[0]*length], v=[normalVect[1]*length], w=[normalVect[2]*length],
        sizemode="absolute",
        sizeref=1.,
        colorscale=[[0, color2], [1, color2]],
        showscale=False
    ))
    fig.add_trace(go.Scatter3d(
        x=[start[0,0], end2[0,0]],
        y=[start[1,0], end2[1,0]],
        z=[start[2,0], end2[2,0]],
        mode='lines',
        line=dict(color=color2, width=10),
        name='Normal Unit Vector'
    ))

    fig.add_trace(go.Scatter3d(
        x=xArc,
        y=yArc,
        z=zArc,
        mode='lines',
        line=dict(color='black', width=5),
        name='IncAngle'
    ))

    fig.add_trace(go.Scatter3d(
        x=xArcZen,
        y=yArcZen,
        z=zArcZen,
        mode='lines',
        line=dict(color='black', width=5),
        name='ZenithAngle'
    ))


    fig.add_trace(go.Scatter3d(
        x=xArcAz,
        y=yArcAz,
        z=zArcAz,
        mode='lines',
        line=dict(color='red', width=5),
        name='AzimuthAngle'
    ))

    fig.add_trace(go.Surface(
        x=xAp, y=yAp, z=zAp,
        colorscale='gray',
        opacity=0.5,
        showscale=False,
        name='PlaneAperture'
    ))

    fig.add_trace(go.Surface(
        x=xLon, y=yLon, z=zLon,
        colorscale='blues',
        opacity=0.5,
        showscale=False,
        name='LongPlane'
    ))

    fig.add_trace(go.Surface(
        x=xSun, y=ySun, z=zSun,
        colorscale='solar',
        opacity=1.0,
        showscale=False,
        name='Sun'
    ))

    fig.add_trace(go.Scatter3d(
        x=endArr[0,iiSun:iiFin+1],
        y=endArr[1,iiSun:iiFin+1],
        z=endArr[2,iiSun:iiFin+1],
        mode='lines',
        line=dict(color='yellow', width=5),
        name='Sun Path'
    ))

    fig.add_trace(go.Cone(
        x=[start[0,0]], y=[start[1,0]], z=[start[2,0]+length],
        u=[0], v=[0], w=[length],
        sizemode="absolute",
        sizeref=1.,
        colorscale=[[0, 'black'], [1, color]],
        showscale=False
        ))

    fig.add_trace(go.Scatter3d(
        x=[start[0,0], start[0,0]],
        y=[start[1,0], start[1,0]],
        z=[start[2,0], start[2,0]+length],
        mode='lines',
        line=dict(color='black', width=10),
        name='VertDir'
    ))

    fig.add_trace(go.Cone(
        x=[start[0,0]+length], y=[start[1,0]], z=[start[2,0]],
        u=[length], v=[0], w=[0],
        sizemode="absolute",
        sizeref=1.,
        colorscale=[[0, 'black'], [1, color]],
        showscale=False
        ))

    fig.add_trace(go.Scatter3d(
        x=[start[0,0], start[0,0]+length],
        y=[start[1,0], start[1,0]],
        z=[start[2,0], start[2,0]],
        mode='lines',
        line=dict(color='black', width=10),
        name='HorzDir'
    ))

    fig.add_trace(go.Surface(
        x=xHor, y=yHor, z=zHor,
        colorscale='blues',
        opacity=0.5,
        showscale=False,
        name='Horizontal Plane'
    ))

    fig.add_trace(go.Scatter3d(
    x=[start[0,0], start[0,0]+projSunHor[0,0]],
    y=[start[1,0], start[1,0]+projSunHor[1,0]],
    z=[start[2,0], start[2,0]+projSunHor[2,0]],
    mode='lines',
    line=dict(color='black', width=2),
    name='HorzDir'
    ))

    fig.update_layout(
        scene=dict(
            annotations=[
            dict(
                showarrow=False,
                x=xArc[len(xArc)//2],
                y=yArc[len(xArc)//2],
                z=zArc[len(xArc)//2],
                text="θinc: "+str(format(incAngle, '.2f'))+"°",
                xanchor="left",
                xshift=10,
                opacity=0.7),
            dict(
                showarrow=False,
                x=start[0,0]+length,
                y=start[1,0],
                z=start[2,0],
                text="North",
                yshift=25,
                opacity=0.7,
                font=dict(
                    color="black",
                    size=16
                ),
                ),      
            dict(
                showarrow=False,
                x=xArcAz[len(xArcAz)//2],
                y=yArcAz[len(xArcAz)//2],
                z=zArcAz[len(xArcAz)//2],
                text="γ: "+str(format(az, '.2f'))+"°",
                yshift=25,
                textangle=0,
                font=dict(
                    color="black",
                    size=12
                ),
                ),
            dict(
                showarrow=True,
                x=xArcZen[len(xArcZen)//2],
                y=yArcZen[len(xArcZen)//2],
                z=zArcZen[len(xArcZen)//2],
                text="θzen: "+str(format(zen, '.2f'))+"°",
                xanchor="left",
                yanchor="bottom"
            )]
        ),
    )
    fig.update_layout(scene_aspectmode='manual',
        scene_aspectratio=dict(x=1, y=1, z=1))
    fig.update_layout(scene=dict(
        xaxis=dict(range=[-length*1.2, length*1.2]),
        yaxis=dict(range=[-length*1.2, length*1.2]),
        zaxis=dict(range=[-length/2, length*1.2]),
        ))
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
        height=800  # Set the height of the figure
    )
    return fig
#endregion



while ((zenArray[ii]<=90) & (zenArray[ii]>0)):
    sunVector = np.array([np.sin(np.deg2rad(zenArray[ii])) * np.cos(np.deg2rad(azArray[ii])), 
                          np.sin(np.deg2rad(zenArray[ii])) * np.sin(np.deg2rad(azArray[ii])), 
                          np.cos(np.deg2rad(zenArray[ii]))])
    comp(np.deg2rad(zenArray[ii]), np.deg2rad(azArray[ii]))
    startArr[:,ii]=start[:,0]
    endArr[:,ii] = startArr[:,ii] + sunVector * length
    ii=ii+1

iiFin=ii-1
azimuth = np.deg2rad(azArray[indTest])
zenith = np.deg2rad(zenArray[indTest])

if firstRun==True:
    fig=drawFig()
    #fig.update_layout(autosize=True) # remove height=800
    fig.show()
    #region print results
    print('Azimuth: ', format(np.rad2deg(azimuth), '.2f'), '°')  
    print('Zenith: ', format(np.rad2deg(zenith), '.2f'), '°')
    print('Incidence Angle: ', format(incAngle, '.2f'), '°')
    print('thetaPerp: ', format(np.rad2deg(thetaPerp), '.2f'), '°')
    firstRun=False


daySlider=IntSlider(min=1, max=31, step=1, value=day, description="Day [d]", layout=Layout(width='100%'))
monthSlider=IntSlider(min=1, max=12, step=1, value=month, description="Month [m]", layout=Layout(width='100%'))
yearSlider=IntSlider(min=2000, max=2100, step=1, value=year, description="Year [y]", layout=Layout(width='100%'))
hourSlider=IntSlider(min=0, max=23, step=1, value=12, description="Hour [h]", layout=Layout(width='100%'))
latSlider=FloatSlider(min=-90, max=90, step=0.1, value=lat, description="Latitude [°]", layout=Layout(width='100%'))
lonSlider=FloatSlider(min=-180, max=180, step=0.1, value=lon, description="Longitude[°]", layout=Layout(width='100%'))
axRotAzSlider=FloatSlider(min=-180, max=180, step=1, value=axRot_az, description="AxOrient [°]", layout=Layout(width='100%'))
slider_box = HBox([daySlider, monthSlider, yearSlider, hourSlider, latSlider, lonSlider,axRotAzSlider], layout=Layout(display='flex', flex_flow='row', align_items='center', width='100%'))
# Display the slider box
display(slider_box)

# Update sun vector and related calculations
def update_figure(day, month, year, hour,lat,lon, axRot_az):
    hh=hour
    initialCompute(lon, lat, year, month, day)
    azimuth = np.deg2rad(azArray[hour])
    zenith = np.deg2rad(zenArray[hour])
    
    comp(zenith, azimuth, axRot_az)
 
    with fig.batch_update():
        #fig.update_layout(scene=dict(
        #    xaxis=dict(range=[-length, length]),
        #    yaxis=dict(range=[-length, length]),
        #    zaxis=dict(range=[-length, length]),
        #))

        fig.update_layout(scene_aspectmode='manual',
                  scene_aspectratio=dict(x=1, y=1, z=1))
        fig.data[0].update(x=x, y=y, z=z)
        fig.data[1].update(x=[start[0,0]], y=[start[1,0]], z=[start[2,0]])
        fig.data[2].update(x=[start[0,0], end[0,0]], y=[start[1,0], end[1,0]], z=[start[2,0], end[2,0]])
        fig.data[3].update(x=[end2[0,0]], y=[end2[1,0]], z=[end2[2,0]],u=[normalVect[0]*length], v=[normalVect[1]*length], w=[normalVect[2]*length])
        fig.data[4].update(x=[start[0,0], end2[0,0]], y=[start[1,0], end2[1,0]], z=[start[2,0], end2[2,0]])
        fig.data[5].update(x=xArc, y=yArc, z=zArc)
        fig.data[6].update(x=xArcZen, y=yArcZen, z=zArcZen)
        fig.data[7].update(x=xArcAz, y=yArcAz, z=zArcAz)
        fig.data[8].update(x=xAp, y=yAp, z=zAp)
        fig.data[9].update(x=xLon, y=yLon, z=zLon)
        fig.data[10].update(x=xSun, y=ySun, z=zSun)
        fig.data[11].update(x=endArr[0,iiSun:iiFin+1], y=endArr[1,iiSun:iiFin+1], z=endArr[2,iiSun:iiFin+1])
        fig.data[12].update(x=[start[0,0]], y=[start[1,0]], z=[start[2,0]+length])
        fig.data[13].update(x=[start[0,0], start[0,0]], y=[start[1,0], start[1,0]], z=[start[2,0], start[2,0]+length])
        fig.data[14].update(x=[start[0,0]+length], y=[start[1,0]], z=[start[2,0]],u=[length], v=[0], w=[0])
        fig.data[15].update(x=[start[0,0], start[0,0]+length], y=[start[1,0], start[1,0]], z=[start[2,0], start[2,0]])
        fig.data[16].update(x=xHor, y=yHor, z=zHor)
        fig.data[17].update(x=[start[0,0], start[0,0]+projSunHor[0,0]], y=[start[1,0], start[1,0]+projSunHor[1,0]], z=[start[2,0], start[2,0]+projSunHor[2,0]])

    fig.update_layout(
        scene=dict(
            annotations=[
            dict(
                showarrow=False,
                x=xArc[len(xArc)//2],
                y=yArc[len(xArc)//2],
                z=zArc[len(xArc)//2],
                text="θinc: "+str(format(incAngle, '.2f'))+"°",
                xanchor="left",
                xshift=10,
                opacity=0.7),
            dict(
                showarrow=False,
                x=start[0,0]+length,
                y=start[1,0],
                z=start[2,0],
                text="North",
                yshift=25,
                opacity=0.7,
                font=dict(
                    color="black",
                    size=16
                ),
                ),      
            dict(
                showarrow=False,
                x=xArcAz[len(xArcAz)//2],
                y=yArcAz[len(xArcAz)//2],
                z=zArcAz[len(xArcAz)//2],
                text="γ: "+str(format(az, '.2f'))+"°",
                yshift=25,
                textangle=0,
                font=dict(
                    color="black",
                    size=12
                ),
                ),
            dict(
                showarrow=True,
                x=xArcZen[len(xArcZen)//2],
                y=yArcZen[len(xArcZen)//2],
                z=zArcZen[len(xArcZen)//2],
                text="θzen: "+str(format(zen, '.2f'))+"°",
                xanchor="left",
                yanchor="bottom"
            )]
        ),
    )        
    fig.show()



# Connect the function to the sliders without creating duplicate UI elements
output = interactive_output(update_figure, {
    'day': daySlider,
    'month': monthSlider,
    'year': yearSlider,
    'hour': hourSlider,
    'lat': latSlider,
    'lon': lonSlider,
    'axRot_az': axRotAzSlider
})
display(output)

        









#endregion

# Matplotlib section
#fig2 = plt.figure()
#ax = fig2.add_subplot(111, projection='3d')
#ax.plot_surface(x, y, z, cmap='viridis')
#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#ax.set_zlabel('Z')
#ax.quiver(end[0], end[1], end[2], -sunVector[0], -sunVector[1], -sunVector[2], length=length, color=color)

#ax.plot_surface(xAp, yAp, zAp, alpha=0.5, color='gray')
#plt.show()
#a=1





# %%
