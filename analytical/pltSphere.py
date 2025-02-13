import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def sphereCoord(radius=1.0, resolution=100):
    import numpy as np
    u, v = np.mgrid[-np.pi:0:resolution*1j, 0:np.pi:resolution*1j]
    rr=radius
    X = rr * np.cos(u)*np.sin(v) 
    Y = rr * np.sin(u)*np.sin(v) 
    Z = rr * np.cos(v) 
    return X, Y, Z

def prismCoord(s, height=1.0, depth=1.0, resolution=100):
    # Create a grid of points
    zz = np.linspace(0, height, resolution)
    xx = np.linspace(-s/2, s/2, resolution)
    yy = np.linspace(-depth/2, depth/2, resolution)
    
    X, Y = np.meshgrid(xx, yy)
    Z = np.zeros_like(X)
    
    # Top face
    top_face = (X, Y, Z + height)
    
    # Bottom face
    bottom_face = (X, Y, Z)
    
    # Side faces
    side_faces = []
    for i in range(resolution):
        side_faces.append((X[i, :], np.full_like(Y[i, :], Y[i, 0]), Z[i, :] + zz))
      
    X = np.concatenate([top_face[0].flatten(), bottom_face[0].flatten()] + [face[0].flatten() for face in side_faces])
    Y = np.concatenate([top_face[1].flatten(), bottom_face[1].flatten()] + [face[1].flatten() for face in side_faces])
    Z = np.concatenate([top_face[2].flatten(), bottom_face[2].flatten()] + [face[2].flatten() for face in side_faces])
        
    return X, Y, Z

def plotSphere(X,Y,Z, clrMatrix=0, ccLim=(0,1),colorMap="Viridis"):
   
    tt=[go.Surface(x=X,
                             y=Y, 
                             z=Z, 
                             surfacecolor=clrMatrix,
                             cmin=ccLim[0],
                             cmax=ccLim[1],
                             colorscale=colorMap
                             )]
    return tt


def plotPrism(X,Y,Z, clrMatrix=0, ccLim=(0,1),colorMap="Viridis"):
    tt=[go.Surface(x=X,
                             y=Y, 
                             z=Z, 
                             surfacecolor=clrMatrix,
                             cmin=ccLim[0],
                             cmax=ccLim[1],
                             colorscale=colorMap
                             )]
    return tt

def circleCoord(r):
    import numpy as np
    # Create a grid of points 
    resolution = len(r)
    theta = np.linspace(0, 2 * np.pi, resolution) 
    T, R = np.meshgrid(theta, r) # Calculate coordinates 
    X = R * np.cos(T) 
    Y = np.zeros_like(X)
    Z = R * np.sin(T) 
    return X, Y, Z

def rectCoord(s,h=1.0):
    import numpy as np
    # Create a grid of points 
    resolution = len(s)
    zz = np.linspace(-h/2, h/2, resolution)
    xx = np.concatenate((-np.flipud(s), s[1:]))
    Z, X = np.meshgrid(zz,xx)
    Y = np.zeros_like(X)
    return X, Y, Z

def plotCircle(X,Y,Z, clrMatrix=0, ccLim=(0,1),colorMap="Viridis"):
        
    if isinstance(clrMatrix, int) and clrMatrix == 0:
            clrMatrix = Z
    
    tt=[go.Surface(x=X,
                             y=Y, 
                             z=Z, 
                             surfacecolor=clrMatrix,
                             cmin=ccLim[0],
                             cmax=ccLim[1],
                             colorscale=colorMap
                             )]
    return tt


def plotRect(X,Y,Z, clrMatrix=0, ccLim=(0,1),colorMap="Viridis"):
        
    
    if np.prod(clrMatrix.shape)==1: 
        clrMatrix=np.ones(X.shape)*clrMatrix

    tt=[go.Surface(x=X,
                   y=Y, 
                   z=Z, 
                   surfacecolor=clrMatrix,
                   cmin=ccLim[0],
                   cmax=ccLim[1],
                   colorscale=colorMap
                   )]
    return tt, X, Y, Z


def drawRectPrismFromVertices(vertices, clrMatrix, ccLim, colorMap):
    # vertices should be a list of 8 tuples, each representing a vertex (x, y, z)
    if len(vertices) != 8:
        raise ValueError("There should be exactly 8 vertices")

    # Extract the left and right faces only
    left_right_faces = [
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right face
        [vertices[3], vertices[0], vertices[4], vertices[7]]   # Left face
    ]

    # Combine the faces
    faces = [left_right_faces[0] + left_right_faces[1]]

    X = np.array([[v[0] for v in face] for face in faces])
    Y = np.array([[v[1] for v in face] for face in faces])
    Z = np.array([[v[2] for v in face] for face in faces])

    tt = plotRect(X, Y, Z, clrMatrix, ccLim, colorMap)
    return tt


if __name__ == "__main__":
    plotSphere()
    print("Done!")

def createSliderSteps(varSlider, M1, M2,M3):
    slider_steps = []
    discrSpaz=(M1.shape[0],M1.shape[1])
    for j in range(len(varSlider)):
        slider_steps = slider_steps + [{
            "method": "update",
            "label": str(varSlider[j]) +" sec.",
            "args": [{
                    "surfacecolor": [
                        M1[:, :, j],
                        np.ones(discrSpaz)*M2[j]
                        ],
                        "y": [[M3[:,j]],
                              [1]]
                    }, {
                "title": f"Temperature Profile - Time Step [sec]: {varSlider[j]}"
            }]
        }]


    return slider_steps
    