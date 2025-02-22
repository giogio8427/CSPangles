from plotly.graph_objects import Surface
from numpy import mgrid, pi, cos, sin,  linspace, prod,meshgrid, zeros_like, concatenate, flipud, full_like, ones, array

def sphereCoord(radius=1.0, resolution=100):
    u, v = mgrid[-pi:0:resolution*1j, 0:pi:resolution*1j]
    rr=radius
    X = rr * cos(u)*sin(v) 
    Y = rr * sin(u)*sin(v) 
    Z = rr * cos(v) 
    return X, Y, Z

def prismCoord(s, height=1.0, depth=1.0, resolution=100):
    # Create a grid of points
    zz = linspace(0, height, resolution)
    xx = linspace(-s/2, s/2, resolution)
    yy = linspace(-depth/2, depth/2, resolution)
    
    X, Y = meshgrid(xx, yy)
    Z = zeros_like(X)
    
    # Top face
    top_face = (X, Y, Z + height)
    
    # Bottom face
    bottom_face = (X, Y, Z)
    
    # Side faces
    side_faces = []
    for i in range(resolution):
        side_faces.append((X[i, :], full_like(Y[i, :], Y[i, 0]), Z[i, :] + zz))
      
    X = concatenate([top_face[0].flatten(), bottom_face[0].flatten()] + [face[0].flatten() for face in side_faces])
    Y = concatenate([top_face[1].flatten(), bottom_face[1].flatten()] + [face[1].flatten() for face in side_faces])
    Z = concatenate([top_face[2].flatten(), bottom_face[2].flatten()] + [face[2].flatten() for face in side_faces])
        
    return X, Y, Z

def plotSphere(X,Y,Z, clrMatrix=0, ccLim=(0,1),colorMap="Viridis"):
   
    tt=[Surface(x=X,
                             y=Y, 
                             z=Z, 
                             surfacecolor=clrMatrix,
                             cmin=ccLim[0],
                             cmax=ccLim[1],
                             colorscale=colorMap
                             )]
    return tt


def plotPrism(X,Y,Z, clrMatrix=0, ccLim=(0,1),colorMap="Viridis"):
    tt=[Surface(x=X,
                             y=Y, 
                             z=Z, 
                             surfacecolor=clrMatrix,
                             cmin=ccLim[0],
                             cmax=ccLim[1],
                             colorscale=colorMap
                             )]
    return tt

def circleCoord(r):
    # Create a grid of points 
    resolution = len(r)
    theta = linspace(0, 2 * pi, resolution) 
    T, R = meshgrid(theta, r) # Calculate coordinates 
    X = R * cos(T) 
    Y = zeros_like(X)
    Z = R * sin(T) 
    return X, Y, Z

def rectCoord(s,h=1.0):
    # Create a grid of points 
    resolution = len(s)
    zz = linspace(-h/2, h/2, resolution)
    xx = concatenate((-flipud(s), s[1:]))
    Z, X = meshgrid(zz,xx)
    Y = zeros_like(X)
    return X, Y, Z

def plotCircle(X,Y,Z, clrMatrix=0, ccLim=(0,1),colorMap="Viridis"):
        
    if isinstance(clrMatrix, int) and clrMatrix == 0:
            clrMatrix = Z
    
    tt=[Surface(x=X,
                             y=Y, 
                             z=Z, 
                             surfacecolor=clrMatrix,
                             cmin=ccLim[0],
                             cmax=ccLim[1],
                             colorscale=colorMap
                             )]
    return tt


def plotRect(X,Y,Z, clrMatrix=0, ccLim=(0,1),colorMap="Viridis"):
        
    
    if prod(clrMatrix.shape)==1: 
        clrMatrix=ones(X.shape)*clrMatrix

    tt=[Surface(x=X,
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

    X = array([[v[0] for v in face] for face in faces])
    Y = array([[v[1] for v in face] for face in faces])
    Z = array([[v[2] for v in face] for face in faces])

    tt = plotRect(X, Y, Z, clrMatrix, ccLim, colorMap)
    return tt


   