# %%
import plotly.graph_objs as go
import numpy as np
from ipywidgets import interact
#First we'll create an empty figure, and add an empty scatter trace to it.
xs=np.linspace(0, 6, 100)
fig = go.Figure()
x=xs
y=np.sin(x)
fig.add_scatter(x=x,y=y)
import plotly.io as pio

#Then, write an update function that inputs the frequency factor (a) and phase factor (b) and sets the x and y properties of the scatter trace. This function is decorated with the interact decorator from the ipywidgets package. The decorator parameters are used to specify the ranges of parameters that we want to sweep over. See http://ipywidgets.readthedocs.io/en/latest/examples/Using%20Interact.html for more details.

@interact(a=(1.0, 4.0, 0.01), b=(0, 10.0, 0.01), color=['red', 'green', 'blue'])
def update(a=3.6, b=4.3, color='blue'):
    fig.data[0].x = xs
    fig.data[0].y = np.sin(a * xs - b)
    fig.data[0].line.color = color
    fig.update_layout(
        title=f'Sine wave with a={a}, b={b}',
        xaxis_title='x',
        yaxis_title='y',
        showlegend=False
    )
    fig.show()
        

# %%
