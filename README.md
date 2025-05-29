
# jupyterPTangles

**Interactive notebook for visualizing and analyzing the characteristic angles of a parabolic trough solar concentrator.**

## Overview

`jupytrPTangles.ipynb` is a Jupyter Notebook that provides an interactive environment to compute and visualize the sun's position and the key angles (zenith, azimuth, incidence, and tracking) relevant to parabolic trough solar collectors. The notebook uses Plotly for 3D visualization and ipywidgets for interactive controls.

## Features

- **3D Visualization:** View the parabolic trough, sun vector, and characteristic angles in an interactive Plotly plot.
- **Daily Simulation:** Adjust date, time, latitude, longitude, axis orientation, and time zone with sliders.
- **Animated Sun Path:** Use the "Daily Play" button to animate the sun's movement and see how angles change throughout the day.
- **Angle Plots:** Graphs of zenith, azimuth, incidence, and tracking angles as functions of time.
- **Custom Location and Orientation:** Easily set geographic and system parameters.

## Requirements

- Python 3.7+
- Jupyter Notebook or JupyterLab
- The following Python packages:
  - numpy
  - scipy
  - plotly
  - ipywidgets
  - sunposition

Install all dependencies with:
```sh
pip install -r requirements.txt
```

## Usage

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/CSPangles.git
   cd CSPangles
   ```

2. **Launch Jupyter Notebook:**
   ```sh
   jupyter notebook
   ```
   Open `jupytrPTangles.ipynb` in your browser.

3. **Interact:**
   - Use the sliders to set date, time, latitude, longitude, axis orientation, and time zone.
   - Click "Daily Play" to animate the sun's path.
   - The 3D plot and angle graphs update in real time.


## Notes

- For best experience, use JupyterLab or a modern Jupyter Notebook interface with widget support.
- If running in Google Colab, set `colabRun=True` at the top of the notebook.
- The notebook is intended for educational and research purposes.
- A note about the code writing: This was my first significant effort in writing in Jupyter/Python. So it is certainly full of syntax that is not “best practice”. 

## License

MIT License

