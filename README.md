# Debris Detection Simulation

This project simulates the detection of debris "solitons" (modeled as expanding conical shells) by a satellite in orbit. It includes tools for orbital propagation, debris cloud modeling, and detection logic.

## Project Structure

```
debris_detection/
├── main.py           # Main entry point for the simulation
├── src/
│   ├── orbit.py      # Orbital mechanics and propagation
│   ├── soliton.py    # Soliton (debris cloud) modeling
│   ├── satellite.py  # Satellite (detector) modeling
│   ├── monte_carlo.py# Monte Carlo simulation tools
│   ├── constants.py  # Physical constants
│   └── possible_region.py # Visualization and analysis of soliton regions
├── test/             # Unit tests
├── plots/            # Output plots
└── archive/          # Archived files
```

## Features

- **Orbital Propagation**: Propagates satellite and debris orbits using `scipy.integrate`. Supports J2 perturbation.
- **Soliton Modeling**: Models debris clouds as expanding conical shells intersected with spherical shells.
- **Detection Logic**: Determines if a satellite is within a debris cloud (soliton) at a given time.
- **Visualization**: Includes 3D plotting capabilities for orbits and solitons.

## Installation

1.  **Clone the repository**.
2.  **Install dependencies**:
    This project requires Python 3 and the following scientific computing libraries:
    *   `numpy`
    *   `scipy`
    *   `matplotlib`

    You can install them via pip:

    ```bash
    pip install numpy scipy matplotlib
    ```

## Usage

To run the main simulation script:

```bash
python main.py
```

The `main.py` script currently sets up a satellite and a debris object, creates a soliton from the debris, and performs detection checks. You can modify the Orbital Elements (OE) in `main.py` to test different scenarios.

## Modules

### `src.orbit`
Handles orbital mechanics, specifically:
- Conversions between Orbital Elements (OE) and State Vectors (SV).
- Orbit propagation using `odeint` or `solve_ivp`.
- Calculation of J2 perturbation effects (RAAN and Argument of Perigee precession).

### `src.soliton`
Defines the `Soliton` class, which represents a debris cloud.
- Models the cloud as an expanding cone.
- Checks if a point (satellite position) is within the soliton shell.
- Provides visualization methods for the soliton cone and shell.

### `src.satellite`
Models the observing satellite.
- Manages state vectors and coordinate transformations (ECI to Body Frame).
- Contains logic to detect if the satellite is inside a soliton.

### `src.monte_carlo`
Contains tools for generating debris samples for statistical analysis.

### `src.possible_region`
Provides visualization and analysis tools to determine the possible regions of a soliton's center based on sensor detections.
- Calculates the intersection of possible soliton regions for multiple sensor points.
- Implements constraints for spherical shells and conical shapes.
- Removes regions blocked by the satellite's wake.
- Visualizes the resulting valid regions using 3D voxel plots.
