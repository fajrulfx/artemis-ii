# Artemis II Mission Tracker

Interactive 3D visualization and physics simulator for NASA's Artemis II lunar
flyby mission. Live demo: [fajrul.com/artemis-ii](https://fajrul.com/artemis-ii)

- `**web/index.html**` — Plays back the real Artemis II trajectory from JPL
Horizons data.
- `**web/simulation.html**` — Live physics sandbox: tweak velocity, altitude,
and Moon angle, and see the trajectory.

See [Physics.md](./Physics.md) for the math.

## Credits

Inspired by [Javilop's Artemis II Tracker](https://github.com/Javilop/artemis-ii-tracker).
Orion 3D model by [DeltaX on Cults3D](https://cults3d.com/).
Big thanks to Claude for doing most of the heavy lifting on the code.

## Files

```
core.py              # Reference Python simulator (Earth + Moon gravity)
data/                # JPL Horizons ephemeris files (real mission data)
utilities/           # Python scripts for parsing & plotting
web/                 # Browser visualizations (Three.js)
```

### `data/`

- `artemis_ephemeris.txt`: Real Artemis II position & velocity over time.
- `moon_ephemeris.txt`: Moon position & velocity over the same period.

### `utilities/`

- `export_trajectory_js.py`: Converts ephemeris → `web/trajectory.js`.
- `detect_maneuvers.py`: Finds engine burns by comparing observed vs. expected gravity.
- `plot_trajectory_interactive.py`: 3D plot of spacecraft + Moon + Earth.
- `plot_artemis_earth.py`: 3D plot of spacecraft + Earth only.
- `plot_acceleration.py`: Trajectory colored by acceleration.
- `plot_velocity.py`: Trajectory colored by speed.
- `plot_maneuvers.py`: Trajectory with detected burns highlighted.

### `web/`

- `index.html`: Real mission viewer.
- `simulation.html`: Interactive physics sandbox.
- `trajectory.js`: Auto-generated trajectory data.
- `orion.stl`: 3D model of the Orion spacecraft.

## Run

```bash
# View web pages
cd web && python3 -m http.server 8000
# open http://localhost:8000/index.html
# open http://localhost:8000/simulation.html

# Run Python scripts
pip install numpy scipy matplotlib
python3 core.py
```

