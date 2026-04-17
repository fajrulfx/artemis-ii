#!/usr/bin/env python3
"""
Interactive 3D trajectory visualization with velocity coloring for Artemis II mission.

Parses JPL Horizons ephemeris data and creates an interactive matplotlib plot
showing the spacecraft trajectory colored by velocity magnitude.
"""

import re
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


@dataclass
class EphemerisData:
    """Container for parsed ephemeris data."""
    name: str
    timestamps: list[datetime]
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    vz: np.ndarray

    @property
    def velocity_magnitude(self) -> np.ndarray:
        """Calculate velocity magnitude at each point."""
        return np.sqrt(self.vx**2 + self.vy**2 + self.vz**2)


def parse_horizons_ephemeris(filepath: Path) -> EphemerisData:
    """
    Parse JPL Horizons ephemeris text file.
    
    Extracts position and velocity data between $$SOE and $$EOE markers.
    """
    content = filepath.read_text()
    
    name_match = re.search(r'Target body name:\s*(.+?)\s*\{', content)
    name = name_match.group(1).strip() if name_match else filepath.stem
    
    soe_idx = content.find('$$SOE')
    eoe_idx = content.find('$$EOE')
    
    if soe_idx == -1 or eoe_idx == -1:
        raise ValueError(f"Could not find $$SOE/$$EOE markers in {filepath}")
    
    data_section = content[soe_idx + 5:eoe_idx].strip()
    lines = [line.strip() for line in data_section.split('\n') if line.strip()]
    
    timestamps = []
    x, y, z = [], [], []
    vx, vy, vz = [], [], []
    
    for line in lines:
        parts = line.split(',')
        if len(parts) < 8:
            continue
        
        date_str = parts[1].strip()
        date_match = re.search(r'(\d{4})-(\w{3})-(\d{2})\s+(\d{2}):(\d{2}):(\d+\.?\d*)', date_str)
        if date_match:
            year = int(date_match.group(1))
            month_str = date_match.group(2)
            day = int(date_match.group(3))
            hour = int(date_match.group(4))
            minute = int(date_match.group(5))
            second = float(date_match.group(6))
            
            month_map = {
                'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
            }
            month = month_map.get(month_str, 1)
            
            try:
                ts = datetime(year, month, day, hour, minute, int(second))
                timestamps.append(ts)
            except ValueError:
                continue
        
        try:
            x.append(float(parts[2].strip()))
            y.append(float(parts[3].strip()))
            z.append(float(parts[4].strip()))
            vx.append(float(parts[5].strip()))
            vy.append(float(parts[6].strip()))
            vz.append(float(parts[7].strip()))
        except (ValueError, IndexError):
            if timestamps:
                timestamps.pop()
            continue
    
    return EphemerisData(
        name=name,
        timestamps=timestamps,
        x=np.array(x),
        y=np.array(y),
        z=np.array(z),
        vx=np.array(vx),
        vy=np.array(vy),
        vz=np.array(vz)
    )


def plot_trajectory_velocity(artemis: EphemerisData, moon: EphemerisData):
    """Create an interactive 3D plot with velocity-colored trajectory."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    def on_scroll(event):
        """Zoom with scroll wheel."""
        if event.inaxes != ax:
            return
        scale = 1.15 if event.button == 'down' else 1/1.15
        ax.set_xlim([x * scale for x in ax.get_xlim()])
        ax.set_ylim([y * scale for y in ax.get_ylim()])
        ax.set_zlim([z * scale for z in ax.get_zlim()])
        fig.canvas.draw_idle()
    
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    
    velocity = artemis.velocity_magnitude
    
    norm = Normalize(vmin=velocity.min(), vmax=velocity.max())
    cmap = plt.cm.plasma
    
    # Pre-compute colors for each segment
    colors = cmap(norm(velocity[:-1]))
    
    points = np.array([artemis.x, artemis.y, artemis.z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    lc = Line3DCollection(segments, colors=colors, linewidth=2)
    ax.add_collection3d(lc)
    
    ax.plot(
        moon.x, moon.y, moon.z,
        'gray', linewidth=2, label='Moon', alpha=0.5
    )
    
    earth_radius = 6371
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    xe = earth_radius * np.outer(np.cos(u), np.sin(v))
    ye = earth_radius * np.outer(np.sin(u), np.sin(v))
    ze = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(xe, ye, ze, color='royalblue', alpha=0.7)
    
    moon_radius = 1737
    moon_pos_idx = len(moon.x) // 2
    moon_center = (moon.x[moon_pos_idx], moon.y[moon_pos_idx], moon.z[moon_pos_idx])
    xm = moon_radius * np.outer(np.cos(u), np.sin(v)) + moon_center[0]
    ym = moon_radius * np.outer(np.sin(u), np.sin(v)) + moon_center[1]
    zm = moon_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + moon_center[2]
    ax.plot_surface(xm, ym, zm, color='silver', alpha=0.8)
    
    ax.scatter(artemis.x[0], artemis.y[0], artemis.z[0], 
               color='lime', s=100, marker='^', label='Start', zorder=5, edgecolors='black')
    ax.scatter(artemis.x[-1], artemis.y[-1], artemis.z[-1], 
               color='red', s=100, marker='v', label='End', zorder=5, edgecolors='black')
    
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, aspect=20, pad=0.1)
    cbar.set_label('Velocity (km/s)', fontsize=11)
    
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('Artemis II Trajectory - Velocity Colored\n(Earth-Centered Ecliptic J2000)')
    
    ax.legend(loc='upper left')
    
    max_range = max(
        np.abs(moon.x).max(), np.abs(moon.y).max(), np.abs(moon.z).max(),
        np.abs(artemis.x).max(), np.abs(artemis.y).max(), np.abs(artemis.z).max()
    ) * 1.1
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    plt.tight_layout()
    return fig, ax


def main():
    data_dir = Path(__file__).parent.parent / 'data'
    
    artemis_file = data_dir / 'artemis_ephemeris.txt'
    moon_file = data_dir / 'moon_ephemeris.txt'
    
    if not artemis_file.exists():
        print(f"Error: {artemis_file} not found")
        return
    if not moon_file.exists():
        print(f"Error: {moon_file} not found")
        return
    
    print("Parsing ephemeris data...")
    artemis = parse_horizons_ephemeris(artemis_file)
    moon = parse_horizons_ephemeris(moon_file)
    
    velocity = artemis.velocity_magnitude
    print(f"Loaded {len(artemis.timestamps)} data points")
    print(f"Velocity range: {velocity.min():.2f} - {velocity.max():.2f} km/s")
    
    plot_trajectory_velocity(artemis, moon)
    plt.show()


if __name__ == '__main__':
    main()
