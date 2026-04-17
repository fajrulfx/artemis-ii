#!/usr/bin/env python3
"""
3D trajectory visualization with detected maneuvers highlighted.
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

GM_EARTH = 398600.4418
GM_MOON = 4902.8


@dataclass
class EphemerisData:
    name: str
    timestamps: list[datetime]
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    vz: np.ndarray


def parse_horizons_ephemeris(filepath: Path) -> EphemerisData:
    content = filepath.read_text()
    name_match = re.search(r'Target body name:\s*(.+?)\s*\{', content)
    name = name_match.group(1).strip() if name_match else filepath.stem
    
    soe_idx = content.find('$$SOE')
    eoe_idx = content.find('$$EOE')
    if soe_idx == -1 or eoe_idx == -1:
        raise ValueError(f"Could not find $$SOE/$$EOE markers in {filepath}")
    
    data_section = content[soe_idx + 5:eoe_idx].strip()
    lines = [line.strip() for line in data_section.split('\n') if line.strip()]
    
    timestamps, x, y, z, vx, vy, vz = [], [], [], [], [], [], []
    
    for line in lines:
        parts = line.split(',')
        if len(parts) < 8:
            continue
        date_str = parts[1].strip()
        date_match = re.search(r'(\d{4})-(\w{3})-(\d{2})\s+(\d{2}):(\d{2}):(\d+\.?\d*)', date_str)
        if date_match:
            year, month_str, day = int(date_match.group(1)), date_match.group(2), int(date_match.group(3))
            hour, minute, second = int(date_match.group(4)), int(date_match.group(5)), float(date_match.group(6))
            month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
            try:
                timestamps.append(datetime(year, month_map.get(month_str, 1), day, hour, minute, int(second)))
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
    
    return EphemerisData(name, timestamps, np.array(x), np.array(y), np.array(z),
                         np.array(vx), np.array(vy), np.array(vz))


def compute_thrust_acceleration(artemis: EphemerisData, moon: EphemerisData):
    """Compute thrust (residual) acceleration magnitude in m/s²."""
    # Observed acceleration
    dt = np.array([(artemis.timestamps[i+1] - artemis.timestamps[i]).total_seconds() 
                   for i in range(len(artemis.timestamps) - 1)])
    ax_obs = np.concatenate([[0], np.diff(artemis.vx) / dt])
    ay_obs = np.concatenate([[0], np.diff(artemis.vy) / dt])
    az_obs = np.concatenate([[0], np.diff(artemis.vz) / dt])
    
    # Gravitational acceleration from Earth
    r_earth = np.sqrt(artemis.x**2 + artemis.y**2 + artemis.z**2)
    a_earth_mag = GM_EARTH / r_earth**2
    ax_earth = -a_earth_mag * artemis.x / r_earth
    ay_earth = -a_earth_mag * artemis.y / r_earth
    az_earth = -a_earth_mag * artemis.z / r_earth
    
    # Gravitational acceleration from Moon
    dx_moon = artemis.x - moon.x
    dy_moon = artemis.y - moon.y
    dz_moon = artemis.z - moon.z
    r_moon = np.sqrt(dx_moon**2 + dy_moon**2 + dz_moon**2)
    a_moon_mag = GM_MOON / r_moon**2
    ax_moon = -a_moon_mag * dx_moon / r_moon
    ay_moon = -a_moon_mag * dy_moon / r_moon
    az_moon = -a_moon_mag * dz_moon / r_moon
    
    # Thrust = Observed - Gravitational
    ax_thrust = ax_obs - (ax_earth + ax_moon)
    ay_thrust = ay_obs - (ay_earth + ay_moon)
    az_thrust = az_obs - (az_earth + az_moon)
    
    return np.sqrt(ax_thrust**2 + ay_thrust**2 + az_thrust**2) * 1000  # m/s²


def plot_trajectory_with_maneuvers(artemis: EphemerisData, moon: EphemerisData):
    """Create 3D plot with maneuvers highlighted."""
    fig = plt.figure(figsize=(14, 10), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')
    
    def on_scroll(event):
        if event.inaxes != ax:
            return
        scale = 1.15 if event.button == 'down' else 1/1.15
        ax.set_xlim([v * scale for v in ax.get_xlim()])
        ax.set_ylim([v * scale for v in ax.get_ylim()])
        ax.set_zlim([v * scale for v in ax.get_zlim()])
        fig.canvas.draw_idle()
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    
    # Compute thrust acceleration
    thrust = compute_thrust_acceleration(artemis, moon)
    threshold = 0.1  # m/s²
    
    # Base trajectory in white/gray
    ax.plot(artemis.x, artemis.y, artemis.z, color='#4488ff', linewidth=1.5, alpha=0.6, label='Trajectory')
    
    # Overlay maneuver segments in bright color
    maneuver_mask = thrust > threshold
    
    # Find contiguous maneuver regions and plot them
    in_maneuver = False
    start_idx = 0
    for i in range(len(maneuver_mask)):
        if maneuver_mask[i] and not in_maneuver:
            start_idx = max(0, i - 1)
            in_maneuver = True
        elif not maneuver_mask[i] and in_maneuver:
            end_idx = min(len(maneuver_mask), i + 1)
            ax.plot(artemis.x[start_idx:end_idx], artemis.y[start_idx:end_idx], artemis.z[start_idx:end_idx],
                    color='#ff4444', linewidth=4, alpha=0.9)
            in_maneuver = False
    
    # Moon trajectory
    ax.plot(moon.x, moon.y, moon.z, color='#666666', linewidth=1.5, alpha=0.5, label='Moon orbit')
    
    # Earth sphere
    earth_radius = 6371
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    xe = earth_radius * np.outer(np.cos(u), np.sin(v))
    ye = earth_radius * np.outer(np.sin(u), np.sin(v))
    ze = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(xe, ye, ze, color='#2266cc', alpha=0.9)
    
    # Moon sphere
    moon_radius = 1737
    moon_pos_idx = len(moon.x) // 2
    moon_center = (moon.x[moon_pos_idx], moon.y[moon_pos_idx], moon.z[moon_pos_idx])
    xm = moon_radius * np.outer(np.cos(u), np.sin(v)) + moon_center[0]
    ym = moon_radius * np.outer(np.sin(u), np.sin(v)) + moon_center[1]
    zm = moon_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + moon_center[2]
    ax.plot_surface(xm, ym, zm, color='#aaaaaa', alpha=0.9)
    
    # Start/End markers
    ax.scatter(artemis.x[0], artemis.y[0], artemis.z[0], 
               color='#00ff00', s=120, marker='^', label='Launch', zorder=10, edgecolors='white', linewidth=1)
    ax.scatter(artemis.x[-1], artemis.y[-1], artemis.z[-1], 
               color='#00ffff', s=120, marker='s', label='Splashdown', zorder=10, edgecolors='white', linewidth=1)
    
    # Find major maneuver peaks
    def find_peaks_simple(arr, min_height=0.5, min_distance=50):
        peaks = []
        for i in range(min_distance, len(arr) - min_distance):
            if arr[i] >= min_height:
                if arr[i] == max(arr[max(0, i-min_distance):min(len(arr), i+min_distance+1)]):
                    if not peaks or i - peaks[-1] >= min_distance:
                        peaks.append(i)
        return peaks
    
    peaks = find_peaks_simple(thrust, min_height=0.5, min_distance=50)
    maneuver_names = ['TLI', 'Lunar Flyby', 'Correction', 'Entry']
    
    for i, peak_idx in enumerate(peaks[:4]):
        name = maneuver_names[i] if i < len(maneuver_names) else f'Burn {i+1}'
        # Marker
        ax.scatter(artemis.x[peak_idx], artemis.y[peak_idx], artemis.z[peak_idx],
                   color='#ffff00', s=250, marker='*', zorder=20, edgecolors='#ff6600', linewidth=1.5)
    
    # Style axes
    ax.set_xlabel('X (km)', color='white')
    ax.set_ylabel('Y (km)', color='white')
    ax.set_zlabel('Z (km)', color='white')
    ax.tick_params(colors='white')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')
    ax.grid(True, alpha=0.2, color='gray')
    
    ax.set_title('Artemis II - Maneuvers Detected\nBlue = coast, Red = thrust, Stars = major burns', 
                 color='white', fontsize=12, pad=10)
    
    # Legend
    legend = ax.legend(loc='upper left', fontsize=9, facecolor='#222222', edgecolor='gray')
    for text in legend.get_texts():
        text.set_color('white')
    
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
    
    if not artemis_file.exists() or not moon_file.exists():
        print("Error: Ephemeris files not found")
        return
    
    print("Parsing ephemeris data...")
    artemis = parse_horizons_ephemeris(artemis_file)
    moon = parse_horizons_ephemeris(moon_file)
    
    thrust = compute_thrust_acceleration(artemis, moon)
    print(f"Loaded {len(artemis.timestamps)} points")
    print(f"Thrust range: {thrust.min():.4f} - {thrust.max():.4f} m/s²")
    print(f"Points with thrust > 0.1 m/s²: {(thrust > 0.1).sum()}")
    
    plot_trajectory_with_maneuvers(artemis, moon)
    plt.show()


if __name__ == '__main__':
    main()
