#!/usr/bin/env python3
"""
Interactive 3D Artemis II trajectory relative to Earth only (no Moon).

Parses JPL Horizons ephemeris and plots the spacecraft path with Earth;
Moon data is not loaded or shown.
"""

import re
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt


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
        date_match = re.search(
            r'(\d{4})-(\w{3})-(\d{2})\s+(\d{2}):(\d{2}):(\d+\.?\d*)', date_str
        )
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


def plot_artemis_earth_3d(artemis: EphemerisData):
    """Interactive 3D plot: Artemis trajectory and Earth only."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(
        artemis.x, artemis.y, artemis.z,
        'b-', linewidth=1.5, label='Artemis II', alpha=0.8
    )

    earth_radius = 6371
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    xe = earth_radius * np.outer(np.cos(u), np.sin(v))
    ye = earth_radius * np.outer(np.sin(u), np.sin(v))
    ze = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(xe, ye, ze, color='royalblue', alpha=0.7)

    ax.scatter(artemis.x[0], artemis.y[0], artemis.z[0],
               color='green', s=100, marker='^', label='Start', zorder=5)
    ax.scatter(artemis.x[-1], artemis.y[-1], artemis.z[-1],
               color='red', s=100, marker='v', label='End', zorder=5)

    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('Artemis II Trajectory — Earth only (Ecliptic J2000)')

    ax.legend(loc='upper left')

    max_range = max(
        np.abs(artemis.x).max(),
        np.abs(artemis.y).max(),
        np.abs(artemis.z).max(),
    ) * 1.1
    max_range = max(max_range, earth_radius * 1.5)

    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])

    plt.tight_layout()
    return fig, ax


def main():
    data_dir = Path(__file__).parent.parent / 'data'
    artemis_file = data_dir / 'artemis_ephemeris.txt'

    if not artemis_file.exists():
        print(f"Error: {artemis_file} not found")
        return

    print("Parsing ephemeris data...")
    artemis = parse_horizons_ephemeris(artemis_file)
    print(f"Loaded {len(artemis.timestamps)} data points")

    plot_artemis_earth_3d(artemis)
    plt.show()


if __name__ == '__main__':
    main()
