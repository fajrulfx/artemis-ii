#!/usr/bin/env python3
"""
Detect thrust maneuvers in Artemis II trajectory by comparing observed
acceleration against expected gravitational acceleration.

Thrust = Observed acceleration - Gravitational acceleration (Earth + Moon)
"""

import re
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

# Gravitational parameters (km³/s²)
GM_EARTH = 398600.4418
GM_MOON = 4902.8


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
    """Parse JPL Horizons ephemeris text file."""
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


def compute_gravitational_acceleration(artemis: EphemerisData, moon: EphemerisData):
    """
    Compute expected gravitational acceleration from Earth and Moon.
    Returns acceleration vectors (ax, ay, az) in km/s².
    """
    # Distance vectors from spacecraft to Earth (Earth is at origin)
    r_earth = np.sqrt(artemis.x**2 + artemis.y**2 + artemis.z**2)
    
    # Acceleration toward Earth: -GM/r² in the direction of -r_hat
    a_earth_mag = GM_EARTH / r_earth**2
    ax_earth = -a_earth_mag * artemis.x / r_earth
    ay_earth = -a_earth_mag * artemis.y / r_earth
    az_earth = -a_earth_mag * artemis.z / r_earth
    
    # Distance vectors from spacecraft to Moon
    dx_moon = artemis.x - moon.x
    dy_moon = artemis.y - moon.y
    dz_moon = artemis.z - moon.z
    r_moon = np.sqrt(dx_moon**2 + dy_moon**2 + dz_moon**2)
    
    # Acceleration toward Moon
    a_moon_mag = GM_MOON / r_moon**2
    ax_moon = -a_moon_mag * dx_moon / r_moon
    ay_moon = -a_moon_mag * dy_moon / r_moon
    az_moon = -a_moon_mag * dz_moon / r_moon
    
    # Total gravitational acceleration
    ax_grav = ax_earth + ax_moon
    ay_grav = ay_earth + ay_moon
    az_grav = az_earth + az_moon
    
    return ax_grav, ay_grav, az_grav


def compute_observed_acceleration(artemis: EphemerisData):
    """
    Compute observed acceleration from velocity derivative.
    Returns acceleration vectors (ax, ay, az) in km/s².
    """
    dt = np.array([(artemis.timestamps[i+1] - artemis.timestamps[i]).total_seconds() 
                   for i in range(len(artemis.timestamps) - 1)])
    
    ax = np.diff(artemis.vx) / dt
    ay = np.diff(artemis.vy) / dt
    az = np.diff(artemis.vz) / dt
    
    # Pad to match original length (use first value for index 0)
    ax = np.concatenate([[ax[0]], ax])
    ay = np.concatenate([[ay[0]], ay])
    az = np.concatenate([[az[0]], az])
    
    return ax, ay, az


def detect_maneuvers(artemis: EphemerisData, moon: EphemerisData, threshold_ms2: float = 0.1):
    """
    Detect maneuvers by finding where thrust acceleration exceeds threshold.
    
    Args:
        threshold_ms2: Minimum thrust acceleration to consider a maneuver (m/s²)
    
    Returns:
        Dictionary with analysis results
    """
    # Compute accelerations
    ax_obs, ay_obs, az_obs = compute_observed_acceleration(artemis)
    ax_grav, ay_grav, az_grav = compute_gravitational_acceleration(artemis, moon)
    
    # Thrust = Observed - Gravitational
    ax_thrust = ax_obs - ax_grav
    ay_thrust = ay_obs - ay_grav
    az_thrust = az_obs - az_grav
    
    # Magnitudes (convert to m/s² for readability)
    a_obs_mag = np.sqrt(ax_obs**2 + ay_obs**2 + az_obs**2) * 1000
    a_grav_mag = np.sqrt(ax_grav**2 + ay_grav**2 + az_grav**2) * 1000
    a_thrust_mag = np.sqrt(ax_thrust**2 + ay_thrust**2 + az_thrust**2) * 1000
    
    # Find maneuver candidates
    threshold_kms2 = threshold_ms2 / 1000
    maneuver_mask = a_thrust_mag > threshold_ms2
    
    # Group consecutive maneuver points
    maneuvers = []
    in_maneuver = False
    start_idx = 0
    
    for i, is_thrust in enumerate(maneuver_mask):
        if is_thrust and not in_maneuver:
            in_maneuver = True
            start_idx = i
        elif not is_thrust and in_maneuver:
            in_maneuver = False
            maneuvers.append({
                'start_idx': start_idx,
                'end_idx': i - 1,
                'start_time': artemis.timestamps[start_idx],
                'end_time': artemis.timestamps[i - 1],
                'peak_thrust': a_thrust_mag[start_idx:i].max(),
                'peak_idx': start_idx + np.argmax(a_thrust_mag[start_idx:i])
            })
    
    if in_maneuver:
        maneuvers.append({
            'start_idx': start_idx,
            'end_idx': len(maneuver_mask) - 1,
            'start_time': artemis.timestamps[start_idx],
            'end_time': artemis.timestamps[-1],
            'peak_thrust': a_thrust_mag[start_idx:].max(),
            'peak_idx': start_idx + np.argmax(a_thrust_mag[start_idx:])
        })
    
    return {
        'a_obs_mag': a_obs_mag,
        'a_grav_mag': a_grav_mag,
        'a_thrust_mag': a_thrust_mag,
        'maneuvers': maneuvers,
        'maneuver_mask': maneuver_mask
    }


def plot_acceleration_analysis(artemis: EphemerisData, results: dict):
    """Plot acceleration components and detected maneuvers."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Time axis (hours from start)
    t0 = artemis.timestamps[0]
    hours = np.array([(t - t0).total_seconds() / 3600 for t in artemis.timestamps])
    
    # Plot 1: Observed vs Gravitational acceleration
    ax1 = axes[0]
    ax1.semilogy(hours, results['a_obs_mag'], 'b-', alpha=0.7, label='Observed', linewidth=0.8)
    ax1.semilogy(hours, results['a_grav_mag'], 'g-', alpha=0.7, label='Gravitational (Earth+Moon)', linewidth=0.8)
    ax1.set_ylabel('Acceleration (m/s²)')
    ax1.set_title('Acceleration Analysis - Observed vs Expected Gravitational')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Thrust (residual) acceleration
    ax2 = axes[1]
    ax2.semilogy(hours, results['a_thrust_mag'], 'r-', alpha=0.7, linewidth=0.8)
    ax2.axhline(y=0.1, color='orange', linestyle='--', label='Detection threshold (0.1 m/s²)')
    ax2.set_ylabel('Thrust Accel (m/s²)')
    ax2.set_title('Residual Acceleration (Observed - Gravitational) = Potential Thrust')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Highlight maneuvers
    for m in results['maneuvers']:
        ax2.axvspan(hours[m['start_idx']], hours[m['end_idx']], alpha=0.3, color='red')
    
    # Plot 3: Distance from Earth and Moon
    ax3 = axes[2]
    r_earth = np.sqrt(artemis.x**2 + artemis.y**2 + artemis.z**2)
    ax3.plot(hours, r_earth, 'b-', label='Distance from Earth', linewidth=1)
    ax3.axhline(y=6371, color='blue', linestyle=':', alpha=0.5, label='Earth radius')
    ax3.axhline(y=384400, color='gray', linestyle=':', alpha=0.5, label='Moon avg distance')
    ax3.set_xlabel('Time (hours from launch)')
    ax3.set_ylabel('Distance (km)')
    ax3.set_title('Distance from Earth')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


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
    print(f"Loaded {len(artemis.timestamps)} data points")
    print(f"Time span: {artemis.timestamps[0]} to {artemis.timestamps[-1]}")
    
    print("\nAnalyzing accelerations...")
    results = detect_maneuvers(artemis, moon, threshold_ms2=0.1)
    
    print(f"\n{'='*60}")
    print("MANEUVER DETECTION RESULTS")
    print(f"{'='*60}")
    print(f"Detection threshold: 0.1 m/s²")
    print(f"Number of potential maneuvers detected: {len(results['maneuvers'])}")
    
    if results['maneuvers']:
        print(f"\n{'─'*60}")
        for i, m in enumerate(results['maneuvers'], 1):
            duration = (m['end_time'] - m['start_time']).total_seconds()
            print(f"\nManeuver #{i}:")
            print(f"  Time: {m['start_time']} to {m['end_time']}")
            print(f"  Duration: {duration:.0f} seconds ({duration/60:.1f} minutes)")
            print(f"  Peak thrust acceleration: {m['peak_thrust']:.4f} m/s²")
            
            # Position at maneuver
            idx = m['peak_idx']
            r = np.sqrt(artemis.x[idx]**2 + artemis.y[idx]**2 + artemis.z[idx]**2)
            print(f"  Distance from Earth at peak: {r:.0f} km ({r/6371:.1f} Earth radii)")
    else:
        print("\nNo significant maneuvers detected above threshold.")
        print("This could mean:")
        print("  1. The trajectory is primarily gravitational (free-return)")
        print("  2. Maneuvers are below detection threshold")
        print("  3. The ephemeris time resolution is too coarse to capture impulsive burns")
    
    print(f"\n{'─'*60}")
    print("Acceleration statistics:")
    print(f"  Observed: {results['a_obs_mag'].min():.6f} - {results['a_obs_mag'].max():.6f} m/s²")
    print(f"  Gravitational: {results['a_grav_mag'].min():.6f} - {results['a_grav_mag'].max():.6f} m/s²")
    print(f"  Residual (thrust): {results['a_thrust_mag'].min():.6f} - {results['a_thrust_mag'].max():.6f} m/s²")
    
    plot_acceleration_analysis(artemis, results)
    plt.show()


if __name__ == '__main__':
    main()
