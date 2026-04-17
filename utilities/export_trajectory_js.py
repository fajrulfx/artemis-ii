#!/usr/bin/env python3
"""
Export JPL Horizons ephemeris data to JavaScript module for Three.js visualization.

Parses the artemis_ephemeris.txt and moon_ephemeris.txt files and outputs
JavaScript module with trajectory data.
"""

import re
from pathlib import Path
from datetime import datetime


def parse_horizons_ephemeris(filepath: Path) -> list[dict]:
    """
    Parse JPL Horizons ephemeris text file.
    
    Returns list of dicts with timestamp and position data.
    """
    content = filepath.read_text()
    
    soe_idx = content.find('$$SOE')
    eoe_idx = content.find('$$EOE')
    
    if soe_idx == -1 or eoe_idx == -1:
        raise ValueError(f"Could not find $$SOE/$$EOE markers in {filepath}")
    
    data_section = content[soe_idx + 5:eoe_idx].strip()
    lines = [line.strip() for line in data_section.split('\n') if line.strip()]
    
    data_points = []
    
    for line in lines:
        parts = line.split(',')
        if len(parts) < 8:
            continue
        
        date_str = parts[1].strip()
        date_match = re.search(r'(\d{4})-(\w{3})-(\d{2})\s+(\d{2}):(\d{2}):(\d+\.?\d*)', date_str)
        if not date_match:
            continue
            
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
        except ValueError:
            continue
        
        try:
            x = float(parts[2].strip())
            y = float(parts[3].strip())
            z = float(parts[4].strip())
            vx = float(parts[5].strip())
            vy = float(parts[6].strip())
            vz = float(parts[7].strip())
        except (ValueError, IndexError):
            continue
        
        data_points.append({
            'timestamp': ts,
            'x': x,
            'y': y,
            'z': z,
            'vx': vx,
            'vy': vy,
            'vz': vz
        })
    
    return data_points


def downsample(data: list[dict], interval_minutes: int = 10) -> list[dict]:
    """
    Downsample data to specified interval.
    Using 10-minute intervals to capture close Earth approaches accurately.
    """
    if not data:
        return []
    
    result = [data[0]]
    last_time = data[0]['timestamp']
    
    for point in data[1:]:
        diff = (point['timestamp'] - last_time).total_seconds() / 60
        if diff >= interval_minutes:
            result.append(point)
            last_time = point['timestamp']
    
    if result[-1] != data[-1]:
        result.append(data[-1])
    
    return result


def format_js_array(data: list[dict], name: str, export: bool = True) -> str:
    """
    Format data as JavaScript array declaration.
    """
    prefix = "export " if export else ""
    lines = [f"{prefix}const {name} = ["]
    
    for i, point in enumerate(data):
        x = round(point['x'])
        y = round(point['y'])
        z = round(point['z'])
        
        if i < len(data) - 1:
            lines.append(f"  [{x},{y},{z}],")
        else:
            lines.append(f"  [{x},{y},{z}]")
    
    lines.append("];")
    return '\n'.join(lines)


def main():
    data_dir = Path(__file__).parent.parent / 'data'
    output_dir = Path(__file__).parent.parent / 'web'
    
    output_dir.mkdir(exist_ok=True)
    
    artemis_file = data_dir / 'artemis_ephemeris.txt'
    moon_file = data_dir / 'moon_ephemeris.txt'
    
    print("Parsing Artemis ephemeris...")
    artemis_data = parse_horizons_ephemeris(artemis_file)
    print(f"  Found {len(artemis_data)} data points")
    
    print("Parsing Moon ephemeris...")
    moon_data = parse_horizons_ephemeris(moon_file)
    print(f"  Found {len(moon_data)} data points")
    
    # Downsample to 10-minute intervals to capture close Earth approaches accurately
    print("\nDownsampling to 10-minute intervals...")
    artemis_sampled = downsample(artemis_data, interval_minutes=10)
    moon_sampled = downsample(moon_data, interval_minutes=10)
    print(f"  Artemis: {len(artemis_sampled)} points")
    print(f"  Moon: {len(moon_sampled)} points")
    
    # Calculate trajectory timing info
    start_time = artemis_sampled[0]['timestamp']
    step_hours = 10 / 60  # 10 minutes
    
    # Generate JavaScript module
    artemis_js = format_js_array(artemis_sampled, "NASA_TRAJ")
    moon_js = format_js_array(moon_sampled, "NASA_MOON")
    
    print(f"\nArtemis data range:")
    print(f"  Start: {artemis_sampled[0]['timestamp']}")
    print(f"  End: {artemis_sampled[-1]['timestamp']}")
    
    print(f"\nMoon data range:")
    print(f"  Start: {moon_sampled[0]['timestamp']}")
    print(f"  End: {moon_sampled[-1]['timestamp']}")
    
    # Output to web/trajectory.js
    output_file = output_dir / 'trajectory.js'
    with open(output_file, 'w') as f:
        f.write("// Artemis II trajectory data from JPL Horizons\n")
        f.write("// Earth-centered Ecliptic J2000 coordinates (km)\n")
        f.write(f"// Artemis: {len(artemis_sampled)} points, 10-minute intervals\n")
        f.write(f"// Moon: {len(moon_sampled)} points, 10-minute intervals\n")
        f.write(f"// Data range: {start_time} to {artemis_sampled[-1]['timestamp']}\n\n")
        f.write(artemis_js)
        f.write("\n\n")
        f.write(moon_js)
        f.write("\n\n")
        # Find TLI index (minimum altitude = perigee just before TLI)
        import math
        min_alt = float('inf')
        tli_index = 0
        for i, point in enumerate(artemis_sampled):
            dist = math.sqrt(point['x']**2 + point['y']**2 + point['z']**2)
            alt = dist - 6371  # Earth radius
            if alt < min_alt:
                min_alt = alt
                tli_index = i
        
        # Calculate timing from actual data
        from datetime import datetime
        launch_time = datetime(2026, 4, 1, 22, 35, 12)
        data_start_time = artemis_sampled[0]['timestamp']
        data_end_time = artemis_sampled[-1]['timestamp']
        traj_start_days = (data_start_time - launch_time).total_seconds() / 86400
        traj_end_days = (data_end_time - launch_time).total_seconds() / 86400
        
        f.write(f"// Trajectory timing constants\n")
        f.write(f"export const TRAJ_START_DAY = {traj_start_days:.6f}; // Days after launch (data starts at ICPS separation)\n")
        f.write(f"export const TRAJ_END_DAY = {traj_end_days:.6f}; // Days after launch (data ends)\n")
        f.write(f"export const TRAJ_STEP_DAYS = {step_hours / 24}; // {step_hours} hours between points\n")
        f.write(f"export const TRAJ_N = {len(artemis_sampled)};\n")
        f.write(f"export const TLI_INDEX = {tli_index + 1}; // Index just after perigee (TLI burn)\n")
    
    print(f"\nOutput written to: {output_file}")


if __name__ == '__main__':
    main()
