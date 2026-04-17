import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.collections import LineCollection
import time

# 1. CONSTANTS (The Universe Rules)
G = 6.67430e-20       # Gravitational constant (km^3 / kg s^2)
M_earth = 5.972e24    # Mass of Earth (kg)
M_moon = 7.348e22     # Mass of Moon (kg)
D_moon = 384400.0     # Average distance to the Moon (km)
R_earth = 6371.0      # Earth's average radius (km)

# NEW: Moon's Orbital Mechanics
T_moon = 27.32 * 24 * 3600      # Moon's orbital period in seconds (~27.3 days)
OMEGA_moon = 2 * np.pi / T_moon # Moon's angular velocity (radians per second)

# We start the Moon at an angle so we "lead the target".
# 126 degrees is the sweet spot for a typical Artemis transfer.
theta_0 = np.radians(126)       

# 2. THE PHYSICS ENGINE (Now with a moving target!)
def gravity_forces(t, state):
    x, y, vx, vy = state
    
    # NEW: Calculate exactly where the Moon is at this specific second (t)
    x_moon = D_moon * np.cos(OMEGA_moon * t + theta_0)
    y_moon = D_moon * np.sin(OMEGA_moon * t + theta_0)
    
    # Distances
    r_earth = np.sqrt(x**2 + y**2)
    r_moon = np.sqrt((x - x_moon)**2 + (y - y_moon)**2)
    
    # Accelerations from Earth
    ax_earth = -G * M_earth * x / r_earth**3
    ay_earth = -G * M_earth * y / r_earth**3
    
    # Accelerations from Moon (Pulling towards its current moving position)
    ax_moon = -G * M_moon * (x - x_moon) / r_moon**3
    ay_moon = -G * M_moon * (y - y_moon) / r_moon**3
    
    # Total acceleration
    ax = ax_earth + ax_moon
    ay = ay_earth + ay_moon
    
    return [vx, vy, ax, ay]

# 3. INITIAL CONDITIONS (Trans-Lunar Injection)
# 3. INITIAL CONDITIONS (Trans-Lunar Injection)
alt_parking = 6371 + 185  
v_injection = 10.945       # Increased from 10.92 to 10.945 km/s

initial_state = [alt_parking, 0, 0, v_injection]

# 4. RUN THE SIMULATION
# Stop conditions:
# - Mission ends after the spacecraft first goes far from Earth, then returns to this distance.
# - Hard timeout: stop if the numerical simulation process takes more than 5 minutes (wall-clock).
earth_hit_altitude = 120.0
earth_hit_distance = R_earth + earth_hit_altitude
departure_distance = 100000.0
max_wall_runtime = 5 * 60
chunk_duration = 12 * 3600
sample_step = 1000.0
max_sim_time = 45 * 24 * 3600

print("Calculating dynamic Earth-Moon trajectory... Please wait.")
start_wall_time = time.time()
current_t = 0.0
current_state = np.array(initial_state, dtype=float)
has_departed = False

t_history = np.array([], dtype=float)
y_history = np.empty((4, 0), dtype=float)
stop_reason = None

while current_t < max_sim_time:
    if time.time() - start_wall_time > max_wall_runtime:
        stop_reason = "Stopped: hit 5-minute simulation timeout."
        break

    chunk_end = min(current_t + chunk_duration, max_sim_time)
    t_eval = np.arange(current_t, chunk_end, sample_step)
    if t_eval.size == 0 or t_eval[-1] < chunk_end:
        t_eval = np.append(t_eval, chunk_end)

    chunk_solution = solve_ivp(
        gravity_forces,
        (current_t, chunk_end),
        current_state,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-8,
    )

    if not chunk_solution.success:
        raise RuntimeError(f"Integrator failed at t={current_t:.1f}s: {chunk_solution.message}")

    # Avoid duplicating the first sample from each chunk after the first chunk.
    start_idx = 0 if t_history.size == 0 else 1
    r_earth_chunk = np.sqrt(chunk_solution.y[0] ** 2 + chunk_solution.y[1] ** 2)

    if not has_departed and np.any(r_earth_chunk >= departure_distance):
        has_departed = True

    hit_idx = None
    if has_departed:
        hit_candidates = np.where(r_earth_chunk <= earth_hit_distance)[0]
        for idx in hit_candidates:
            if idx >= start_idx:
                hit_idx = idx
                break

    if hit_idx is not None:
        t_slice = chunk_solution.t[start_idx : hit_idx + 1]
        y_slice = chunk_solution.y[:, start_idx : hit_idx + 1]
        t_history = np.concatenate([t_history, t_slice])
        y_history = np.concatenate([y_history, y_slice], axis=1)
        stop_reason = f"Stopped: returned within {earth_hit_distance:.1f} km of Earth's center."
        break

    t_slice = chunk_solution.t[start_idx:]
    y_slice = chunk_solution.y[:, start_idx:]
    t_history = np.concatenate([t_history, t_slice])
    y_history = np.concatenate([y_history, y_slice], axis=1)

    current_t = chunk_solution.t[-1]
    current_state = chunk_solution.y[:, -1]
else:
    stop_reason = "Stopped: reached maximum simulated mission time."

print(stop_reason)

class SimResult:
    def __init__(self, t, y):
        self.t = t
        self.y = y

solution = SimResult(t_history, y_history)

# 5. CALCULATE THE MOON'S PATH FOR THE CHART
moon_x_path = D_moon * np.cos(OMEGA_moon * solution.t + theta_0)
moon_y_path = D_moon * np.sin(OMEGA_moon * solution.t + theta_0)

# 6. PLOT THE RESULTS
plt.figure(figsize=(10, 10))
ax = plt.gca()

# Plot the spacecraft's path with a time gradient (start -> end)
points = np.array([solution.y[0], solution.y[1]]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
norm = plt.Normalize(solution.t.min(), solution.t.max())
line_collection = LineCollection(segments, cmap='plasma', norm=norm)
line_collection.set_array(solution.t[:-1])
line_collection.set_linewidth(2.5)
ax.add_collection(line_collection)

# Mark start/end clearly
plt.scatter(solution.y[0][0], solution.y[1][0], color='deepskyblue', s=60, zorder=5, label='Artemis (Start)')
plt.scatter(solution.y[0][-1], solution.y[1][-1], color='crimson', s=80, zorder=5, label='Artemis (End)')

# Plot the Moon's entire orbit path (light grey dashed line)
circle = plt.Circle((0, 0), D_moon, color='lightgray', fill=False, linestyle='--')
ax.add_patch(circle)

# Plot where the Moon started and ended
plt.scatter(moon_x_path[0], moon_y_path[0], color='lightgray', s=50, label='Moon (Start)')
plt.scatter(moon_x_path[-1], moon_y_path[-1], color='gray', s=80, label='Moon (End)')

# Plot Earth
plt.scatter(0, 0, color='green', s=150, label='Earth')

plt.title('Artemis II Simulation (Moving Moon)')
plt.xlabel('Distance (km)')
plt.ylabel('Distance (km)')
plt.legend()
plt.grid(True)
plt.axis('equal')
ax.autoscale()
plt.colorbar(line_collection, label='Mission Time (s)')
plt.show()