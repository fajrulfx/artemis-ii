# [Physics.md](http://Physics.md)

The core principles behind the simulation in `core.py` (and its JS port in
`web/simulation.html`).

## Assumptions

We're simulating the restricted three-body problem in 2D:

- Earth sits fixed at the origin.
- The Moon orbits Earth on a perfect circle at constant speed.
- The spacecraft is treated as a point mass with no gravity of its own
(it's pulled by Earth and Moon, but doesn't pull back).
- Only gravity matters
- Everything happens in a flat 2D plane.

These simplifications are wrong in detail but right enough to reproduce the
key behavior of a lunar transfer.

## Constants


| Symbol | Value | What it is |
| ------ | ----- | ---------- |
| $G$ | $6.6743 \times 10^{-20}$ | Gravitational constant (km³/kg·s²) |
| $M_\oplus$ | $5.972 \times 10^{24}$ kg | Earth mass |
| $M_{\text{moon}}$ | $7.348 \times 10^{22}$ kg | Moon mass |
| $D_{\text{moon}}$ | 384,400 km | Earth–Moon distance |
| $T_{\text{moon}}$ | 27.32 days | Moon's orbital period |


## How the Moon moves

The Moon's position is just a circle, parameterized by time:

$$x_{\text{moon}}(t) = D_{\text{moon}}\cos(\Omega t + \theta_0)$$

$$y_{\text{moon}}(t) = D_{\text{moon}}\sin(\Omega t + \theta_0)$$

where $\Omega = 2\pi / T_{\text{moon}}$ is the angular speed.

The starting angle $\theta_0 \approx 126°$ is a **lead angle**. The
spacecraft is not aimed at the Moon's position at launch, but at the position
the Moon will occupy roughly three days later when the spacecraft arrives.

## The force on the spacecraft

Newton's law of gravity from each body, summed:

$$\mathbf{a} = -\frac{G M_\oplus}{r_\oplus^{3}}\mathbf{r} - \frac{G M_{\text{moon}}}{r_{\text{moon}}^{3}}(\mathbf{r} - \mathbf{r}_{\text{moon}})$$

In words: the spacecraft is pulled toward Earth and toward wherever the Moon
currently is. Each pull weakens with the inverse square of the distance.
The $\mathbf{r}/r^3$ form is just $(1/r^2) \times \hat{r}$.

This gives us a system of 4 equations (position and velocity in x and y):

$$\dot{x} = v_x, \quad \dot{y} = v_y, \quad \dot{v_x} = a_x, \quad \dot{v_y} = a_y$$




## Initial conditions (the TLI burn)

The simulation starts at the moment of Trans-Lunar Injection, the engine burn that flings the spacecraft from parking orbit toward the Moon.

- Position: 185 km above Earth's surface, on the +x axis.
- Velocity: 10.945 km/s, perpendicular to the position (tangential).

Why 10.945 km/s? It's just below escape velocity (~11.01 km/s at that
altitude), giving an elongated ellipse whose far end reaches the Moon.

## Numerical integration

The three-body problem has no closed-form solution, so the state must be
advanced one small time step at a time using a numerical integrator.

The simulation uses **Runge-Kutta 4 (RK4)**. Each step samples the
acceleration four times (at the start, twice at the midpoint, and at the
end) and combines them in a weighted average to advance position and
velocity. This is much more accurate than the simpler Euler method (a single
sample at the start of the step), which accumulates noticeable energy drift
in orbital problems.

`core.py` uses **RK45** through SciPy's `solve_ivp`. RK45 is RK4 with an
adaptive step size: it automatically shrinks $dt$ near close approaches,
where gravity is strong and the trajectory curves sharply, and grows $dt$
during long coasting phases. This keeps error bounded without wasting work.

`web/simulation.html` uses **fixed-step RK4** in JavaScript. It is less
accurate than RK45 but has a predictable cost per frame, which is required
for a real-time interactive UI.

## Stop conditions

The Python simulation stops when one of these happens:

1. The spacecraft returns within ~120 km of Earth's surface (after having
  first traveled far away). Mission complete (or crash).
2. Simulated time exceeds 45 days.
3. Real wall-clock time exceeds 5 minutes.

