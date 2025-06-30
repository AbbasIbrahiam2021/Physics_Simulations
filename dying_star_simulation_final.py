"""
Star Death Simulation: Red Giant → White Dwarf Collapse
------------------------------------------------------

This program simulates the thermal and electromagnetic evolution of a dying star,
as it expands from main sequence to red giant, then collapses into a white dwarf.

Core physics:
- Solves the 1D radial heat diffusion equation (explicit Euler method)
  under spherical symmetry, assuming no internal heat generation.
- Visualizes magnetic field lines as static dipole loops, derived from the
  analytical solution to Maxwell's equations for a static magnetic dipole.

Magnetic field model:
We plot magnetic field lines based on the equation of a magnetic dipole in polar coordinates:

    B(r, θ) = (μ₀ / 4π) * (m / r³) * sqrt(1 + 3 cos²θ)

Instead of computing B directly, we plot curves of constant C in the dipole equation:

    r(θ) = C * sin²θ

which represent the shape of dipole field lines (C is constant for each line).

Visualization:
- The star is displayed as concentric rings colored by temperature.
- Magnetic field loops are drawn dynamically with arrows to represent field direction,
  increasing in brightness and thickness as the star collapses.

Assumptions:
- Heat conduction modeled with normalized diffusivity constant.
- Magnetic field is static and idealized; no interaction with plasma or currents.
- All quantities are in normalized or pixel units for visualization.
"""

import tkinter as tk
import numpy as np
import math

#  Simulation parameters 
TOTAL_TIME = 48.0  # total simulation time in seconds
FPS = 20           # frames per second
DT = 1.0 / FPS     # time step in seconds per frame
dt_ms = int(1000 / FPS)  # time step in milliseconds for Tkinter scheduling

# Star physical properties 
ALPHA = 1.0            # normalized thermal diffusivity
INITIAL_R = 60         # radius (pixels) at main sequence
GIANT_R = 150          # radius (pixels) at red giant phase
FINAL_R = 40           # radius (pixels) at white dwarf
INITIAL_T_SURF = 0.7   # normalized surface temp at main sequence
MIN_T_SURF = 0.3       # normalized surface temp at red giant
FINAL_T_SURF = 0.9     # normalized surface temp at white dwarf

# Radial grid setup 
MAX_R = GIANT_R
N_RADIAL = MAX_R + 1

dr = 1.0  # radial grid spacing (pixels)

# Initialize temperature profile: quadratic gradient from core to surface
T = np.zeros(N_RADIAL)
for r in range(N_RADIAL):
    if r <= INITIAL_R:
        frac = r / INITIAL_R
        T[r] = INITIAL_T_SURF + (1.0 - INITIAL_T_SURF) * (1 - frac ** 2)
    else:
        T[r] = 0.0  # space beyond star is cold

# Tkinter GUI
root = tk.Tk()
root.title("Red Giant → White Dwarf Collapse")
canvas_size = 700
canvas = tk.Canvas(root, width=canvas_size, height=canvas_size, bg='black')
canvas.pack()
cx = cy = canvas_size // 2  # center coordinates

# Create concentric oval shapes for each radial cell (outermost to innermost)
rings = [canvas.create_oval(cx, cy, cx, cy, fill='', outline='') for _ in range(N_RADIAL)]

# Add a simple surface temperature meter (side UI bar)
canvas.create_text(550, 80, text='Surface Temp', fill='white', font=('Arial', 12))
canvas.create_rectangle(550, 90, 680, 130, outline='white')
surf_bar = canvas.create_rectangle(552, 92, 552, 128, fill='cyan', width=0)

frame_count = 0
total_frames = int(TOTAL_TIME * FPS)

def temp_to_color(t):
    """Map normalized temperature [0,1] to RGB color: red → orange → yellow → white."""
    t = max(0.0, min(1.0, t))
    if t < 0.3:
        f = t / 0.3
        r, g, b = int(255 * f), 0, 0
    elif t < 0.5:
        f = (t - 0.3) / 0.2
        r, g, b = 255, int(165 * f), 0
    elif t < 0.7:
        f = (t - 0.5) / 0.2
        r, g, b = 255, int(165 + (255 - 165) * f), 0
    else:
        f = (t - 0.7) / 0.3
        r, g, b = 255, 255, int(255 * f)
    return f'#{r:02x}{g:02x}{b:02x}'

def update_frame():
    """Advance simulation by one time step and render visualization."""
    global frame_count, T

    frac = frame_count / total_frames  # fraction of timeline (0 to 1)

    # --- Calculate current star radius and target surface temperature ---
    if frac <= 0.5:
        # expansion phase (main sequence → red giant)
        phase = frac / 0.5
        R_star = int(INITIAL_R + (GIANT_R - INITIAL_R) * phase)
        T_surf = INITIAL_T_SURF + (MIN_T_SURF - INITIAL_T_SURF) * phase
    else:
        # collapse phase (red giant → white dwarf)
        phase = (frac - 0.5) / 0.5
        R_star = int(GIANT_R + (FINAL_R - GIANT_R) * phase)
        T_surf = MIN_T_SURF + (FINAL_T_SURF - MIN_T_SURF) * phase

    R_star = max(R_star, 1)  # ensure radius doesn't collapse to zero

    # --- Solve heat diffusion equation (1D radial, explicit Euler) ---
    # Discretized from: ∂T/∂t = α (∂²T/∂r² + (1/r) ∂T/∂r)
    new_T = T.copy()
    for r in range(1, R_star):
        d2 = T[r + 1] - 2 * T[r] + T[r - 1]  # second derivative (∂²T/∂r²)
        d1 = T[r + 1] - T[r - 1]              # finite difference (∂T/∂r)
        new_T[r] = T[r] + ALPHA * DT * (d2 + d1 / r)
    new_T[0] = T[0] + ALPHA * DT * 2 * (T[1] - T[0])  # symmetry at center (zero gradient)

    new_T[R_star] = T_surf  # enforce surface temperature (Dirichlet boundary)
    new_T[R_star + 1:] = 0.0  # beyond surface is cold space

    T = np.clip(new_T, 0.0, 1.0)  # clamp to valid [0,1]

    # Update surface temperature meter
    surf_temp = T[R_star]
    bar_max_width = 680 - 552
    bar_width = int(bar_max_width * surf_temp)
    canvas.coords(surf_bar, 552, 92, 552 + bar_width, 128)
    canvas.itemconfig(surf_bar, fill='white')

    # Render temperature rings
    for r in range(N_RADIAL):
        col = temp_to_color(T[r])
        x0, y0 = cx - r, cy - r
        x1, y1 = cx + r, cy + r
        canvas.coords(rings[N_RADIAL - 1 - r], x0, y0, x1, y1)
        canvas.itemconfig(rings[N_RADIAL - 1 - r], fill=col)

    # Clear old magnetic field lines
    canvas.delete('fieldline')

    # Draw magnetic dipole field lines
    # We use the equation r(θ) = C * sin²θ to trace each dipole field line,
    # where C is a constant defining each line's scale (lines of constant magnetic potential)
    min_lines, max_lines = 3, 20
    count = int(min_lines + frac * (max_lines - min_lines))
    count = max(min(count, max_lines), min_lines)
    lw = 1 + int(frac * 3)  # line width grows as star collapses
    bright = int(100 + frac * 155)  # brightness increases toward collapse
    color = f'#{bright:02x}{bright:02x}{bright:02x}'

    # Generate anchor angles for dipole footpoints from 10° to 80°
    angles = np.linspace(math.radians(10), math.radians(80), count)
    for theta0 in angles:
        C = R_star / (math.sin(theta0) ** 2)  # calculate constant C for this line
        pts = []
        steps = 100  # number of points per curve
        for j in range(steps + 1):
            theta = theta0 + (math.pi - 2 * theta0) * j / steps
            r_val = C * (math.sin(theta) ** 2)  # evaluate r(θ)
            x = r_val * math.sin(theta)
            y = r_val * math.cos(theta)
            pts.extend((cx + x, cy - y))  # map to canvas coordinates

        # draw right-hand (east) loop
        canvas.create_line(*pts, fill=color, width=lw, smooth=True,
                           arrow='last', arrowshape=(8, 10, 3), tag='fieldline')

        # mirror to left-hand (west) side by reflecting x across center
        pts_m = [(2 * cx - px, py) for px, py in zip(pts[::2], pts[1::2])]
        flat = []
        for px, py in pts_m:
            flat.extend((px, py))
        canvas.create_line(*flat, fill=color, width=lw, smooth=True,
                           arrow='last', arrowshape=(8, 10, 3), tag='fieldline')

    frame_count += 1
    if frame_count <= total_frames:
        root.after(dt_ms, update_frame)

# Launch simulation 
root.after(0, update_frame)
root.mainloop()
print("Simulation Finished")


