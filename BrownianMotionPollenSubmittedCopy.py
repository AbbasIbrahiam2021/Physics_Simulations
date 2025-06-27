import tkinter as tk
import random
import math
import matplotlib
matplotlib.use("TkAgg") # Use Tkinter backend for Matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Constants
CANVAS_WIDTH = 800
CANVAS_HEIGHT = 600 # Height of the simulation canvas
PARTICLE_VISUAL_RADIUS = 4
PHYSICAL_PARTICLE_RADIUS_M = 1.0e-6

POLLEN_DENSITY_KG_M3 = 1200.0
PARTICLE_VOLUME_M3 = (4/3) * math.pi * (PHYSICAL_PARTICLE_RADIUS_M**3)
PHYSICAL_PARTICLE_MASS_KG = POLLEN_DENSITY_KG_M3 * PARTICLE_VOLUME_M3

NUM_PARTICLES = 75
BOLTZMANN_CONSTANT_KB = 1.380649e-23

# Reduced timestep to fufill Bot's crtitism 
GUI_UPDATE_INTERVAL_MS = 30
PHYSICAL_TIME_STEP_S = 0.001

PLOT_HEIGHT_PIXELS = 300
PLOT_UPDATE_FRAMES_INTERVAL = 10

# Wave "tickles" pollen grains
WAVE_PULSE_INTERVAL_S = 5.0
WAVE_PROPAGATION_SPEED_PIXELS_S = 150.0
WAVE_MAX_AMPLITUDE_VELOCITY_PIXELS_S = 8000.0
WAVE_PULSE_WIDTH_PIXELS = 60.0
WAVE_AMPLITUDE_DECAY_FACTOR = 0.5

# Viscosity constants
WATER_VISCOSITY_DATA_PA_S = {
    0: 0.001792,     5: 0.001519,     10: 0.001307,     15: 0.001138,
    20: 0.001002,     25: 0.000890,     30: 0.000798,     35: 0.000719,
    40: 0.000653,     50: 0.000547,     60: 0.000467,     70: 0.000404,
    80: 0.000355,     90: 0.000315,     100: 0.000282
}

particles = []
active_wave_pulses = []
current_temp_celsius = 20.0 # This is the one variable controlled by the user via GUI
total_elapsed_physical_time_s = 0.0
last_wave_creation_time_s = -WAVE_PULSE_INTERVAL_S

CANVAS_PHYSICAL_WIDTH_M = 200e-6
PIXELS_PER_METER = CANVAS_WIDTH / CANVAS_PHYSICAL_WIDTH_M
CANVAS_CENTER_X = CANVAS_WIDTH / 2
CANVAS_CENTER_Y = CANVAS_HEIGHT / 2
CANVAS_DIAGONAL = math.sqrt(CANVAS_WIDTH**2 + CANVAS_HEIGHT**2)

plot_data_time_s = []
plot_data_rms_disp_um = []
plot_update_counter = 0

class WavePulse:
    def __init__(self, creation_time_s, origin_x_px, origin_y_px,
                 speed_px_s, initial_max_amplitude_px_s, pulse_width_px, decay_factor):
        self.creation_time_s = creation_time_s
        self.origin_x_px = origin_x_px; self.origin_y_px = origin_y_px
        self.speed_px_s = speed_px_s
        self.initial_max_amplitude_px_s = initial_max_amplitude_px_s
        self.pulse_width_px = pulse_width_px; self.decay_factor = decay_factor
        self.visual_id = None

    def get_current_leading_edge_radius(self, current_sim_time_s):
        age = current_sim_time_s - self.creation_time_s
        return self.speed_px_s * age if age >= 0 else 0

    def get_fluid_velocity_contribution(self, particle_x_px, particle_y_px, current_sim_time_s):
        dx = particle_x_px - self.origin_x_px; dy = particle_y_px - self.origin_y_px
        dist_from_origin = math.sqrt(dx**2 + dy**2)
        if dist_from_origin < 1e-6: return 0, 0

        current_wave_front_radius = self.get_current_leading_edge_radius(current_sim_time_s)
        if (current_wave_front_radius - self.pulse_width_px) < dist_from_origin <= current_wave_front_radius and current_wave_front_radius > 0:
            phase_in_pulse = math.pi * (current_wave_front_radius - dist_from_origin) / self.pulse_width_px
            decayed_amplitude = self.initial_max_amplitude_px_s
            effective_radius_for_decay = max(current_wave_front_radius, self.pulse_width_px * 0.1)
            if self.decay_factor > 0 and effective_radius_for_decay > 0:
                reference_decay_length = max(1.0, self.pulse_width_px)
                decayed_amplitude /= (effective_radius_for_decay / reference_decay_length)**self.decay_factor
            radial_vel_mag = decayed_amplitude * math.sin(phase_in_pulse)
            return radial_vel_mag * (dx / dist_from_origin), radial_vel_mag * (dy / dist_from_origin)
        return 0, 0

    def is_finished(self, current_sim_time_s, max_canvas_dim):
        age = current_sim_time_s - self.creation_time_s
        return self.speed_px_s * age > max_canvas_dim * 1.2 if age >=0 else False

    def draw(self, canvas_widget, current_sim_time_s):
        current_r = self.get_current_leading_edge_radius(current_sim_time_s)
        if not (0 < current_r <= CANVAS_DIAGONAL * 1.2):
            if self.visual_id: canvas_widget.delete(self.visual_id); self.visual_id = None
            return
        x0, y0, x1, y1 = (self.origin_x_px - current_r, self.origin_y_px - current_r,
                          self.origin_x_px + current_r, self.origin_y_px + current_r)
        if self.visual_id: canvas_widget.coords(self.visual_id, x0, y0, x1, y1)
        else: self.visual_id = canvas_widget.create_oval(x0, y0, x1, y1, outline="SteelBlue4", width=2, tags="wave_visual")

class Particle:
    def __init__(self, x_px, y_px, visual_radius_px, color_str, canvas_widget):
        self.x_px = x_px; self.y_px = y_px
        self.initial_x_px = x_px
        self.initial_y_px = y_px
        self.radius_px = visual_radius_px; self.color = color_str
        self.canvas = canvas_widget
        self.vx_m_s = 0.0; self.vy_m_s = 0.0
        self.id = self.canvas.create_oval(
            self.x_px - self.radius_px, self.y_px - self.radius_px,
            self.x_px + self.radius_px, self.y_px + self.radius_px,
            fill=self.color, outline="black")

    def update_state_langevin(self, gamma, mass, temp_kelvin, dt_s, fluid_ux_m_s, fluid_uy_m_s):
        if mass <= 1e-25 or gamma <= 1e-15:
            if gamma <= 1e-15: pass
            else: return
        vx_rel_m_s = self.vx_m_s - fluid_ux_m_s; vy_rel_m_s = self.vy_m_s - fluid_uy_m_s
        beta = gamma / mass; exp_beta_dt = math.exp(-beta * dt_s)
        term_sqrt_factor = (BOLTZMANN_CONSTANT_KB * temp_kelvin / mass) * (1.0 - math.exp(-2.0 * beta * dt_s))
        random_strength = math.sqrt(max(0, term_sqrt_factor))
        new_vx_rel_m_s = vx_rel_m_s * exp_beta_dt + random_strength * random.gauss(0,1)
        new_vy_rel_m_s = vy_rel_m_s * exp_beta_dt + random_strength * random.gauss(0,1)
        self.vx_m_s = new_vx_rel_m_s + fluid_ux_m_s; self.vy_m_s = new_vy_rel_m_s + fluid_uy_m_s
        self.x_px += self.vx_m_s * dt_s * PIXELS_PER_METER
        self.y_px += self.vy_m_s * dt_s * PIXELS_PER_METER
        if self.x_px - self.radius_px < 0: self.x_px = self.radius_px; self.vx_m_s *= -1 if self.vx_m_s < 0 else 0 # Simplified assignment
        elif self.x_px + self.radius_px > CANVAS_WIDTH: self.x_px = CANVAS_WIDTH - self.radius_px; self.vx_m_s *= -1 if self.vx_m_s > 0 else 0
        if self.y_px - self.radius_px < 0: self.y_px = self.radius_px; self.vy_m_s *= -1 if self.vy_m_s < 0 else 0
        elif self.y_px + self.radius_px > CANVAS_HEIGHT: self.y_px = CANVAS_HEIGHT - self.radius_px; self.vy_m_s *= -1 if self.vy_m_s > 0 else 0
        self.canvas.coords(self.id, self.x_px - self.radius_px, self.y_px - self.radius_px,
                           self.x_px + self.radius_px, self.y_px + self.radius_px)

def get_water_viscosity_pa_s(temp_c):
    sorted_temps = sorted(WATER_VISCOSITY_DATA_PA_S.keys())
    if temp_c <= sorted_temps[0]: return WATER_VISCOSITY_DATA_PA_S[sorted_temps[0]]
    if temp_c >= sorted_temps[-1]: return WATER_VISCOSITY_DATA_PA_S[sorted_temps[-1]]
    for i in range(len(sorted_temps) - 1):
        t1, t2 = sorted_temps[i], sorted_temps[i+1]
        if t1 <= temp_c <= t2:
            visc1 = WATER_VISCOSITY_DATA_PA_S[t1]; visc2 = WATER_VISCOSITY_DATA_PA_S[t2]
            return visc1 + (visc2 - visc1) * (temp_c - t1) / (t2 - t1)
    return WATER_VISCOSITY_DATA_PA_S[sorted_temps[-1]]

def calculate_total_fluid_velocity_at_point(px, py, time_s):
    total_ux_px_s = 0; total_uy_px_s = 0
    for wave in active_wave_pulses:
        ux, uy = wave.get_fluid_velocity_contribution(px, py, time_s)
        total_ux_px_s += ux; total_uy_px_s += uy
    return total_ux_px_s, total_uy_px_s

def handle_temperature_change(temp_val_str):
    global current_temp_celsius
    current_temp_celsius = float(temp_val_str)

def initialize_simulation_particles():
    global particles, plot_data_time_s, plot_data_rms_disp_um, total_elapsed_physical_time_s, last_wave_creation_time_s, active_wave_pulses
    for p_obj in particles: canvas.delete(p_obj.id)
    particles = []
    for wave_obj in active_wave_pulses:
        if wave_obj.visual_id: canvas.delete(wave_obj.visual_id)
    active_wave_pulses = []
    
    particle_colors_list = ["yellow", "orange", "red", "lime green", "pink", "white", "cyan", "magenta"]
    for _ in range(NUM_PARTICLES):
        color = random.choice(particle_colors_list)
        x_start_px = random.uniform(PARTICLE_VISUAL_RADIUS, CANVAS_WIDTH - PARTICLE_VISUAL_RADIUS)
        y_start_px = random.uniform(PARTICLE_VISUAL_RADIUS, CANVAS_HEIGHT - PARTICLE_VISUAL_RADIUS)
        p = Particle(x_start_px, y_start_px, PARTICLE_VISUAL_RADIUS, color, canvas)
        particles.append(p)
    
    plot_data_time_s = []
    plot_data_rms_disp_um = []
    total_elapsed_physical_time_s = 0.0
    last_wave_creation_time_s = -WAVE_PULSE_INTERVAL_S
    update_rms_displacement_plot()

def update_rms_displacement_plot():
    plot_ax.clear()
    if plot_data_time_s and plot_data_rms_disp_um:
        plot_ax.plot(plot_data_time_s, plot_data_rms_disp_um, marker='.', linestyle='-', markersize=2, color='green')
    plot_ax.set_xlabel("Time (s)")
    plot_ax.set_ylabel("RMS Displacement (µm)")
    plot_ax.set_title("Pollen RMS Displacement vs. Time")
    plot_ax.grid(True)
    if plot_data_rms_disp_um:
         plot_ax.set_ylim(bottom=min(0, min(plot_data_rms_disp_um)*0.9 if plot_data_rms_disp_um else 0), 
                          top = max(1, max(plot_data_rms_disp_um)*1.1 if plot_data_rms_disp_um else 1) )
    else:
        plot_ax.set_ylim(bottom=0, top=1) 
    plot_canvas_widget.draw()

def simulation_animation_step():
    global total_elapsed_physical_time_s, last_wave_creation_time_s, active_wave_pulses, plot_update_counter

    num_physics_steps = max(1, int((GUI_UPDATE_INTERVAL_MS / 1000.0) / PHYSICAL_TIME_STEP_S)) if PHYSICAL_TIME_STEP_S > 1e-9 else 1
    frame_start_time = total_elapsed_physical_time_s

    for i_step in range(num_physics_steps):
        sub_step_time = frame_start_time + i_step * PHYSICAL_TIME_STEP_S
        if sub_step_time >= last_wave_creation_time_s + WAVE_PULSE_INTERVAL_S:
            new_wave = WavePulse(sub_step_time, CANVAS_CENTER_X, CANVAS_CENTER_Y, WAVE_PROPAGATION_SPEED_PIXELS_S,
                                 WAVE_MAX_AMPLITUDE_VELOCITY_PIXELS_S, WAVE_PULSE_WIDTH_PIXELS, WAVE_AMPLITUDE_DECAY_FACTOR)
            active_wave_pulses.append(new_wave)
            last_wave_creation_time_s = sub_step_time
        
        temp_K = current_temp_celsius + 273.15
        visc = get_water_viscosity_pa_s(current_temp_celsius)
        gamma_val = 6 * math.pi * visc * PHYSICAL_PARTICLE_RADIUS_M

        for p_obj in particles:
            fluid_ux_px, fluid_uy_px = calculate_total_fluid_velocity_at_point(p_obj.x_px, p_obj.y_px, sub_step_time)
            fluid_ux_m = fluid_ux_px / PIXELS_PER_METER
            fluid_uy_m = fluid_uy_px / PIXELS_PER_METER
            p_obj.update_state_langevin(gamma_val, PHYSICAL_PARTICLE_MASS_KG, temp_K, PHYSICAL_TIME_STEP_S, fluid_ux_m, fluid_uy_m)
    
    total_elapsed_physical_time_s = frame_start_time + num_physics_steps * PHYSICAL_TIME_STEP_S

    if particles:
        sum_sq_disp_px2 = 0
        for p_obj in particles:
            dx_px = p_obj.x_px - p_obj.initial_x_px
            dy_px = p_obj.y_px - p_obj.initial_y_px
            sum_sq_disp_px2 += dx_px**2 + dy_px**2
        msd_px2 = sum_sq_disp_px2 / len(particles)
        rms_disp_px = math.sqrt(msd_px2)
        rms_disp_um = (rms_disp_px / PIXELS_PER_METER) * 1e6
        plot_data_time_s.append(total_elapsed_physical_time_s)
        plot_data_rms_disp_um.append(rms_disp_um)

    plot_update_counter += 1
    if plot_update_counter >= PLOT_UPDATE_FRAMES_INTERVAL:
        update_rms_displacement_plot()
        plot_update_counter = 0
    
    next_active_waves = []
    for wave_pulse_item in active_wave_pulses:
        if not wave_pulse_item.is_finished(total_elapsed_physical_time_s, CANVAS_DIAGONAL):
            next_active_waves.append(wave_pulse_item)
            wave_pulse_item.draw(canvas, total_elapsed_physical_time_s)
        elif wave_pulse_item.visual_id: canvas.delete(wave_pulse_item.visual_id)
    active_wave_pulses = next_active_waves
        
    root.after(GUI_UPDATE_INTERVAL_MS, simulation_animation_step)

# Main script
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Pollen Dynamics with Waves & RMS Displacement Plot")
    
    canvas = tk.Canvas(root, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="light blue") 
    canvas.pack()
    
    controls_frame_bg = "gray10" # Dark background for controls
    controls_frame = tk.Frame(root, bg=controls_frame_bg)
    controls_frame.pack(pady=5, fill="x")
    
    # Temperature scale
    temp_scale = tk.Scale(controls_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                          label="Water Temperature (°C)", resolution=1, length=300,
                          command=handle_temperature_change, fg="white", bg=controls_frame_bg,
                          troughcolor="gray40", highlightbackground=controls_frame_bg,
                          activebackground="gray25")
    temp_scale.set(current_temp_celsius)
    temp_scale.pack(pady=5) 

    plot_frame = tk.Frame(root)
    plot_frame.pack(fill=tk.BOTH, expand=True)

    plot_fig = Figure(figsize=(CANVAS_WIDTH/100, PLOT_HEIGHT_PIXELS/100), dpi=100)
    plot_ax = plot_fig.add_subplot(111)
    
    plot_canvas_widget = FigureCanvasTkAgg(plot_fig, master=plot_frame)
    plot_canvas_widget.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    # Call handle_temperature_change once to set the initial current_temp_celsius from the scale
    handle_temperature_change(str(current_temp_celsius)) 
    initialize_simulation_particles()
    simulation_animation_step()
    
    root.mainloop()