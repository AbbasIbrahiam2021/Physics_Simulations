import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
import random

class ForestFireSimulation:
    """
    A detailed forest fire simulation modeling heat diffusion, advection,
    wind variability, spotting, and heterogeneous terrain features.
    """
    def __init__(self,
                 logical_width: int = 200,
                 logical_height: int = 200,
                 scale: int = 4,
                 dt: float = 0.4):
        """
        Initialize simulation parameters, terrain, and GUI.
        Args:
            logical_width: Number of grid cells horizontally.
            logical_height: Number of grid cells vertically.
            scale: Pixel scaling factor for visualization.
            dt: Time step for each simulation iteration.
        """
        # Dimensions
        self.logical_width = logical_width
        self.logical_height = logical_height
        self.scale = scale
        self.width = logical_width * scale
        self.height = logical_height * scale
        self.dt = dt

        # Physical parameters
        self.diffusivity = 1.2
        self.base_wind = np.array([1.0, 0.0])
        self.base_gust_amplitude = 1.0
        self.wind_random_strength = 0.3
        self.wind_smooth_factor = 0.3
        self.cool_rate = 0.01
        self.wind_cooling_factor = 0.005
        self.ignition_temp = 1.0
        self.burn_rate = 0.02
        self.heat_release = 40.0
        self.T_ambient = 0.0

        # Spotting parameters
        self.enable_spotting = True
        self.ember_threshold = 4.5
        self.p_launch = 0.008
        self.min_spot_dist = 4
        self.max_spot_dist = 18
        self.wind_influence_factor = 0.6
        self.random_angle_deg = 25
        self.min_fuel_for_source = 0.4
        self.min_fuel_for_spot = 0.15
        self.max_moisture_for_spot = 0.7
        self.spot_ign_temp = self.ignition_temp * 1.3

        # Terrain parameters
        self.enable_fuel_heterogeneity = True
        self.enable_moisture_heterogeneity = True
        self.base_fuel = 1.0
        self.fuel_variation = 0.25
        self.base_moisture_range = (0.05, 0.25)
        self.fuel_noise_seed = 44
        self.moisture_noise_seed = 45
        self.noise_smoothing = 3
        self.min_fuel = 0.05
        self.max_fuel = 1.5

        # Pond aura parameters
        self.pond_aura_enabled = True
        self.pond_aura_width = 3
        self.pond_moisture_target = 0.55
        self.pond_fuel_factor = 0.7

        # Brown spot parameters
        self.enable_brown_spots = True
        self.brown_seed = 42
        self.brown_smoothing = 3
        self.brown_factor = 2.0
        self.brown_threshold = 0.55

        # Organic patches
        self.patch_configs = [
            # (count, base_radius, subcircles, radius_factor, offset_factor, type)
            (25, (7, 12), 3, (0.4, 0.7), 0.5, 'dead_grass'),
            (12, (12, 20), 4, (0.3, 0.6), 0.4, 'fast_burn'),
            (12, (12, 20), 4, (0.3, 0.6), 0.4, 'slow_burn'),
        ]
        self.fast_burn_modifier = 1.8
        self.slow_burn_modifier = 0.4
        self.num_rocks = 35
        self.rock_size_range = (2, 8)

        # Initialize state arrays
        self.T = np.zeros((logical_height, logical_width))
        self.fuel = np.ones((logical_height, logical_width))
        self.moisture = np.zeros((logical_height, logical_width))
        self.burning = np.zeros((logical_height, logical_width), dtype=bool)
        self.ignition_mult = np.ones((logical_height, logical_width))
        self.burn_rate_mult = np.ones((logical_height, logical_width))

        # Prepare terrain and initial wind noise
        self._generate_terrain()
        self._init_wind_noise()

        # Ignite leftmost border
        self.burning[:, 0] = True

        # Setup GUI
        self.root = tk.Tk()
        self.root.title("Forest Fire Simulation")
        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height)
        self.canvas.pack()
        self.image = Image.new("RGB", (self.width, self.height))
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas_image = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        # Time counter for gust oscillation
        self.time_step = 0
        self._schedule_update()

    def _generate_noise_map(self, seed: int, smoothing: int) -> np.ndarray:
        """
        Generate a smoothed noise map using mean filtering.
        """
        rng_state = np.random.get_state()
        np.random.seed(seed)
        noise = np.random.rand(self.logical_height, self.logical_width)
        np.random.set_state(rng_state)

        for _ in range(smoothing):
            noise = self._smooth_matrix(noise)
        minv, maxv = noise.min(), noise.max()
        return (noise - minv) / (maxv - minv) if maxv > minv else np.full_like(noise, 0.5)

    def _smooth_matrix(self, mat: np.ndarray) -> np.ndarray:
        """
        Apply a 3x3 mean filter to the matrix.
        """
        padded = np.pad(mat, 1, mode='edge')
        sm = np.zeros_like(mat)
        for i in range(self.logical_height):
            for j in range(self.logical_width):
                sm[i, j] = padded[i:i+3, j:j+3].mean()
        return sm

    def _generate_terrain(self):
        """
        Generate fuel, moisture, river, ponds, patches, rocks, and brown spots.
        """
        # Fuel heterogeneity
        if self.enable_fuel_heterogeneity:
            noise = self._generate_noise_map(self.fuel_noise_seed, self.noise_smoothing)
            mod = (noise - 0.5) * 2 * self.fuel_variation
            self.fuel = np.clip(self.base_fuel + mod, self.min_fuel, self.max_fuel)
        else:
            self.fuel.fill(self.base_fuel)

        # Moisture heterogeneity
        if self.enable_moisture_heterogeneity:
            noise = self._generate_noise_map(self.moisture_noise_seed, self.noise_smoothing)
            lo, hi = self.base_moisture_range
            self.moisture = np.clip(lo + noise * (hi - lo), 0, 0.99)
        else:
            self.moisture.fill(self.base_moisture_range[0])

        # River
        for i in range(self.logical_height):
            for j in range(self.logical_width):
                x_norm, y_norm = j / self.logical_width, i / self.logical_height
                center = 0.5 + 0.2 * np.sin(10 * y_norm) * np.cos(5 * x_norm)
                dist = abs(x_norm - center)
                if dist < 0.05:
                    self.moisture[i,j], self.fuel[i,j] = 1.0, 0.0
                elif dist < 0.08:
                    self.moisture[i,j] = max(self.moisture[i,j], 0.5)

        # Ponds and aura
        for _ in range(4):
            cy, cx = np.random.randint(0, self.logical_height), np.random.randint(0, self.logical_width)
            r = random.randint(6, 15)
            for i in range(max(0, cy-r), min(self.logical_height, cy+r+1)):
                for j in range(max(0, cx-r), min(self.logical_width, cx+r+1)):
                    if (i-cy)**2 + (j-cx)**2 < r*r:
                        if self.fuel[i,j] > 0:
                            self.moisture[i,j], self.fuel[i,j] = 1.0, 0.0
            if self.pond_aura_enabled:
                for i in range(max(0, cy-r-self.pond_aura_width), min(self.logical_height, cy+r+self.pond_aura_width+1)):
                    for j in range(max(0, cx-r-self.pond_aura_width), min(self.logical_width, cx+r+self.pond_aura_width+1)):
                        d2 = (i-cy)**2 + (j-cx)**2
                        if r*r <= d2 < (r+self.pond_aura_width)**2 and self.fuel[i,j]>0:
                            self.moisture[i,j] = max(self.moisture[i,j], self.pond_moisture_target)
                            self.fuel[i,j] = max(self.fuel[i,j]*self.pond_fuel_factor, 0.01)

        # Organic patches & rocks & brown spots
        self._apply_patches()
        self._apply_rocks()
        self._apply_brown_spots()

    def _apply_patches(self):
        """
        Add dead grass, fast-burn, and slow-burn circular patches.
        """
        for count, rad_range, subcircles, rf_range, off_factor, ptype in self.patch_configs:
            for _ in range(count):
                cx, cy = np.random.randint(0, self.logical_width), np.random.randint(0, self.logical_height)
                base_radius = random.uniform(*rad_range)
                mask = self._circular_multi_mask(cx, cy, base_radius, subcircles, rf_range, off_factor)
                valid = mask & (self.fuel > 0.01) & (self.moisture < 0.95)
                if ptype == 'dead_grass':
                    self.fuel[valid] *= 0.2
                    self.moisture[valid] = np.clip(self.moisture[valid] + 0.2, 0, 0.8)
                elif ptype == 'fast_burn':
                    self.burn_rate_mult[valid] = self.fast_burn_modifier
                elif ptype == 'slow_burn':
                    self.burn_rate_mult[valid] = self.slow_burn_modifier

    def _circular_multi_mask(self, cx, cy, radius, n_sub, r_fact_range, off_factor) -> np.ndarray:
        """
        Create a boolean mask for multiple overlapping circles around (cx, cy).
        """
        mask = np.zeros((self.logical_height, self.logical_width), dtype=bool)
        for _ in range(n_sub):
            angle, offset = random.uniform(0, 2*np.pi), random.uniform(0, radius*off_factor)
            ccx = int(np.clip(cx + offset*np.cos(angle), 0, self.logical_width-1))
            ccy = int(np.clip(cy + offset*np.sin(angle), 0, self.logical_height-1))
            rsub = max(1, radius*random.uniform(*r_fact_range))
            for i in range(self.logical_height):
                for j in range(self.logical_width):
                    if (i-ccy)**2 + (j-ccx)**2 < rsub*rsub:
                        mask[i,j] = True
        return mask

    def _apply_rocks(self):
        """
        Scatter rock patches reducing fuel and increasing moisture.
        """
        for _ in range(self.num_rocks):
            ry, rx = np.random.randint(0, self.logical_height), np.random.randint(0, self.logical_width)
            rh, rw = random.randint(*self.rock_size_range), random.randint(*self.rock_size_range)
            sub = self.fuel[max(0,ry-rh//2):ry+rh//2, max(0,rx-rw//2):rx+rw//2] > 0
            self.moisture[max(0,ry-rh//2):ry+rh//2, max(0,rx-rw//2):rx+rw//2][sub] = np.maximum(
                self.moisture[max(0,ry-rh//2):ry+rh//2, max(0,rx-rw//2):rx+rw//2][sub], 0.9)
            self.fuel[max(0,ry-rh//2):ry+rh//2, max(0,rx-rw//2):rx+rw//2][sub] = np.minimum(
                self.fuel[max(0,ry-rh//2):ry+rh//2, max(0,rx-rw//2):rx+rw//2][sub], 0.1)

    def _apply_brown_spots(self):
        """
        Speckled ignition multiplier brown spots.
        """
        if not self.enable_brown_spots:
            return
        noise = self._generate_noise_map(self.brown_seed, self.brown_smoothing)
        mask = noise >= self.brown_threshold
        strength = (noise - self.brown_threshold)/(1-self.brown_threshold)
        rand = np.random.rand(self.logical_height, self.logical_width)
        final = (rand < strength) & mask & (self.fuel>0.01) & (self.moisture<0.95)
        self.ignition_mult[final] = self.brown_factor

    def _init_wind_noise(self):
        """
        Initialize and smooth the per-cell wind gust noise.
        """
        np.random.seed(0)
        gx = (np.random.rand(self.logical_height, self.logical_width)*2-1) * self.base_gust_amplitude
        gy = (np.random.rand(self.logical_height, self.logical_width)*2-1) * self.base_gust_amplitude
        for _ in range(5):
            gx, gy = self._smooth_matrix(gx), self._smooth_matrix(gy)
        norm = np.hypot(gx, gy).max()
        if norm>1e-8:
            gx *= self.base_gust_amplitude / norm
        self.noise_x, self.noise_y = gx, gy

    def _update_temperature(self):
        """
        Compute heat diffusion, advection, cooling, and update temperature grid.
        """
        T = self.T
        pad = np.pad(T, 1, mode='edge')
        neighbors = pad[1:-1,2:], pad[1:-1,:-2], pad[2:,1:-1], pad[:-2,1:-1]
        laplacian = sum(neighbors) - 4*T
        diff = self.diffusivity * laplacian

        wx = self.base_wind[0] + self.noise_x
        wy = self.base_wind[1] + self.noise_y
        adv = self._compute_advection(T, neighbors, wx, wy)
        cooling = -self.cool_rate * T
        wind_cool = self._compute_wind_front_cooling(wx, wy)

        self.T = np.maximum(self.T + self.dt*(diff + adv + cooling - wind_cool), self.T_ambient)
        water = (self.fuel==0)&(self.moisture==1)
        self.T[water] = self.T_ambient

    def _compute_advection(self, T, neigh, wx, wy):
        """
        Compute advection term from wind components.
        """
        T_rt, T_lf, T_dn, T_up = neigh
        adv = np.zeros_like(T)
        pos_x = wx>0; neg_x=wx<0
        adv[pos_x] -= wx[pos_x]*(T[pos_x]-T_lf[pos_x])
        adv[neg_x] -= wx[neg_x]*(T_rt[neg_x]-T[neg_x])
        pos_y = wy>0; neg_y=wy<0
        adv[pos_y] -= wy[pos_y]*(T[pos_y]-T_up[pos_y])
        adv[neg_y] -= wy[neg_y]*(T_dn[neg_y]-T[neg_y])
        return adv

    def _compute_wind_front_cooling(self, wx, wy):
        """
        Cooling effect at fire front due to wind.
        """
        mask = self.burning & ~(self.fuel>0.01)
        cool = np.zeros_like(self.T)
        if np.any(self.burning):
            unburn = (self.fuel>0.01)&~self.burning
            adj = (np.pad(unburn,1)[1:-1, :-2]|
                   np.pad(unburn,1)[1:-1,2: ]|
                   np.pad(unburn,1)[:-2,1:-1]|
                   np.pad(unburn,1)[2:,1:-1])
            front = self.burning & adj
            fi,fj = np.where(front)
            speeds = np.hypot(wx[fi,fj], wy[fi,fj])
            cool_val = self.wind_cooling_factor * speeds * (self.T[fi,fj]-self.T_ambient)
            cool[fi,fj] = np.maximum(cool_val, 0)
        return cool

    def _update_fuel_and_ignite(self):
        """
        Burn fuel at burning cells and apply ignition of neighbors and spotting.
        """
        i_idx, j_idx = np.where(self.burning)
        if i_idx.size>0:
            dry = np.clip(1-self.moisture[i_idx,j_idx], 0,1)
            rates = self.burn_rate * self.burn_rate_mult[i_idx,j_idx]
            delta = rates * dry * self.dt * np.random.uniform(0.6,1.4, size=dry.shape)
            burnt = np.minimum(delta, self.fuel[i_idx,j_idx])
            self.fuel[i_idx,j_idx] -= burnt
            self.T[i_idx,j_idx] += self.heat_release * burnt * (1-self.moisture[i_idx,j_idx])
            done = burnt>=self.fuel[i_idx,j_idx]
            self.burning[i_idx[done], j_idx[done]] = False

        # Main ignition
        thresh = self.ignition_temp * self.ignition_mult * (1+3*self.moisture +
                                                              np.random.uniform(-0.5,0.5, self.moisture.shape))
        main_mask = (~self.burning)&(self.fuel>0.01)&(self.T>=thresh)

        # Spotting
        if self.enable_spotting and np.any(self.burning):
            self._ignite_spots(main_mask)

        self.burning[main_mask] = True

    def _ignite_spots(self, main_mask: np.ndarray):
        """
        Model ember spotting based on wind and randomly launch new fires.
        """
        sources = np.where((self.burning)&(self.T>self.ember_threshold)&(self.fuel>self.min_fuel_for_source))
        if sources[0].size==0:
            return
        probs = np.random.rand(sources[0].size)
        for s, p in zip(zip(*sources), probs):
            if p>=self.p_launch:
                continue
            i,j = s
            wx, wy = self.base_wind + np.array([self.noise_x[i,j], self.noise_y[i,j]])
            speed = np.hypot(wx, wy)
            maxd = min(max(self.max_spot_dist, speed*self.wind_influence_factor),
                       max(self.logical_width, self.logical_height)/2)
            dist = random.uniform(self.min_spot_dist, max(self.min_spot_dist, maxd))
            angle = (random.uniform(0,2*np.pi) if speed<1e-5 else
                     np.arctan2(wy,wx) + np.deg2rad(random.uniform(-self.random_angle_deg, self.random_angle_deg)))
            ni = int(round(i + dist*np.sin(angle)))
            nj = int(round(j + dist*np.cos(angle)))
            if (0<=ni<self.logical_height and 0<=nj<self.logical_width and
                not self.burning[ni,nj] and not main_mask[ni,nj] and
                self.fuel[ni,nj]>self.min_fuel_for_spot and
                self.moisture[ni,nj]<self.max_moisture_for_spot):
                self.burning[ni,nj] = True
                self.T[ni,nj] = max(self.T[ni,nj], self.spot_ign_temp)

    def _update_wind(self):
        """
        Add randomness to wind, then smooth noise field.
        """
        self.noise_x += (np.random.rand(self.logical_height, self.logical_width)*2-1)*self.wind_random_strength
        self.noise_y += (np.random.rand(self.logical_height, self.logical_width)*2-1)*self.wind_random_strength
        for _ in range(1):
            self.noise_x, self.noise_y = self._smooth_matrix(self.noise_x), self._smooth_matrix(self.noise_y)
        norm = np.hypot(self.noise_x, self.noise_y).max()
        if norm>1e-8:
            factor = (self.base_gust_amplitude*(1+0.3*np.sin(2*np.pi*self.time_step/400))) / norm
            self.noise_x *= factor
            self.noise_y *= factor

    def _render(self):
        """
        Draw the current temperature, fuel, and fire state to the canvas image.
        """
        pixels = self.image.load()
        for i in range(self.logical_height):
            for j in range(self.logical_width):
                clr = (0, 0, 0)
                if self.moisture[i, j] >= 1 and self.fuel[i, j] == 0:
                    clr = (51, 102, 204)
                elif self.burning[i, j]:
                    tval = self.T[i, j]
                    tn = min(tval / (5 * self.ignition_temp), 1)
                    if tn < 0.5:
                        r, g, b = 255, int(255 * (tn / 0.5)), 0
                    else:
                        r, g, b = 255, 255, int(255 * ((tn - 0.5) / 0.5))
                    clr = (r, g, b)
                elif self.fuel[i, j] <= 0.01:
                    clr = (17, 17, 17)
                elif self.ignition_mult[i, j] > 1:
                    cs = 1 - self.moisture[i, j] * 0.4
                    r, g, b = int(139 * cs), int(69 * cs), int(19 * cs)
                    clr = (r, g, b)
                else:
                    mv, fv, bm = self.moisture[i, j], self.fuel[i, j], self.burn_rate_mult[i, j]
                    r = int(34 + mv * 30 * (1 - (1 - fv) * 0.5))
                    g = int(139 + mv * 60 * (1 - (1 - fv) * 0.5))
                    b = int(34 + mv * 50 * (1 - (1 - fv) * 0.5))
                    if bm > 1:
                        r, g, b = int(r * 1.1 + g * 0.1), int(g * 1.05), int(b * 0.85)
                    elif bm < 1:
                        r, g, b = int(r * 0.85), int(g * 0.95), int(b * 1.1 + g * 0.05)
                    clr = (max(0, min(r, 255)), max(0, min(g, 255)), max(0, min(b, 255)))
                for dy in range(self.scale):
                    for dx in range(self.scale):
                        pixels[j * self.scale + dx, i * self.scale + dy] = clr

        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.itemconfig(self.canvas_image, image=self.tk_image)

    def _schedule_update(self):
        self.root.after(30, self._update)

    def _update(self):
        self._update_temperature()
        self._update_fuel_and_ignite()
        self._update_wind()
        self._render()
        self.time_step += 1
        self._schedule_update()

if __name__ == '__main__':
    sim = ForestFireSimulation()
    sim.root.mainloop()