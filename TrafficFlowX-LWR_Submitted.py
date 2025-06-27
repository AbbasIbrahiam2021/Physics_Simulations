import tkinter as tk
import numpy as np
import random
import math

# --- Simulation Constants (keeping your aesthetics) ---
CELL_SIZE = 12
SIMULATION_DELAY = 100
ROAD_SPAN_CELLS = 70
NUM_EW_LANES = 2
NUM_NS_LANES = 2
TOTAL_NUM_LANES = NUM_EW_LANES + NUM_NS_LANES

# Lane indices
EW_LANE_W_E = 0  # West to East
EW_LANE_E_W = 1  # East to West
NS_LANE_N_S = 2  # North to South
NS_LANE_S_N = 3  # South to North

LANE_VISUAL_THICKNESS_CELLS = 8  # Make roads much wider (was 4) to fit larger cars
CAR_LENGTH_CELLS = 3  # Make cars longer (was 2)
CAR_WIDTH_CELLS = 1.5  # Make cars wider
CAR_VISUAL_THICKNESS_PIXELS = CELL_SIZE * 0.8
CAR_COLORS = ["#E74C3C", "#3498DB", "#2ECC71", "#F1C40F", "#9B59B6", "#34495E", "#FF7F50", "#ADFF2F"]

# --- LWR Model Parameters ---
FREE_FLOW_SPEED = 5.0  # Maximum speed (cells/step)
JAM_DENSITY = 0.5      # Maximum density (cars/cell)
MIN_FOLLOWING_DISTANCE = CAR_LENGTH_CELLS + 2.0  # Minimum distance = car length + 2 cells (was +1)
SAFE_FOLLOWING_DISTANCE = CAR_LENGTH_CELLS + 4.0  # Safe distance = car length + 4 cells (was +2)

class LWRCarFixed:
    def __init__(self, car_id, position, lane_index, color=None):
        self.id = car_id
        self.position = position
        self.lane_index = lane_index
        self.color = color if color else random.choice(CAR_COLORS)
        self.canvas_item_body = None
        self.canvas_item_text = None
        
        # LWR properties
        self.speed = 0.0
        self.local_density = 0.0
        self.desired_speed = FREE_FLOW_SPEED
        self.max_speed = FREE_FLOW_SPEED

    def calculate_safe_speed(self, distance_to_next_car, next_car_speed=0):
        """Calculate safe speed based on distance to next car (car-following model)"""
        if distance_to_next_car <= MIN_FOLLOWING_DISTANCE:
            return 0.0  # Stop if too close
        elif distance_to_next_car <= SAFE_FOLLOWING_DISTANCE:
            # Slow down proportionally
            return min(next_car_speed, distance_to_next_car - MIN_FOLLOWING_DISTANCE)
        else:
            # Can go at desired speed
            return self.desired_speed

    def calculate_lwr_speed(self, local_density, distance_to_next_car=float('inf'), next_car_speed=0):
        """Calculate speed based on both LWR and car-following"""
        # LWR speed based on density
        if local_density >= JAM_DENSITY:
            lwr_speed = 0.0
        else:
            lwr_speed = FREE_FLOW_SPEED * (1.0 - local_density / JAM_DENSITY)
        
        # Car-following speed based on distance to next car
        safe_speed = self.calculate_safe_speed(distance_to_next_car, next_car_speed)
        
        # Take the minimum of LWR speed and safe speed
        self.speed = max(0.0, min(lwr_speed, safe_speed, self.max_speed))
        self.local_density = local_density
        return self.speed

    def __repr__(self):
        direction = "W->E"
        if self.lane_index == EW_LANE_E_W: direction = "E->W"
        elif self.lane_index == NS_LANE_N_S: direction = "N->S"
        elif self.lane_index == NS_LANE_S_N: direction = "S->N"
        return f"Car({self.id} L{self.lane_index}({direction}) @{self.position:.1f} spd:{self.speed:.1f})"

class TrafficSimulationLWRFixed:
    def __init__(self, master_window):
        self.master = master_window
        master_window.title("Traffic Simulation - LWR + Car Following Model")

        # --- Core simulation state ---
        self.cars_in_lane = [[] for _ in range(TOTAL_NUM_LANES)]
        self.next_car_id_counter = 0
        self.step_count = 0
        self.cars_exited_this_step = [0] * TOTAL_NUM_LANES

        # --- LWR intersection model ---
        self.intersection_capacity = 0.2  # Lower capacity for more realistic intersection
        self.intersection_center = ROAD_SPAN_CELLS // 2
        self.intersection_zone = range(self.intersection_center - 4, self.intersection_center + 5)  # Larger intersection zone

        # --- UI Controls (keeping your design) ---
        self.inflow_rate_vars = [
            tk.DoubleVar(value=0.15), tk.DoubleVar(value=0.15),
            tk.DoubleVar(value=0.1), tk.DoubleVar(value=0.1)
        ]
        self.global_max_speed_var = tk.DoubleVar(value=FREE_FLOW_SPEED)
        self.jam_density_var = tk.DoubleVar(value=JAM_DENSITY)

        # --- Display variables (keeping your format) ---
        self.sim_step_display = tk.StringVar(value="Step: 0")
        self.num_cars_display = [tk.StringVar(value=f"Cars L{i}: 0") for i in range(TOTAL_NUM_LANES)]
        self.avg_speed_display = [tk.StringVar(value=f"Avg Spd L{i}: 0.00") for i in range(TOTAL_NUM_LANES)]
        self.avg_density_display = [tk.StringVar(value=f"Avg Density L{i}: 0.00") for i in range(TOTAL_NUM_LANES)]
        self.flow_rate_display = [tk.StringVar(value=f"Flow L{i}: 0") for i in range(TOTAL_NUM_LANES)]
        self.total_cars_display = tk.StringVar(value="Total Cars: 0")
        self.intersection_status_display = tk.StringVar(value="Intersection: FREE")

        # --- Setup UI (keeping your layout) ---
        self.setup_ui()
        self.is_running = False
        self.update_displays()

    def setup_ui(self):
        # Main layout (exactly like yours)
        app_frame = tk.Frame(self.master)
        app_frame.pack(fill=tk.BOTH, expand=True)

        # Left panel for controls
        left_panel = tk.Frame(app_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Parameters
        param_frame = tk.LabelFrame(left_panel, text="LWR + Car Following Model", padx=10, pady=10)
        param_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        self.setup_parameter_controls(param_frame)

        # Statistics
        stats_frame = tk.LabelFrame(left_panel, text="Live Statistics", padx=10, pady=10)
        stats_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        self.setup_stats_display(stats_frame)

        # Canvas (same dimensions and colors)
        self.road_visual_span_pixels = ROAD_SPAN_CELLS * CELL_SIZE
        self.lane_visual_thickness_pixels = LANE_VISUAL_THICKNESS_CELLS * CELL_SIZE
        
        self.canvas = tk.Canvas(app_frame, 
                               width=self.road_visual_span_pixels, 
                               height=self.road_visual_span_pixels + 100, 
                               bg="#EAEAEA")
        self.canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Lane positions (updated for wider roads)
        road_center = self.road_visual_span_pixels / 2
        lane_width = self.lane_visual_thickness_pixels / 4  # Quarter of road width per lane
        
        # East-West lanes (horizontal) - positioned above and below center line
        self.ew_lane_center_y = [
            road_center - lane_width,  # Top lane (W->E)
            road_center + lane_width   # Bottom lane (E->W)
        ]
        
        # North-South lanes (vertical) - positioned left and right of center line
        self.ns_lane_center_x = [
            road_center - lane_width,  # Left lane (N->S)
            road_center + lane_width   # Right lane (S->N)
        ]

        self.draw_road_background()

    def setup_parameter_controls(self, parent_frame):
        row_idx = 0
        lane_labels = ["Inflow L0 (W->E):", "Inflow L1 (E->W):", "Inflow L2 (N->S):", "Inflow L3 (S->N):"]
        
        for i in range(TOTAL_NUM_LANES):
            tk.Label(parent_frame, text=lane_labels[i]).grid(row=row_idx, column=0, sticky=tk.W, padx=5)
            tk.Scale(parent_frame, from_=0.0, to=0.5, resolution=0.05, orient=tk.HORIZONTAL,
                     variable=self.inflow_rate_vars[i], length=180).grid(row=row_idx, column=1, sticky=tk.EW, pady=2)
            row_idx += 1
        
        # LWR Parameters
        tk.Label(parent_frame, text="Free Flow Speed:").grid(row=row_idx, column=0, sticky=tk.W, padx=5)
        tk.Scale(parent_frame, from_=1.0, to=8.0, resolution=0.5, orient=tk.HORIZONTAL,
                 variable=self.global_max_speed_var, length=180).grid(row=row_idx, column=1, sticky=tk.EW, pady=2)
        row_idx += 1
        
        tk.Label(parent_frame, text="Jam Density:").grid(row=row_idx, column=0, sticky=tk.W, padx=5)
        tk.Scale(parent_frame, from_=0.1, to=1.0, resolution=0.05, orient=tk.HORIZONTAL,
                 variable=self.jam_density_var, length=180).grid(row=row_idx, column=1, sticky=tk.EW, pady=2)
        row_idx += 1
        
        parent_frame.grid_columnconfigure(1, weight=1)
        
        # Buttons (same style)
        button_frame = tk.Frame(parent_frame)
        button_frame.grid(row=row_idx, column=0, columnspan=2, pady=10, sticky="ew")
        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=1)

        self.start_button = tk.Button(button_frame, text="Start/Pause", command=self.toggle_simulation, width=12)
        self.start_button.grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        self.reset_button = tk.Button(button_frame, text="Reset", command=self.reset_simulation, width=12)
        self.reset_button.grid(row=0, column=1, sticky="ew", padx=2, pady=2)

    def setup_stats_display(self, parent_frame):
        tk.Label(parent_frame, textvariable=self.sim_step_display, anchor="w").pack(fill="x")
        tk.Label(parent_frame, textvariable=self.total_cars_display, anchor="w").pack(fill="x")
        tk.Label(parent_frame, textvariable=self.intersection_status_display, anchor="w").pack(fill="x")
        
        lane_long_desc = ["L0 (W->E)", "L1 (E->W)", "L2 (N->S)", "L3 (S->N)"]
        for i in range(TOTAL_NUM_LANES):
            tk.Label(parent_frame, textvariable=self.num_cars_display[i], anchor="w").pack(fill="x")
            self.num_cars_display[i].set(f"Cars {lane_long_desc[i]}: 0")
            tk.Label(parent_frame, textvariable=self.avg_speed_display[i], anchor="w").pack(fill="x")
            tk.Label(parent_frame, textvariable=self.avg_density_display[i], anchor="w").pack(fill="x")
            tk.Label(parent_frame, textvariable=self.flow_rate_display[i], anchor="w").pack(fill="x")

    def draw_road_background(self):
        """Draw wider roads with white dotted lane dividers and city aesthetics"""
        self.canvas.delete("background_items")
        
        # First draw the city background
        self.draw_city_background()
        
        # Then draw roads on top
        road_color = "#404040"  # Darker road color for realism
        divider_color = "white"
        
        road_center = self.road_visual_span_pixels / 2
        road_half_width = self.lane_visual_thickness_pixels / 2
        
        # Draw horizontal road (East-West) with rounded edges
        ew_road_top = road_center - road_half_width
        ew_road_bottom = road_center + road_half_width
        self.canvas.create_rectangle(0, ew_road_top, self.road_visual_span_pixels, ew_road_bottom,
                                     fill=road_color, outline="#303030", width=2, tags="background_items")
        
        # Draw vertical road (North-South) with rounded edges
        ns_road_left = road_center - road_half_width
        ns_road_right = road_center + road_half_width
        self.canvas.create_rectangle(ns_road_left, 0, ns_road_right, self.road_visual_span_pixels,
                                     fill=road_color, outline="#303030", width=2, tags="background_items")
        
        # Draw white dotted lane dividers
        self.draw_lane_dividers(road_center, divider_color)
        
        # Draw crosswalks
        self.draw_crosswalks(road_center, road_half_width)
        
        # Draw intersection outline (traffic light area)
        ix1 = (self.intersection_center - 4) * CELL_SIZE
        iy1 = (self.intersection_center - 4) * CELL_SIZE
        ix2 = (self.intersection_center + 4) * CELL_SIZE
        iy2 = (self.intersection_center + 4) * CELL_SIZE
        self.canvas.create_rectangle(ix1, iy1, ix2, iy2, outline="#FFD700", width=3, tags="background_items")
        
        # Add traffic lights
        self.draw_traffic_lights(road_center)

    def draw_city_background(self):
        """Draw buildings, trees, and city elements"""
        # Background color (sky/grass)
        self.canvas.create_rectangle(0, 0, self.road_visual_span_pixels, self.road_visual_span_pixels,
                                     fill="#87CEEB", outline="", tags="background_items")  # Sky blue
        
        # Draw buildings in each quadrant
        self.draw_buildings_quadrant(0, 0, "northwest")
        self.draw_buildings_quadrant(self.road_visual_span_pixels//2 + 50, 0, "northeast")
        self.draw_buildings_quadrant(0, self.road_visual_span_pixels//2 + 50, "southwest")
        self.draw_buildings_quadrant(self.road_visual_span_pixels//2 + 50, self.road_visual_span_pixels//2 + 50, "southeast")
        
        # Add some trees and parks
        self.draw_trees_and_parks()

    def draw_buildings_quadrant(self, start_x, start_y, quadrant):
        """Draw buildings in a quadrant"""
        building_colors = ["#8B4513", "#A0522D", "#CD853F", "#D2691E", "#B22222", "#696969", "#708090"]
        
        # Create 3-4 buildings per quadrant
        buildings = [
            {"x": start_x + 20, "y": start_y + 20, "w": 80, "h": 120, "color": random.choice(building_colors)},
            {"x": start_x + 120, "y": start_y + 30, "w": 60, "h": 100, "color": random.choice(building_colors)},
            {"x": start_x + 200, "y": start_y + 10, "w": 90, "h": 140, "color": random.choice(building_colors)},
            {"x": start_x + 40, "y": start_y + 160, "w": 70, "h": 80, "color": random.choice(building_colors)},
        ]
        
        for building in buildings:
            # Main building
            self.canvas.create_rectangle(building["x"], building["y"], 
                                       building["x"] + building["w"], building["y"] + building["h"],
                                       fill=building["color"], outline="#000000", width=1, tags="background_items")
            
            # Add windows
            self.draw_windows(building["x"], building["y"], building["w"], building["h"])

    def draw_windows(self, x, y, width, height):
        """Draw windows on buildings"""
        window_color = "#FFFF99"  # Light yellow
        window_size = 8
        spacing = 15
        
        for row in range(y + 10, y + height - 10, spacing):
            for col in range(x + 10, x + width - 10, spacing):
                if random.random() > 0.3:  # 70% chance of window being lit
                    self.canvas.create_rectangle(col, row, col + window_size, row + window_size,
                                               fill=window_color, outline="#000000", tags="background_items")

    def draw_trees_and_parks(self):
        """Add trees and green spaces - positioned to avoid buildings"""
        tree_positions = [
            # Top area - between buildings
            (300, 50), (320, 80), (340, 60),  # Top center area
            # Bottom area - between buildings  
            (300, 500), (320, 520), (340, 480),  # Bottom center area
            # Right area - in open spaces
            (500, 200), (520, 280), (480, 320),  # Right side open areas
            # Left area - in open spaces
            (50, 250), (80, 320), (60, 380),  # Left side open areas
            # Additional trees in safe zones
            (150, 300), (450, 150), (150, 450), (450, 450),  # Corner safe zones
        ]
        
        for x, y in tree_positions:
            # Only draw tree if it's not too close to roads
            road_center = self.road_visual_span_pixels / 2
            road_half_width = self.lane_visual_thickness_pixels / 2
            
            # Check if tree is far enough from roads
            if (abs(x - road_center) > road_half_width + 30 and 
                abs(y - road_center) > road_half_width + 30):
                # Tree trunk
                self.canvas.create_rectangle(x-3, y+10, x+3, y+25, fill="#8B4513", tags="background_items")
                # Tree foliage
                self.canvas.create_oval(x-12, y-5, x+12, y+15, fill="#228B22", outline="#006400", tags="background_items")

    def draw_crosswalks(self, road_center, road_half_width):
        """Draw pedestrian crosswalks"""
        crosswalk_color = "white"
        stripe_width = 4
        stripe_spacing = 6
        
        # Horizontal crosswalks (north and south of intersection)
        intersection_left = road_center - road_half_width
        intersection_right = road_center + road_half_width
        
        # North crosswalk
        crosswalk_y = (self.intersection_center - 4) * CELL_SIZE - 15
        for i in range(int(intersection_left), int(intersection_right), stripe_spacing):
            self.canvas.create_rectangle(i, crosswalk_y, i + stripe_width, crosswalk_y + 10,
                                       fill=crosswalk_color, tags="background_items")
        
        # South crosswalk
        crosswalk_y = (self.intersection_center + 4) * CELL_SIZE + 5
        for i in range(int(intersection_left), int(intersection_right), stripe_spacing):
            self.canvas.create_rectangle(i, crosswalk_y, i + stripe_width, crosswalk_y + 10,
                                       fill=crosswalk_color, tags="background_items")
        
        # Vertical crosswalks (east and west of intersection)
        intersection_top = road_center - road_half_width
        intersection_bottom = road_center + road_half_width
        
        # West crosswalk
        crosswalk_x = (self.intersection_center - 4) * CELL_SIZE - 15
        for i in range(int(intersection_top), int(intersection_bottom), stripe_spacing):
            self.canvas.create_rectangle(crosswalk_x, i, crosswalk_x + 10, i + stripe_width,
                                       fill=crosswalk_color, tags="background_items")
        
        # East crosswalk
        crosswalk_x = (self.intersection_center + 4) * CELL_SIZE + 5
        for i in range(int(intersection_top), int(intersection_bottom), stripe_spacing):
            self.canvas.create_rectangle(crosswalk_x, i, crosswalk_x + 10, i + stripe_width,
                                       fill=crosswalk_color, tags="background_items")

    def draw_traffic_lights(self, road_center):
        """Draw traffic lights at intersection corners - ALL GREEN"""
        light_positions = [
            (road_center - 60, road_center - 60),  # Northwest
            (road_center + 60, road_center - 60),  # Northeast
            (road_center - 60, road_center + 60),  # Southwest
            (road_center + 60, road_center + 60),  # Southeast
        ]
        
        for x, y in light_positions:
            # Traffic light pole
            self.canvas.create_rectangle(x-2, y, x+2, y+40, fill="#404040", tags="background_items")
            # Traffic light box
            self.canvas.create_rectangle(x-8, y-15, x+8, y+5, fill="#000000", outline="#404040", tags="background_items")
            # ALL LIGHTS GREEN for smooth traffic flow
            self.canvas.create_oval(x-5, y-12, x+5, y-2, fill="#00FF00", tags="background_items")

    def draw_lane_dividers(self, road_center, divider_color):
        """Draw white dotted lines to separate lanes"""
        dash_length = 8
        gap_length = 6
        line_width = 2
        
        # Horizontal lane divider (separates W->E from E->W)
        y_divider = road_center
        x = 0
        while x < self.road_visual_span_pixels:
            # Skip intersection area (larger intersection)
            if not (self.intersection_center - 4) * CELL_SIZE <= x <= (self.intersection_center + 4) * CELL_SIZE:
                self.canvas.create_line(x, y_divider, x + dash_length, y_divider,
                                       fill=divider_color, width=line_width, tags="background_items")
            x += dash_length + gap_length
        
        # Vertical lane divider (separates N->S from S->N)
        x_divider = road_center
        y = 0
        while y < self.road_visual_span_pixels:
            # Skip intersection area (larger intersection)
            if not (self.intersection_center - 4) * CELL_SIZE <= y <= (self.intersection_center + 4) * CELL_SIZE:
                self.canvas.create_line(x_divider, y, x_divider, y + dash_length,
                                       fill=divider_color, width=line_width, tags="background_items")
            y += dash_length + gap_length

    def find_next_car_ahead(self, car, cars_in_lane):
        """Find the next car ahead in the same lane"""
        next_car = None
        min_distance = float('inf')
        
        for other_car in cars_in_lane:
            if other_car.id != car.id and other_car.position > car.position:
                distance = other_car.position - car.position
                if distance < min_distance:
                    min_distance = distance
                    next_car = other_car
        
        return next_car, min_distance

    def calculate_local_density(self, car, cars_in_lane):
        """Calculate local density around a car"""
        density_count = 0
        search_range = 8  # Look ahead/behind 8 cells
        
        for other_car in cars_in_lane:
            if other_car.id != car.id:
                distance = abs(other_car.position - car.position)
                if distance <= search_range:
                    density_count += 1
        
        local_density = density_count / (2 * search_range)
        return min(local_density, JAM_DENSITY)

    def calculate_intersection_density(self):
        """Calculate density in intersection zone"""
        cars_in_intersection = 0
        for lane_cars in self.cars_in_lane:
            for car in lane_cars:
                if int(car.position) in self.intersection_zone:
                    cars_in_intersection += 1
        
        intersection_area = len(self.intersection_zone)
        return cars_in_intersection / intersection_area

    def spawn_cars_realistic(self):
        """Spawn cars with realistic spacing"""
        global FREE_FLOW_SPEED, JAM_DENSITY
        FREE_FLOW_SPEED = self.global_max_speed_var.get()
        JAM_DENSITY = self.jam_density_var.get()
        
        for lane_idx in range(TOTAL_NUM_LANES):
            if random.random() < self.inflow_rate_vars[lane_idx].get():
                # Check if entry has enough space (realistic spacing for larger cars)
                can_spawn = True
                for car in self.cars_in_lane[lane_idx]:
                    # Need more space for larger cars
                    if car.position < SAFE_FOLLOWING_DISTANCE + CAR_LENGTH_CELLS:
                        can_spawn = False
                        break
                
                if can_spawn:
                    new_car = LWRCarFixed(f"c{self.next_car_id_counter}", 0.0, lane_idx)
                    new_car.max_speed = FREE_FLOW_SPEED
                    self.cars_in_lane[lane_idx].append(new_car)
                    self.next_car_id_counter += 1

    def check_intersection_collision(self, proposed_position, lane_idx, car_id):
        """Check if moving to proposed position would cause collision in intersection"""
        # Only check if the proposed position is in the intersection
        if int(proposed_position) not in self.intersection_zone:
            return False  # No collision if not in intersection
        
        # Check against cars from ALL lanes (not just same lane)
        for other_lane_idx in range(TOTAL_NUM_LANES):
            for other_car in self.cars_in_lane[other_lane_idx]:
                if other_car.id != car_id:  # Don't check against self
                    # Check if other car is also in intersection
                    if int(other_car.position) in self.intersection_zone:
                        # Calculate distance between cars in intersection
                        distance = abs(proposed_position - other_car.position)
                        
                        # For intersection, use stricter spacing
                        if distance < CAR_LENGTH_CELLS + 1:  # Cars need to be further apart in intersection
                            return True  # Collision detected
        
        return False  # No collision

    def move_cars_realistic(self):
        """Move cars with realistic car-following and LWR physics - FIXED COLLISION DETECTION"""
        intersection_density = self.calculate_intersection_density()
        
        for lane_idx in range(TOTAL_NUM_LANES):
            cars_to_remove = []
            self.cars_exited_this_step[lane_idx] = 0
            
            # Sort cars by position to process from front to back
            self.cars_in_lane[lane_idx].sort(key=lambda c: c.position, reverse=True)
            
            for car in self.cars_in_lane[lane_idx][:]:
                # Calculate local density
                local_density = self.calculate_local_density(car, self.cars_in_lane[lane_idx])
                
                # Find next car ahead
                next_car, distance_to_next = self.find_next_car_ahead(car, self.cars_in_lane[lane_idx])
                next_car_speed = next_car.speed if next_car else 0
                
                # Special handling for intersection
                if int(car.position) in self.intersection_zone:
                    if intersection_density > self.intersection_capacity:
                        local_density = max(local_density, JAM_DENSITY * 0.8)
                
                # Calculate realistic speed (LWR + car-following)
                car.calculate_lwr_speed(local_density, distance_to_next, next_car_speed)
                
                # COLLISION PREVENTION: Check both same-lane and intersection collisions
                proposed_new_position = car.position + car.speed
                can_move = True
                
                # Check against cars in the same lane
                for other_car in self.cars_in_lane[lane_idx]:
                    if other_car.id != car.id:
                        distance_after_move = abs(proposed_new_position - other_car.position)
                        if distance_after_move < MIN_FOLLOWING_DISTANCE:
                            can_move = False
                            break
                
                # ADDITIONAL CHECK: Intersection collision with cars from other lanes
                if can_move and self.check_intersection_collision(proposed_new_position, lane_idx, car.id):
                    can_move = False
                
                # Only move if safe
                if can_move:
                    car.position = proposed_new_position
                else:
                    # Stop the car if it can't move safely
                    car.speed = 0
                
                # Remove car if it exits (account for car length)
                if car.position > ROAD_SPAN_CELLS + CAR_LENGTH_CELLS:
                    cars_to_remove.append(car)
                    self.cars_exited_this_step[lane_idx] += 1
            
            # Remove exited cars
            for car_to_remove in cars_to_remove:
                if car_to_remove.canvas_item_body:
                    self.canvas.delete(car_to_remove.canvas_item_body)
                if car_to_remove.canvas_item_text:
                    self.canvas.delete(car_to_remove.canvas_item_text)
                self.cars_in_lane[lane_idx].remove(car_to_remove)

    def draw_cars(self):
        """Draw cars with your exact visual style - LARGER CARS"""
        self.canvas.delete('car')
        
        for lane_idx in range(TOTAL_NUM_LANES):
            for car in self.cars_in_lane[lane_idx]:
                # Calculate position (same logic as yours)
                if lane_idx == EW_LANE_W_E:  # West to East
                    x = car.position * CELL_SIZE
                    y = self.ew_lane_center_y[0]
                    # Make car longer horizontally
                    car_width = CAR_LENGTH_CELLS * CELL_SIZE
                    car_height = CAR_WIDTH_CELLS * CELL_SIZE
                elif lane_idx == EW_LANE_E_W:  # East to West
                    x = (ROAD_SPAN_CELLS - car.position) * CELL_SIZE - (CAR_LENGTH_CELLS * CELL_SIZE)
                    y = self.ew_lane_center_y[1]
                    # Make car longer horizontally
                    car_width = CAR_LENGTH_CELLS * CELL_SIZE
                    car_height = CAR_WIDTH_CELLS * CELL_SIZE
                elif lane_idx == NS_LANE_N_S:  # North to South
                    x = self.ns_lane_center_x[0]
                    y = car.position * CELL_SIZE
                    # Make car longer vertically
                    car_width = CAR_WIDTH_CELLS * CELL_SIZE
                    car_height = CAR_LENGTH_CELLS * CELL_SIZE
                else:  # NS_LANE_S_N - South to North
                    x = self.ns_lane_center_x[1]
                    y = (ROAD_SPAN_CELLS - car.position) * CELL_SIZE - (CAR_LENGTH_CELLS * CELL_SIZE)
                    # Make car longer vertically
                    car_width = CAR_WIDTH_CELLS * CELL_SIZE
                    car_height = CAR_LENGTH_CELLS * CELL_SIZE
                
                # Draw LARGER car rectangle
                car.canvas_item_body = self.canvas.create_rectangle(
                    x, y, x + car_width, y + car_height, 
                    fill=car.color, tags='car'
                )
                
                # Show INTEGER speed (centered in larger car)
                speed_display = int(round(car.speed))
                car.canvas_item_text = self.canvas.create_text(
                    x + car_width/2, y + car_height/2, 
                    text=str(speed_display), fill="white", 
                    font=("Arial", 10, "bold"), tags='car'  # Slightly larger font
                )

    def update_displays(self):
        """Update displays with realistic statistics"""
        self.sim_step_display.set(f"Step: {self.step_count}")
        total_cars = sum(len(lane) for lane in self.cars_in_lane)
        self.total_cars_display.set(f"Total Cars: {total_cars}")
        
        # Intersection status
        intersection_density = self.calculate_intersection_density()
        if intersection_density > self.intersection_capacity:
            self.intersection_status_display.set(f"Intersection: CONGESTED (ρ={intersection_density:.2f})")
        else:
            self.intersection_status_display.set(f"Intersection: FREE (ρ={intersection_density:.2f})")
        
        # Lane statistics
        lane_desc = ["L0(W->E)", "L1(E->W)", "L2(N->S)", "L3(S->N)"]
        for i in range(TOTAL_NUM_LANES):
            cars_in_this_lane = self.cars_in_lane[i]
            self.num_cars_display[i].set(f"Cars {lane_desc[i]}: {len(cars_in_this_lane)}")
            
            if cars_in_this_lane:
                avg_speed = sum(car.speed for car in cars_in_this_lane) / len(cars_in_this_lane)
                avg_density = sum(car.local_density for car in cars_in_this_lane) / len(cars_in_this_lane)
                self.avg_speed_display[i].set(f"AvgSpd {lane_desc[i]}: {avg_speed:.2f}")
                self.avg_density_display[i].set(f"AvgDens {lane_desc[i]}: {avg_density:.2f}")
            else:
                self.avg_speed_display[i].set(f"AvgSpd {lane_desc[i]}: 0.00")
                self.avg_density_display[i].set(f"AvgDens {lane_desc[i]}: 0.00")
            
            flow_rate = self.cars_exited_this_step[i]
            self.flow_rate_display[i].set(f"Flow {lane_desc[i]}: {flow_rate}")

    def simulation_step(self):
        """Main simulation step with realistic physics"""
        if not self.is_running:
            return
        
        self.step_count += 1
        self.spawn_cars_realistic()
        self.move_cars_realistic()
        self.draw_cars()
        self.update_displays()
        
        self.master.after(SIMULATION_DELAY, self.simulation_step)

    def toggle_simulation(self):
        """Start/pause simulation"""
        self.is_running = not self.is_running
        if self.is_running:
            self.start_button.config(text="Pause")
            self.simulation_step()
        else:
            self.start_button.config(text="Start")

    def reset_simulation(self):
        """Reset simulation"""
        self.is_running = False
        self.start_button.config(text="Start")
        
        # Clear all cars
        for lane_idx in range(TOTAL_NUM_LANES):
            for car in self.cars_in_lane[lane_idx]:
                if car.canvas_item_body:
                    self.canvas.delete(car.canvas_item_body)
                if car.canvas_item_text:
                    self.canvas.delete(car.canvas_item_text)
            self.cars_in_lane[lane_idx] = []
            self.cars_exited_this_step[lane_idx] = 0
        
        self.next_car_id_counter = 0
        self.step_count = 0
        
        self.canvas.delete("all")
        self.draw_road_background()
        self.update_displays()
        print("Realistic LWR Traffic Simulation Reset.")

if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficSimulationLWRFixed(root)
    root.mainloop() 