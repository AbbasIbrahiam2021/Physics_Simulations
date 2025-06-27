"""
2-D Perpetual Bending Light Simulation
We integrate light rays using the first-order weak-field Schwarzschild geodesic (improving over Newtonian),
including the proper GR deflection term. We also compute gravitational red/blueshift using the Schwarzschild metric.
References:
- In the weak-field limit, a light ray deflection by mass M at impact b is θ = 2GM/(b c^2) (Newtonian, half Einstein's)
  Einstein’s GR doubles the deflection, so we apply a factor ~2 in the photon acceleration.
- Schwarzschild gravitational redshift: (1+z) = 1/√(1 - 2GM/(r c^2))
- Particle roughly follows a geodisic on the Schwarzschild metric as it includes the "first term"
- Schwarzschild gravitational redshift: (1+z) = 1/√(1 - 2GM/(r c^2))

Physical Laws
-We ideally want to solve the geodisic that satisfies the Schwarzschild metric
-Since this is too computationally complex, we use the Runge-Kutta method doesn't
require much coding or computational power

Aesthetic liberties taken
- Stars keep colored cores + white halos which is technically incorrect
- Photon path persists until wrap & resets
    -Photon acceleration is set to have a minimum value to ensure its always moving
    -It's not physically accurate as the photon should pause, but it makes the simulation ru
     better if a>0

"""
import tkinter as tk
from PIL import Image, ImageDraw, ImageTk
import random, math

# Simulation constants
WIDTH, HEIGHT   = 900, 600
C                = 12.0        # effective light speed (px/frame)
DT               = 0.04        # base time step (s)
G                = 120.0       # gravitational constant scale
STAR_COUNT       = 4           # Number of stars (Only up to 8)
MIN_STAR_DIST    = 100         # Minimum spacing between stars
HALO_RES         = 64          # Halo resolution for star glow
FPS_MS           = int(DT*1000)

# Star archetypes (mass and radius ranges, and core color)
ARCH = [
    (12,18,20,26,'red'),        # red giant
    (10,15,16,22,'cyan'),       # blue giant
    ( 8,12,13,18,'yellow'),     # yellow dwarf
    ( 5,10, 9,14,'lightgray'),  # white dwarf
    (15,25, 8,12,'magenta'),    # red dwarf
    (25,35, 6,10,'sienna'),     # brown dwarf
]

# For aesthestics
def draw_starfield(canvas, num_stars=600):
    """
    Paints a randomized static starfield of small white dots.
    Random size gives an impression of distance
    These do not contribute to the ODEs, just for aesthetics
    """
    for _ in range(num_stars):
        x = random.uniform(0, WIDTH)
        y = random.uniform(0, HEIGHT)
        size = random.choice((1, 2, 3))
        canvas.create_oval(x, y, x+size, y+size, fill="white", outline="")


class Star:
    def __init__(self, x, y, mass, radius, color):
        self.x, self.y    = x, y
        self.mass         = mass
        self.radius       = radius
        self.color        = color

    def accel(self, px, py, vx, vy):
        """Compute photon acceleration due to this star"""
        dx, dy = px - self.x, py - self.y
        r2 = dx*dx + dy*dy or 1e-6
        r = math.sqrt(r2)
        if r < self.radius:
            # Inside star: no acceleration (photon is blocked/absorbed)
            return 0.0, 0.0
        # Weak-field Schwarzschild geodesic acceleration:
        # Newtonian term (-GM/r^2) plus first-order correction (angular momentum term)
        # Photon deflection is doubled (factor ~2)
        # Following derivation for null geodesics:
        ax = -G * self.mass * (dx + (dx*vy*vy + 2*dy*vx*vy)/(C*C)) / (r2 * r)
        ay = -G * self.mass * (dy + (dy*vx*vx + 2*dx*vx*vy)/(C*C)) / (r2 * r)
        return ax, ay

    def draw(self, canvas, halo_img):
        # Draw star with a glowing halo and colored core
        canvas.create_image(self.x, self.y, image=halo_img, tags=("star",))
        r = self.radius
        canvas.create_oval(
            self.x-r, self.y-r, self.x+r, self.y+r,
            fill=self.color, outline="", tags=("star",)
        )

# Cache for halo images
halo_cache = {}
def get_halo(r):
    key = round(r*1.5)
    if key in halo_cache:
        return halo_cache[key]
    img = Image.new("RGBA", (HALO_RES, HALO_RES), (0,0,0,0))
    d   = ImageDraw.Draw(img)
    cx = cy = HALO_RES/2
    for rad in range(int(cx), 0, -1):
        alpha = int(200 * (1 - (rad/cx)**1.5))
        d.ellipse((cx-rad, cy-rad, cx+rad, cy+rad), fill=(255,255,255,alpha))
    sprite = ImageTk.PhotoImage(
        img.resize((int(r*8), int(r*8)), Image.LANCZOS)
    )
    halo_cache[key] = sprite
    return sprite

class Photon:
    def __init__(self, x, y, vx, vy):
        self.x, self.y   = x, y
        self.vx, self.vy = vx, vy
        self.id         = None

    def draw(self, canvas):
        # Draw photon as a small white point
        if self.id:
            canvas.delete(self.id)
        self.id = canvas.create_oval(self.x-2, self.y-2, self.x+2, self.y+2, fill="white", tags=("photon",))

    def rk4(self, stars, dt):
        """Integrate photon position and velocity using RK4."""
        def deriv(state):
            x, y, vx, vy = state
            ax = ay = 0.0
            for st in stars:
                dax, day = st.accel(x, y, vx, vy)
                ax += dax; ay += day
            return vx, vy, ax, ay
        s = (self.x, self.y, self.vx, self.vy)
        # RK4 routine, compute derivatives
        # Not computationally demanding
        k1 = deriv(s)
        k2 = deriv((s[0]+0.5*dt*k1[0], s[1]+0.5*dt*k1[1],
                    s[2]+0.5*dt*k1[2], s[3]+0.5*dt*k1[3]))
        k3 = deriv((s[0]+0.5*dt*k2[0], s[1]+0.5*dt*k2[1],
                    s[2]+0.5*dt*k2[2], s[3]+0.5*dt*k2[3]))
        k4 = deriv((s[0]+   dt*k3[0], s[1]+   dt*k3[1],
                    s[2]+   dt*k3[2], s[3]+   dt*k3[3]))
        # Compute terms to solve ODEs
        self.x  += dt/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0])
        self.y  += dt/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
        self.vx += dt/6*(k1[2]+2*k2[2]+2*k3[2]+k4[2])
        self.vy += dt/6*(k1[3]+2*k2[3]+2*k3[3]+k4[3])

# Setup Tkinter interface and simulation
root = tk.Tk()
root.title("Gravitational Light Bending (First-Order GR)")
canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, background="black")
canvas.pack()
draw_starfield(canvas, num_stars=800)


def spawn_stars():
    canvas.delete("star")
    stars = []
    while len(stars) < STAR_COUNT:
        m1,m2,r1,r2,col = random.choice(ARCH)
        mass = random.uniform(m1, m2)
        rad  = random.uniform(r1, r2)
        x = random.uniform(60, WIDTH-60)
        y = random.uniform(60, HEIGHT-60)
        if all(math.hypot(x-s.x, y-s.y) > s.radius + rad + MIN_STAR_DIST for s in stars):
            s = Star(x, y, mass, rad, col)
            stars.append(s)
            s.draw(canvas, get_halo(rad))
    return stars

def random_photon_edge():
    side = random.choice(("left","right","top","bottom"))
    if side == "left":
        return Photon(0, random.uniform(0,HEIGHT),  C, random.uniform(-0.5*C,0.5*C))
    if side == "right":
        return Photon(WIDTH, random.uniform(0,HEIGHT), -C, random.uniform(-0.5*C,0.5*C))
    if side == "top":
        return Photon(random.uniform(0,WIDTH), 0, random.uniform(-0.5*C,0.5*C), C)
    # bottom edge
    return Photon(random.uniform(0,WIDTH), HEIGHT, random.uniform(-0.5*C,0.5*C), -C)

# Initialize simulation objects
stars  = spawn_stars()
photon = random_photon_edge()
photon.draw(canvas)
prev_x, prev_y = photon.x, photon.y
tail = []

def update():
    global stars, photon, tail, prev_x, prev_y
    # Find nearest star (dominant potential for this photon)
    nearest = min(stars, key=lambda s: math.hypot(photon.x-s.x, photon.y-s.y))
    dx, dy = photon.x - nearest.x, photon.y - nearest.y
    r = math.hypot(dx, dy)
    # Gravitational time dilation factor for dt:
    # g_tt ≈ 1 - 2GM/(r c^2) as an approximation when expanding Taylor Series
    rs = 2 * G * nearest.mass / (C*C)
    gamma = math.sqrt(max(0.0, 1 - rs/r))
    dt_eff = DT * gamma

    # Integrate motion under all stars
    photon.rk4(stars, dt_eff)

    # Reset if photon leaves screen
    if photon.x<0 or photon.x>WIDTH or photon.y<0 or photon.y>HEIGHT:
        canvas.delete("star"); canvas.delete("photon")
        for seg in tail: canvas.delete(seg["id"])
        tail.clear()
        stars  = spawn_stars()
        photon = random_photon_edge()
        photon.draw(canvas)
        prev_x, prev_y = photon.x, photon.y
        root.after(FPS_MS, update)
        return

    # Compute gravitational red/blueshift colour for tail segment
    dot = photon.vx*dx + photon.vy*dy
    if dot > 0:
        # Photon moving outward (up the potential) for redshift
        # freq ratio = √(g_tt) = √(1 - 2GM/(r c^2)):contentReference[oaicite:3]{index=3}
        ratio = math.sqrt(max(0.0, 1 - 2*G*nearest.mass/(r*C*C)))
        rcol = 255
        gcol = int(255 * ratio)
        bcol = int(255 * ratio)
    else:
        # Photon moving inward (down the potential) for blueshift
        ratio = 1.0 / math.sqrt(max(1e-6, 1 - 2*G*nearest.mass/(r*C*C)))
        rcol = int(min(255, 255/ratio))
        gcol = int(min(255, 255/ratio))
        bcol = 255
    seg_color = f"#{rcol:02x}{gcol:02x}{bcol:02x}"

    # Draw tail segment
    seg_id = canvas.create_line(prev_x, prev_y, photon.x, photon.y, fill=seg_color, width=2)
    tail.append({"id": seg_id})
    prev_x, prev_y = photon.x, photon.y

    # Draw photon as a beam of white light
    photon.draw(canvas)
    root.after(FPS_MS, update)

root.after(FPS_MS, update)
root.mainloop()
