import numpy as np
from numba import njit
import math
from perlin_noise import perlin_noise_2d

# =============================================================================
# REGULAR WAVE PATTERNS
# =============================================================================

@njit
def sine_waves_2d(nx, ny, freq_x=0.1, freq_y=0.1, phase_x=0, phase_y=0, amplitude=1.0):
    """2D sine wave pattern"""
    pattern = np.zeros((ny, nx))
    for y in range(ny):
        for x in range(nx):
            pattern[y, x] = amplitude * (
                np.sin(2 * np.pi * freq_x * x + phase_x) + 
                np.sin(2 * np.pi * freq_y * y + phase_y)
            ) / 2.0
    return (pattern + amplitude) / (2 * amplitude)  # Normalize to [0,1]

@njit
def cosine_waves_2d(nx, ny, freq_x=0.05, freq_y=0.08, phase_x=0, phase_y=0):
    """2D cosine wave pattern"""
    pattern = np.zeros((ny, nx))
    for y in range(ny):
        for x in range(nx):
            pattern[y, x] = (
                np.cos(2 * np.pi * freq_x * x + phase_x) * 
                np.cos(2 * np.pi * freq_y * y + phase_y)
            )
    return (pattern + 1.0) / 2.0

@njit
def radial_waves(nx, ny, center_x=None, center_y=None, frequency=0.05, decay=0.01):
    """Radial wave pattern emanating from center"""
    if center_x is None:
        center_x = nx // 2
    if center_y is None:
        center_y = ny // 2
    
    pattern = np.zeros((ny, nx))
    for y in range(ny):
        for x in range(nx):
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            pattern[y, x] = np.sin(2 * np.pi * frequency * dist) * np.exp(-decay * dist)
    
    return (pattern + 1.0) / 2.0

# =============================================================================
# IRREGULAR AND VARIABLE WAVELENGTH PATTERNS
# =============================================================================

@njit
def variable_frequency_waves(nx, ny, base_freq=0.02, freq_variation=0.8, seed=42):
    """Sine waves with spatially varying frequency"""
    np.random.seed(seed)
    pattern = np.zeros((ny, nx))
    
    for y in range(ny):
        for x in range(nx):
            # Frequency varies smoothly across space
            local_freq_x = base_freq * (1 + freq_variation * np.sin(0.01 * x) * np.cos(0.015 * y))
            local_freq_y = base_freq * (1 + freq_variation * np.cos(0.012 * x) * np.sin(0.008 * y))
            
            pattern[y, x] = (
                np.sin(2 * np.pi * local_freq_x * x) + 
                np.sin(2 * np.pi * local_freq_y * y)
            ) / 2.0
    
    return (pattern + 1.0) / 2.0

@njit
def chirp_pattern(nx, ny, start_freq=0.01, end_freq=0.2, direction=0):
    """Frequency chirp pattern (linearly increasing frequency)"""
    pattern = np.zeros((ny, nx))
    
    for y in range(ny):
        for x in range(nx):
            if direction == 0:  # Horizontal chirp
                t = x / nx
                freq = start_freq + (end_freq - start_freq) * t
                pattern[y, x] = np.sin(2 * np.pi * freq * x)
            else:  # Vertical chirp
                t = y / ny
                freq = start_freq + (end_freq - start_freq) * t
                pattern[y, x] = np.sin(2 * np.pi * freq * y)
    
    return (pattern + 1.0) / 2.0

# =============================================================================
# GEOMETRIC PATTERNS
# =============================================================================

@njit
def checkerboard(nx, ny, tile_size_x=8, tile_size_y=8):
    """Checkerboard pattern"""
    pattern = np.zeros((ny, nx))
    for y in range(ny):
        for x in range(nx):
            if ((x // tile_size_x) + (y // tile_size_y)) % 2 == 0:
                pattern[y, x] = 1.0
    return pattern

@njit
def concentric_circles(nx, ny, center_x=None, center_y=None, ring_width=10):
    """Concentric circles pattern"""
    if center_x is None:
        center_x = nx // 2
    if center_y is None:
        center_y = ny // 2
    
    pattern = np.zeros((ny, nx))
    for y in range(ny):
        for x in range(nx):
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            pattern[y, x] = (int(dist / ring_width) % 2)
    return pattern

@njit
def spiral_pattern(nx, ny, center_x=None, center_y=None, spiral_freq=0.05, radial_freq=0.1):
    """Logarithmic spiral pattern"""
    if center_x is None:
        center_x = nx // 2
    if center_y is None:
        center_y = ny // 2
    
    pattern = np.zeros((ny, nx))
    for y in range(ny):
        for x in range(nx):
            dx, dy = x - center_x, y - center_y
            if dx == 0 and dy == 0:
                angle = 0
            else:
                angle = np.arctan2(dy, dx)
            radius = np.sqrt(dx**2 + dy**2)
            
            spiral_val = np.sin(spiral_freq * radius + radial_freq * angle)
            pattern[y, x] = spiral_val
    
    return (pattern + 1.0) / 2.0

# =============================================================================
# FRACTAL AND COMPLEX PATTERNS
# =============================================================================

@njit
def mandelbrot_set(nx, ny, max_iter=50, x_min=-2.5, x_max=1.5, y_min=-2.0, y_max=2.0):
    """Mandelbrot set visualization"""
    pattern = np.zeros((ny, nx))
    
    for y in range(ny):
        for x in range(nx):
            c_real = x_min + (x_max - x_min) * x / nx
            c_imag = y_min + (y_max - y_min) * y / ny
            
            z_real, z_imag = 0.0, 0.0
            iterations = 0
            
            while z_real**2 + z_imag**2 < 4 and iterations < max_iter:
                temp = z_real**2 - z_imag**2 + c_real
                z_imag = 2 * z_real * z_imag + c_imag
                z_real = temp
                iterations += 1
            
            pattern[y, x] = iterations / max_iter
    
    return pattern

@njit
def julia_set(nx, ny, c_real=-0.4, c_imag=0.6, max_iter=50, zoom=1.0):
    """Julia set visualization"""
    pattern = np.zeros((ny, nx))
    
    for y in range(ny):
        for x in range(nx):
            z_real = (x - nx/2) * 4.0 / (nx * zoom)
            z_imag = (y - ny/2) * 4.0 / (ny * zoom)
            
            iterations = 0
            while z_real**2 + z_imag**2 < 4 and iterations < max_iter:
                temp = z_real**2 - z_imag**2 + c_real
                z_imag = 2 * z_real * z_imag + c_imag
                z_real = temp
                iterations += 1
            
            pattern[y, x] = iterations / max_iter
    
    return pattern

# =============================================================================
# NOISE-BASED PATTERNS
# =============================================================================

def fractal_brownian_motion(nx, ny, octaves=6, persistence=0.5, scale=0.05, seed=42):
    """Multi-octave Perlin noise (Fractal Brownian Motion)"""
    return perlin_noise_2d(nx, ny, scale=scale, octaves=octaves, 
                          persistence=persistence, seed=seed)

def turbulence_pattern(nx, ny, octaves=4, scale=0.08, seed=42):
    """Turbulence pattern using absolute values of noise"""
    noise = perlin_noise_2d(nx, ny, scale=scale, octaves=octaves, seed=seed)
    return np.abs(2 * noise - 1)  # Convert to absolute values

@njit
def white_noise(nx, ny, seed=42):
    """Pure white noise"""
    np.random.seed(seed)
    return np.random.rand(ny, nx)

@njit
def pink_noise(nx, ny, seed=42):
    """Pink noise approximation"""
    np.random.seed(seed)
    noise = np.random.randn(ny, nx)
    
    # Simple pink noise filter approximation
    pattern = np.zeros((ny, nx))
    for y in range(ny):
        for x in range(nx):
            if x > 0:
                pattern[y, x] = 0.7 * pattern[y, x-1] + 0.3 * noise[y, x]
            else:
                pattern[y, x] = noise[y, x]
    
    # Normalize to [0,1]
    min_val, max_val = pattern.min(), pattern.max()
    if max_val != min_val:
        pattern = (pattern - min_val) / (max_val - min_val)
    
    return pattern

# =============================================================================
# CELLULAR AUTOMATA AND DYNAMIC PATTERNS
# =============================================================================

@njit
def cellular_automata(nx, ny, iterations=10, seed=42, rule_threshold=0.5):
    """Simple cellular automaton pattern"""
    np.random.seed(seed)
    pattern = (np.random.rand(ny, nx) > 0.5).astype(np.float64)
    
    for _ in range(iterations):
        new_pattern = np.zeros((ny, nx))
        for y in range(1, ny-1):
            for x in range(1, nx-1):
                neighbors = (pattern[y-1:y+2, x-1:x+2].sum() - pattern[y, x])
                if pattern[y, x] == 1:
                    new_pattern[y, x] = 1 if neighbors >= 2 and neighbors <= 3 else 0
                else:
                    new_pattern[y, x] = 1 if neighbors == 3 else 0
        pattern = new_pattern
    
    return pattern

@njit
def wave_interference(nx, ny, n_sources=5, seed=42):
    """Multiple wave source interference pattern"""
    np.random.seed(seed)
    pattern = np.zeros((ny, nx))
    
    # Generate random wave sources
    sources = []
    for _ in range(n_sources):
        x_src = int(np.random.rand() * nx)
        y_src = int(np.random.rand() * ny)
        freq = 0.02 + np.random.rand() * 0.08
        phase = np.random.rand() * 2 * np.pi
        sources.append((x_src, y_src, freq, phase))
    
    for y in range(ny):
        for x in range(nx):
            wave_sum = 0.0
            for x_src, y_src, freq, phase in sources:
                dist = np.sqrt((x - x_src)**2 + (y - y_src)**2)
                wave_sum += np.sin(2 * np.pi * freq * dist + phase)
            pattern[y, x] = wave_sum / n_sources
    
    return (pattern + 1.0) / 2.0

# =============================================================================
# GRADIENT AND SMOOTH PATTERNS
# =============================================================================

@njit
def linear_gradient(nx, ny, direction=0, start_val=0.0, end_val=1.0):
    """Linear gradient pattern
    direction: 0=horizontal, 1=vertical, 2=diagonal"""
    pattern = np.zeros((ny, nx))
    
    for y in range(ny):
        for x in range(nx):
            if direction == 0:  # Horizontal
                t = x / (nx - 1)
            elif direction == 1:  # Vertical
                t = y / (ny - 1)
            else:  # Diagonal
                t = (x + y) / (nx + ny - 2)
            
            pattern[y, x] = start_val + (end_val - start_val) * t
    
    return pattern

@njit
def radial_gradient(nx, ny, center_x=None, center_y=None, inner_val=0.0, outer_val=1.0):
    """Radial gradient from center"""
    if center_x is None:
        center_x = nx // 2
    if center_y is None:
        center_y = ny // 2
    
    max_dist = np.sqrt(center_x**2 + center_y**2)
    pattern = np.zeros((ny, nx))
    
    for y in range(ny):
        for x in range(nx):
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            t = min(dist / max_dist, 1.0)
            pattern[y, x] = inner_val + (outer_val - inner_val) * t
    
    return pattern

# =============================================================================
# TEXTURE PATTERNS
# =============================================================================

@njit
def voronoi_texture(nx, ny, n_points=20, seed=42):
    """Voronoi diagram texture"""
    np.random.seed(seed)
    
    # Generate random points
    points = []
    for _ in range(n_points):
        x = int(np.random.rand() * nx)
        y = int(np.random.rand() * ny)
        points.append((x, y))
    
    pattern = np.zeros((ny, nx))
    for y in range(ny):
        for x in range(nx):
            min_dist = float('inf')
            closest_idx = 0
            for i, (px, py) in enumerate(points):
                dist = (x - px)**2 + (y - py)**2
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
            pattern[y, x] = closest_idx / (n_points - 1)
    
    return pattern

@njit
def wood_grain(nx, ny, grain_freq=0.03, ring_freq=0.8, noise_amp=0.3, seed=42):
    """Wood grain texture pattern"""
    np.random.seed(seed)
    pattern = np.zeros((ny, nx))
    
    center_x, center_y = nx // 2, ny // 2
    
    for y in range(ny):
        for x in range(nx):
            # Distance from center (tree rings)
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            # Add noise for irregular grain
            noise = (np.random.rand() - 0.5) * noise_amp
            
            # Wood ring pattern
            ring_val = np.sin(ring_freq * dist + noise)
            
            # Grain direction
            angle = np.arctan2(y - center_y, x - center_x)
            grain_val = np.sin(grain_freq * (x + y) + angle + noise)
            
            pattern[y, x] = (ring_val + grain_val) / 2.0
    
    return (pattern + 1.0) / 2.0

# =============================================================================
# MASTER FUNCTION FOR RANDOM PATTERN GENERATION
# =============================================================================

def generate_random_pattern(nx, ny, seed=None):
    """
    Generate a completely random pattern using one of all available generators
    with randomized parameters
    """
    if seed is not None:
        np.random.seed(seed)
    
    # List of all pattern generators with their parameter ranges
    generators = [
        # Regular waves
        lambda: sine_waves_2d(nx, ny, 
                             freq_x=np.random.uniform(0.01, 0.2),
                             freq_y=np.random.uniform(0.01, 0.2),
                             phase_x=np.random.uniform(0, 2*np.pi),
                             phase_y=np.random.uniform(0, 2*np.pi)),
        
        lambda: cosine_waves_2d(nx, ny,
                               freq_x=np.random.uniform(0.01, 0.15),
                               freq_y=np.random.uniform(0.01, 0.15)),
        
        lambda: radial_waves(nx, ny,
                            center_x=np.random.randint(0, nx),
                            center_y=np.random.randint(0, ny),
                            frequency=np.random.uniform(0.01, 0.1),
                            decay=np.random.uniform(0.001, 0.05)),
        
        # Variable frequency patterns
        lambda: variable_frequency_waves(nx, ny,
                                        base_freq=np.random.uniform(0.01, 0.05),
                                        freq_variation=np.random.uniform(0.2, 1.5),
                                        seed=np.random.randint(0, 10000)),
        
        lambda: chirp_pattern(nx, ny,
                             start_freq=np.random.uniform(0.005, 0.02),
                             end_freq=np.random.uniform(0.05, 0.3),
                             direction=np.random.randint(0, 2)),
        
        # Geometric patterns
        lambda: checkerboard(nx, ny,
                            tile_size_x=np.random.randint(2, 32),
                            tile_size_y=np.random.randint(2, 32)),
        
        lambda: concentric_circles(nx, ny,
                                  center_x=np.random.randint(0, nx),
                                  center_y=np.random.randint(0, ny),
                                  ring_width=np.random.randint(3, 25)),
        
        lambda: spiral_pattern(nx, ny,
                              spiral_freq=np.random.uniform(0.01, 0.1),
                              radial_freq=np.random.uniform(0.05, 0.3)),
        
        # Fractals
        lambda: mandelbrot_set(nx, ny,
                              max_iter=np.random.randint(20, 100),
                              x_min=np.random.uniform(-3, -1),
                              x_max=np.random.uniform(0.5, 2),
                              y_min=np.random.uniform(-2.5, -0.5),
                              y_max=np.random.uniform(0.5, 2.5)),
        
        lambda: julia_set(nx, ny,
                         c_real=np.random.uniform(-1, 1),
                         c_imag=np.random.uniform(-1, 1),
                         max_iter=np.random.randint(30, 80),
                         zoom=np.random.uniform(0.5, 3.0)),
        
        # Noise patterns
        lambda: fractal_brownian_motion(nx, ny,
                                       octaves=np.random.randint(3, 8),
                                       persistence=np.random.uniform(0.3, 0.8),
                                       scale=np.random.uniform(0.02, 0.15),
                                       seed=np.random.randint(0, 10000)),
        
        lambda: turbulence_pattern(nx, ny,
                                  octaves=np.random.randint(2, 6),
                                  scale=np.random.uniform(0.03, 0.12),
                                  seed=np.random.randint(0, 10000)),
        
        lambda: white_noise(nx, ny, seed=np.random.randint(0, 10000)),
        
        lambda: pink_noise(nx, ny, seed=np.random.randint(0, 10000)),
        
        # Dynamic patterns
        lambda: cellular_automata(nx, ny,
                                 iterations=np.random.randint(5, 20),
                                 seed=np.random.randint(0, 10000)),
        
        lambda: wave_interference(nx, ny,
                                 n_sources=np.random.randint(3, 8),
                                 seed=np.random.randint(0, 10000)),
        
        # Gradients
        lambda: linear_gradient(nx, ny,
                               direction=np.random.randint(0, 3),
                               start_val=np.random.uniform(0, 0.3),
                               end_val=np.random.uniform(0.7, 1.0)),
        
        lambda: radial_gradient(nx, ny,
                               center_x=np.random.randint(0, nx),
                               center_y=np.random.randint(0, ny),
                               inner_val=np.random.uniform(0, 0.3),
                               outer_val=np.random.uniform(0.7, 1.0)),
        
        # Textures
        lambda: voronoi_texture(nx, ny,
                               n_points=np.random.randint(10, 50),
                               seed=np.random.randint(0, 10000)),
        
        lambda: wood_grain(nx, ny,
                          grain_freq=np.random.uniform(0.01, 0.06),
                          ring_freq=np.random.uniform(0.3, 1.5),
                          noise_amp=np.random.uniform(0.1, 0.5),
                          seed=np.random.randint(0, 10000)),
    ]
    
    # Randomly select and execute a generator
    generator = np.random.choice(generators)
    pattern = generator()
    
    # Ensure the pattern is properly normalized to [0,1]
    min_val, max_val = pattern.min(), pattern.max()
    if max_val != min_val:
        pattern = (pattern - min_val) / (max_val - min_val)
    else:
        pattern = np.full((ny, nx), 0.5)
    
    return pattern

# =============================================================================
# ADDITIONAL SPATIAL GRADIENTS
# =============================================================================

@njit
def dome_gradient(nx, ny, center_x=None, center_y=None, height=1.0, steepness=1.0):
    """Dome/bell-shaped gradient"""
    if center_x is None:
        center_x = nx // 2
    if center_y is None:
        center_y = ny // 2
    
    pattern = np.zeros((ny, nx))
    max_dist = np.sqrt(center_x**2 + center_y**2)
    
    for y in range(ny):
        for x in range(nx):
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            normalized_dist = dist / max_dist
            # Gaussian-like dome shape
            pattern[y, x] = height * np.exp(-steepness * normalized_dist**2)
    
    return pattern

@njit
def inverse_dome_gradient(nx, ny, center_x=None, center_y=None, depth=1.0, steepness=1.0):
    """Inverted dome (crater/bowl) gradient"""
    dome = dome_gradient(nx, ny, center_x, center_y, depth, steepness)
    return 1.0 - dome

@njit
def parabolic_gradient(nx, ny, direction=0, vertex_pos=0.5, curvature=1.0):
    """Parabolic gradient
    direction: 0=horizontal, 1=vertical, 2=diagonal"""
    pattern = np.zeros((ny, nx))
    
    for y in range(ny):
        for x in range(nx):
            if direction == 0:  # Horizontal parabola
                t = x / (nx - 1)
                pattern[y, x] = curvature * (t - vertex_pos)**2
            elif direction == 1:  # Vertical parabola
                t = y / (ny - 1)
                pattern[y, x] = curvature * (t - vertex_pos)**2
            else:  # Diagonal parabola
                t = (x + y) / (nx + ny - 2)
                pattern[y, x] = curvature * (t - vertex_pos)**2
    
    # Normalize to [0,1]
    min_val, max_val = pattern.min(), pattern.max()
    if max_val != min_val:
        pattern = (pattern - min_val) / (max_val - min_val)
    
    return pattern

@njit
def exponential_gradient(nx, ny, direction=0, decay_rate=2.0, flip=False):
    """Exponential decay gradient"""
    pattern = np.zeros((ny, nx))
    
    for y in range(ny):
        for x in range(nx):
            if direction == 0:  # Horizontal
                t = x / (nx - 1)
            elif direction == 1:  # Vertical
                t = y / (ny - 1)
            else:  # Diagonal
                t = (x + y) / (nx + ny - 2)
            
            if flip:
                t = 1.0 - t
            
            pattern[y, x] = np.exp(-decay_rate * t)
    
    # Normalize to [0,1]
    min_val, max_val = pattern.min(), pattern.max()
    if max_val != min_val:
        pattern = (pattern - min_val) / (max_val - min_val)
    
    return pattern

@njit
def sinusoidal_gradient(nx, ny, direction=0, frequency=1.0, phase=0.0):
    """Sinusoidal gradient (half or full wave)"""
    pattern = np.zeros((ny, nx))
    
    for y in range(ny):
        for x in range(nx):
            if direction == 0:  # Horizontal
                t = x / (nx - 1)
            elif direction == 1:  # Vertical
                t = y / (ny - 1)
            else:  # Diagonal
                t = (x + y) / (nx + ny - 2)
            
            pattern[y, x] = np.sin(frequency * np.pi * t + phase)
    
    return (pattern + 1.0) / 2.0

@njit
def step_gradient(nx, ny, n_steps=5, direction=0, noise_level=0.0, seed=42):
    """Step gradient with optional noise"""
    if noise_level > 0:
        np.random.seed(seed)
    
    pattern = np.zeros((ny, nx))
    
    for y in range(ny):
        for x in range(nx):
            if direction == 0:  # Horizontal
                t = x / (nx - 1)
            elif direction == 1:  # Vertical
                t = y / (ny - 1)
            else:  # Diagonal
                t = (x + y) / (nx + ny - 2)
            
            # Create steps
            step_val = int(t * n_steps) / (n_steps - 1) if n_steps > 1 else 0.5
            
            # Add noise if requested
            if noise_level > 0:
                noise = (np.random.rand() - 0.5) * noise_level
                step_val = max(0, min(1, step_val + noise))
            
            pattern[y, x] = step_val
    
    return pattern

@njit
def multi_center_gradient(nx, ny, n_centers=3, blend_mode=0, seed=42):
    """Multiple overlapping gradients from different centers
    blend_mode: 0=max, 1=sum, 2=multiply"""
    np.random.seed(seed)
    
    # Generate random centers and parameters
    centers = []
    for _ in range(n_centers):
        cx = int(np.random.rand() * nx)
        cy = int(np.random.rand() * ny)
        intensity = np.random.rand()
        decay = np.random.uniform(0.5, 3.0)
        centers.append((cx, cy, intensity, decay))
    
    pattern = np.zeros((ny, nx))
    
    for y in range(ny):
        for x in range(nx):
            values = []
            for cx, cy, intensity, decay in centers:
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                max_dist = np.sqrt(nx**2 + ny**2)
                normalized_dist = dist / max_dist
                value = intensity * np.exp(-decay * normalized_dist**2)
                values.append(value)
            
            if blend_mode == 0:  # Max
                pattern[y, x] = max(values)
            elif blend_mode == 1:  # Sum
                pattern[y, x] = sum(values) / len(values)
            else:  # Multiply
                result = 1.0
                for v in values:
                    result *= (v + 0.1)  # Add small offset to avoid zero
                pattern[y, x] = result
    
    # Normalize to [0,1]
    min_val, max_val = pattern.min(), pattern.max()
    if max_val != min_val:
        pattern = (pattern - min_val) / (max_val - min_val)
    
    return pattern

# =============================================================================
# WRAP-UP FUNCTIONS
# =============================================================================

def combine_random_patterns(nx, ny, n_patterns=3, blend_mode='average', weights=None, seed=None):
    """
    Combine N randomly selected patterns with various blending modes
    
    Parameters:
    -----------
    nx, ny : int
        Dimensions
    n_patterns : int
        Number of patterns to combine
    blend_mode : str
        'average', 'max', 'min', 'multiply', 'screen', 'overlay', 'weighted'
    weights : list, optional
        Weights for weighted blending (only used if blend_mode='weighted')
    seed : int, optional
        Random seed
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate N random patterns
    patterns = []
    for i in range(n_patterns):
        pattern_seed = np.random.randint(0, 100000) if seed is None else seed + i
        pattern = generate_random_pattern(nx, ny, seed=pattern_seed)
        patterns.append(pattern)
    
    patterns = np.array(patterns)
    
    # Set up weights
    if weights is None:
        if blend_mode == 'weighted':
            weights = np.random.rand(n_patterns)
            weights = weights / weights.sum()  # Normalize
        else:
            weights = np.ones(n_patterns) / n_patterns
    else:
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
    
    # Combine patterns based on blend mode
    if blend_mode == 'average' or blend_mode == 'weighted':
        result = np.zeros((ny, nx))
        for i, pattern in enumerate(patterns):
            result += weights[i] * pattern
    
    elif blend_mode == 'max':
        result = np.max(patterns, axis=0)
    
    elif blend_mode == 'min':
        result = np.min(patterns, axis=0)
    
    elif blend_mode == 'multiply':
        result = np.ones((ny, nx))
        for pattern in patterns:
            result *= pattern
    
    elif blend_mode == 'screen':
        # Screen blend: 1 - (1-a)(1-b)
        result = np.ones((ny, nx))
        for pattern in patterns:
            result = 1 - (1 - result) * (1 - pattern)
    
    elif blend_mode == 'overlay':
        # Overlay blend
        result = patterns[0]
        for i in range(1, len(patterns)):
            mask = result < 0.5
            result = np.where(mask, 
                            2 * result * patterns[i],
                            1 - 2 * (1 - result) * (1 - patterns[i]))
    
    else:
        raise ValueError(f"Unknown blend mode: {blend_mode}")
    
    # Ensure result is in [0,1] range
    result = np.clip(result, 0, 1)
    
    return result

def noise_with_spatial_gradient(nx, ny, noise_type='perlin', gradient_type='radial', 
                               blend_mode='multiply', noise_params=None, gradient_params=None, seed=None):
    """
    Combine a noise pattern with a spatial gradient
    
    Parameters:
    -----------
    nx, ny : int
        Dimensions
    noise_type : str
        'perlin', 'white', 'pink', 'turbulence'
    gradient_type : str
        'linear', 'radial', 'dome', 'inverse_dome', 'parabolic', 'exponential', 
        'sinusoidal', 'step', 'multi_center'
    blend_mode : str
        'multiply', 'screen', 'overlay', 'add', 'subtract'
    noise_params : dict, optional
        Parameters for noise generation
    gradient_params : dict, optional
        Parameters for gradient generation
    seed : int, optional
        Random seed
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Set default parameters
    if noise_params is None:
        noise_params = {}
    if gradient_params is None:
        gradient_params = {}
    
    # Generate noise pattern
    if noise_type == 'perlin':
        default_noise_params = {
            'scale': np.random.uniform(0.02, 0.1),
            'octaves': np.random.randint(3, 6),
            'persistence': np.random.uniform(0.3, 0.7),
            'seed': np.random.randint(0, 10000) if seed is None else seed
        }
        default_noise_params.update(noise_params)
        noise = fractal_brownian_motion(nx, ny, **default_noise_params)
    
    elif noise_type == 'white':
        noise_seed = noise_params.get('seed', np.random.randint(0, 10000) if seed is None else seed)
        noise = white_noise(nx, ny, seed=noise_seed)
    
    elif noise_type == 'pink':
        noise_seed = noise_params.get('seed', np.random.randint(0, 10000) if seed is None else seed)
        noise = pink_noise(nx, ny, seed=noise_seed)
    
    elif noise_type == 'turbulence':
        default_noise_params = {
            'scale': np.random.uniform(0.03, 0.08),
            'octaves': np.random.randint(2, 5),
            'seed': np.random.randint(0, 10000) if seed is None else seed
        }
        default_noise_params.update(noise_params)
        noise = turbulence_pattern(nx, ny, **default_noise_params)
    
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    # Generate gradient pattern
    if gradient_type == 'linear':
        default_grad_params = {
            'direction': np.random.randint(0, 3),
            'start_val': np.random.uniform(0, 0.3),
            'end_val': np.random.uniform(0.7, 1.0)
        }
        default_grad_params.update(gradient_params)
        gradient = linear_gradient(nx, ny, **default_grad_params)
    
    elif gradient_type == 'radial':
        default_grad_params = {
            'center_x': gradient_params.get('center_x', nx // 2),
            'center_y': gradient_params.get('center_y', ny // 2),
            'inner_val': np.random.uniform(0, 0.3),
            'outer_val': np.random.uniform(0.7, 1.0)
        }
        default_grad_params.update(gradient_params)
        gradient = radial_gradient(nx, ny, **default_grad_params)
    
    elif gradient_type == 'dome':
        default_grad_params = {
            'height': np.random.uniform(0.8, 1.0),
            'steepness': np.random.uniform(0.5, 2.0)
        }
        default_grad_params.update(gradient_params)
        gradient = dome_gradient(nx, ny, **default_grad_params)
    
    elif gradient_type == 'inverse_dome':
        default_grad_params = {
            'depth': np.random.uniform(0.8, 1.0),
            'steepness': np.random.uniform(0.5, 2.0)
        }
        default_grad_params.update(gradient_params)
        gradient = inverse_dome_gradient(nx, ny, **default_grad_params)
    
    elif gradient_type == 'parabolic':
        default_grad_params = {
            'direction': np.random.randint(0, 3),
            'vertex_pos': np.random.uniform(0.3, 0.7),
            'curvature': np.random.uniform(0.5, 2.0)
        }
        default_grad_params.update(gradient_params)
        gradient = parabolic_gradient(nx, ny, **default_grad_params)
    
    elif gradient_type == 'exponential':
        default_grad_params = {
            'direction': np.random.randint(0, 3),
            'decay_rate': np.random.uniform(1.0, 4.0),
            'flip': np.random.choice([True, False])
        }
        default_grad_params.update(gradient_params)
        gradient = exponential_gradient(nx, ny, **default_grad_params)
    
    elif gradient_type == 'sinusoidal':
        default_grad_params = {
            'direction': np.random.randint(0, 3),
            'frequency': np.random.uniform(0.5, 2.0),
            'phase': np.random.uniform(0, 2*np.pi)
        }
        default_grad_params.update(gradient_params)
        gradient = sinusoidal_gradient(nx, ny, **default_grad_params)
    
    elif gradient_type == 'step':
        default_grad_params = {
            'n_steps': np.random.randint(3, 8),
            'direction': np.random.randint(0, 3),
            'noise_level': np.random.uniform(0, 0.1)
        }
        default_grad_params.update(gradient_params)
        gradient = step_gradient(nx, ny, **default_grad_params)
    
    elif gradient_type == 'multi_center':
        default_grad_params = {
            'n_centers': np.random.randint(2, 5),
            'blend_mode': np.random.randint(0, 3),
            'seed': np.random.randint(0, 10000) if seed is None else seed
        }
        default_grad_params.update(gradient_params)
        gradient = multi_center_gradient(nx, ny, **default_grad_params)
    
    else:
        raise ValueError(f"Unknown gradient type: {gradient_type}")
    
    # Combine noise and gradient
    if blend_mode == 'multiply':
        result = noise * gradient
    elif blend_mode == 'screen':
        result = 1 - (1 - noise) * (1 - gradient)
    elif blend_mode == 'overlay':
        mask = gradient < 0.5
        result = np.where(mask, 
                         2 * noise * gradient,
                         1 - 2 * (1 - noise) * (1 - gradient))
    elif blend_mode == 'add':
        result = np.clip(noise + gradient, 0, 1)
    elif blend_mode == 'subtract':
        result = np.clip(noise - gradient + 0.5, 0, 1)  # Add 0.5 to keep in reasonable range
    else:
        raise ValueError(f"Unknown blend mode: {blend_mode}")
    
    return result

def generate_ultimate_random_pattern(nx, ny, seed=None):
    """
    Generate the most random pattern possible by combining multiple techniques
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Randomly choose the generation method
    method = np.random.choice([
        'single_pattern',      # 40% chance
        'combine_patterns',    # 30% chance  
        'noise_with_gradient'  # 30% chance
    ], p=[0.4, 0.3, 0.3])
    
    if method == 'single_pattern':
        return generate_random_pattern(nx, ny, seed)
    
    elif method == 'combine_patterns':
        n_patterns = np.random.randint(2, 5)
        blend_modes = ['average', 'max', 'min', 'multiply', 'screen', 'overlay', 'weighted']
        blend_mode = np.random.choice(blend_modes)
        return combine_random_patterns(nx, ny, n_patterns=n_patterns, 
                                     blend_mode=blend_mode, seed=seed)
    
    else:  # noise_with_gradient
        noise_types = ['perlin', 'white', 'pink', 'turbulence']
        gradient_types = ['linear', 'radial', 'dome', 'inverse_dome', 'parabolic', 
                         'exponential', 'sinusoidal', 'step', 'multi_center']
        blend_modes = ['multiply', 'screen', 'overlay', 'add', 'subtract']
        
        noise_type = np.random.choice(noise_types)
        gradient_type = np.random.choice(gradient_types)
        blend_mode = np.random.choice(blend_modes)
        
        return noise_with_spatial_gradient(nx, ny, noise_type=noise_type,
                                         gradient_type=gradient_type,
                                         blend_mode=blend_mode, seed=seed)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_all_pattern_names():
    """Return list of all available pattern generator names"""
    return [
        'sine_waves_2d', 'cosine_waves_2d', 'radial_waves',
        'variable_frequency_waves', 'chirp_pattern',
        'checkerboard', 'concentric_circles', 'spiral_pattern',
        'mandelbrot_set', 'julia_set',
        'fractal_brownian_motion', 'turbulence_pattern', 'white_noise', 'pink_noise',
        'cellular_automata', 'wave_interference',
        'linear_gradient', 'radial_gradient',
        'voronoi_texture', 'wood_grain',
        # Additional gradients
        'dome_gradient', 'inverse_dome_gradient', 'parabolic_gradient',
        'exponential_gradient', 'sinusoidal_gradient', 'step_gradient', 'multi_center_gradient'
    ]

def get_all_gradient_types():
    """Return list of all available gradient types"""
    return [
        'linear', 'radial', 'dome', 'inverse_dome', 'parabolic',
        'exponential', 'sinusoidal', 'step', 'multi_center'
    ]

def get_all_noise_types():
    """Return list of all available noise types"""
    return ['perlin', 'white', 'pink', 'turbulence']

def get_all_blend_modes():
    """Return list of all available blend modes"""
    return ['average', 'max', 'min', 'multiply', 'screen', 'overlay', 'weighted', 'add', 'subtract']

def batch_generate_patterns(nx, ny, n_patterns, output_dir=None, seed_start=0, 
                           use_ultimate=False):
    """Generate a batch of random patterns for training data"""
    patterns = []
    
    for i in range(n_patterns):
        if use_ultimate:
            pattern = generate_ultimate_random_pattern(nx, ny, seed=seed_start + i)
        else:
            pattern = generate_random_pattern(nx, ny, seed=seed_start + i)
        patterns.append(pattern)
        
        if output_dir:
            np.save(f"{output_dir}/pattern_{i:04d}.npy", pattern)
    
    return np.array(patterns) if not output_dir else None


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ny,nx = 1024,1024

    fig,ax = plt.subplots()
    im = ax.imshow(radial_waves(nx,ny))
    fig.show()

    while True:
        z = generate_ultimate_random_pattern(nx,ny)
        im.set_data(z)
        fig.canvas.draw_idle()
        fig.canvas.start_event_loop(0.2)