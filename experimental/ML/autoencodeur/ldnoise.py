import numpy as np
from numba import njit
import math
from scipy import ndimage
from perlin_noise import perlin_noise_2d

# =============================================================================
# NORMALIZATION UTILITIES
# =============================================================================

def normalize_linear(pattern):
    """Linear normalization to [0,1]"""
    min_val, max_val = pattern.min(), pattern.max()
    if max_val != min_val:
        return (pattern - min_val) / (max_val - min_val)
    return np.full_like(pattern, 0.5)

def normalize_power(pattern, power=2.0):
    """Power law normalization to [0,1]"""
    normalized = normalize_linear(pattern)
    return normalized ** power

def normalize_sigmoid(pattern, steepness=5.0):
    """Sigmoid normalization to [0,1]"""
    centered = pattern - pattern.mean()
    sigmoid = 1 / (1 + np.exp(-steepness * centered / pattern.std()))
    return normalize_linear(sigmoid)

def normalize_random(pattern, seed=None):
    """Random normalization method"""
    if seed is not None:
        np.random.seed(seed)
    
    method = np.random.choice(['linear', 'power', 'sigmoid'])
    if method == 'linear':
        return normalize_linear(pattern)
    elif method == 'power':
        power = np.random.uniform(0.5, 3.0)
        return normalize_power(pattern, power)
    else:
        steepness = np.random.uniform(2.0, 8.0)
        return normalize_sigmoid(pattern, steepness)

# =============================================================================
# 1. BASE TOPOGRAPHY GENERATORS - MORE VARIED SCALES
# =============================================================================

def base_topography_continental(nx, ny, seed=42):
    """Very broad continental-scale features"""
    return perlin_noise_2d(nx, ny, scale=0.003, octaves=2, persistence=0.8, seed=seed)

def base_topography_regional(nx, ny, seed=42):
    """Regional-scale features"""
    return perlin_noise_2d(nx, ny, scale=0.01, octaves=3, persistence=0.7, seed=seed)

def base_topography_local(nx, ny, seed=42):
    """Local-scale features"""
    return perlin_noise_2d(nx, ny, scale=0.04, octaves=4, persistence=0.6, seed=seed)

def base_topography_detailed(nx, ny, seed=42):
    """Detailed features"""
    return perlin_noise_2d(nx, ny, scale=0.08, octaves=5, persistence=0.5, seed=seed)

def base_topography_mixed_scales(nx, ny, seed=42):
    """Mix of multiple scales with proper weighting"""
    np.random.seed(seed)
    
    # Generate different scales
    continental = base_topography_continental(nx, ny, seed)
    regional = base_topography_regional(nx, ny, seed + 1000)
    local = base_topography_local(nx, ny, seed + 2000)
    detailed = base_topography_detailed(nx, ny, seed + 3000)
    
    # Weight toward larger scales for more realistic topography
    weights = np.random.dirichlet([4, 3, 2, 1])  # Strongly favor larger scales
    result = (weights[0] * continental + weights[1] * regional + 
              weights[2] * local + weights[3] * detailed)
    
    return normalize_random(result, seed)

def base_topography_ridged_broad(nx, ny, scale=0.015, octaves=3, seed=42):
    """Broader ridged topography"""
    noise = perlin_noise_2d(nx, ny, scale=scale, octaves=octaves, 
                           persistence=0.6, seed=seed)
    ridged = 1 - 2 * np.abs(noise - 0.5)
    return normalize_random(ridged, seed)

def base_topography_plateaus_broad(nx, ny, n_levels=3, smoothness=0.5, seed=42):
    """Broader plateau systems"""
    # Use much broader base noise
    base_noise = perlin_noise_2d(nx, ny, scale=0.008, octaves=2, 
                                persistence=0.7, seed=seed)
    
    # Create stepped levels
    levels = np.linspace(0, 1, n_levels)
    stepped = np.zeros_like(base_noise)
    
    for i, level in enumerate(levels[:-1]):
        mask = (base_noise >= level) & (base_noise < levels[i+1])
        stepped[mask] = level
    stepped[base_noise >= levels[-1]] = levels[-1]
    
    # Apply more smoothing for broader features
    if smoothness > 0:
        sigma = smoothness * min(nx, ny) * 0.04  # Increased smoothing
        stepped = ndimage.gaussian_filter(stepped, sigma=sigma)
    
    return normalize_random(stepped, seed)

# =============================================================================
# 2. ERODABILITY (K) PATTERNS - CONTINUOUS BANDS, NO STICKS
# =============================================================================

@njit
def erodability_volcanic_dome(nx, ny, center_x=None, center_y=None, 
                             size_factor=0.4, hardness=0.2, deformation=1.0, seed=42):
    """Volcanic intrusion - continuous hard rock dome"""
    np.random.seed(seed)
    
    if center_x is None:
        center_x = 0.2 + np.random.random() * 0.6
    if center_y is None:
        center_y = 0.2 + np.random.random() * 0.6
    
    pattern = np.ones((ny, nx))
    
    # Larger, more continuous dome
    a = size_factor * min(nx, ny) * (1.0 + np.random.random() * 0.5)  # Bigger
    b = a * (0.7 + np.random.random() * 0.6) * deformation
    angle = np.random.random() * np.pi
    
    cos_angle, sin_angle = np.cos(angle), np.sin(angle)
    
    for y in range(ny):
        for x in range(nx):
            dx = x - center_x * nx
            dy = y - center_y * ny
            
            rx = cos_angle * dx - sin_angle * dy
            ry = sin_angle * dx + cos_angle * dy
            
            dist = np.sqrt((rx/a)**2 + (ry/b)**2)
            
            if dist < 1.5:  # Extend influence further
                # Smoother, more continuous hardness variation
                rock_hardness = hardness + (1 - hardness) * (dist/1.5)**1.5
                pattern[y, x] = rock_hardness
    
    return pattern

@njit
def erodability_continuous_bands(nx, ny, n_bands=4, band_angle=15, 
                                thickness_variation=0.3, hardness_contrast=0.8, seed=42):
    """Continuous stratified rock layers - no discontinuities"""
    np.random.seed(seed)
    
    pattern = np.zeros((ny, nx))
    angle_rad = np.radians(band_angle)
    cos_angle, sin_angle = np.cos(angle_rad), np.sin(angle_rad)
    
    # Generate smooth band properties
    band_hardness = []
    for i in range(n_bands):
        # Create smoother transitions between bands
        base_hardness = i / (n_bands - 1) if n_bands > 1 else 0.5
        variation = (np.random.random() - 0.5) * 0.3
        hardness = max(0.1, min(0.9, base_hardness + variation))
        band_hardness.append(hardness)
    
    for y in range(ny):
        for x in range(nx):
            # Project onto band direction
            proj = (cos_angle * x/nx + sin_angle * y/ny) % 1.0
            
            # Smooth band assignment
            band_pos = proj * n_bands
            band_idx = int(band_pos) % n_bands
            band_frac = band_pos - int(band_pos)
            
            # Smooth interpolation between bands
            if band_frac < 0.5:
                prev_idx = (band_idx - 1) % n_bands
                t = band_frac + 0.5
                hardness = (1-t) * band_hardness[prev_idx] + t * band_hardness[band_idx]
            else:
                next_idx = (band_idx + 1) % n_bands
                t = band_frac - 0.5
                hardness = (1-t) * band_hardness[band_idx] + t * band_hardness[next_idx]
            
            pattern[y, x] = hardness
    
    # Enhance contrast but keep continuity
    pattern = pattern * hardness_contrast + (1 - hardness_contrast) * 0.5
    
    return pattern

def erodability_folded_continuous(nx, ny, fold_wavelength=0.15, fold_amplitude=0.08, 
                                 n_layers=6, fold_angle=0, seed=42):
    """Continuously folded rock layers"""
    np.random.seed(seed)
    
    pattern = np.zeros((ny, nx))
    angle_rad = math.radians(fold_angle)
    
    # Smoother layer hardness progression
    layer_hardness = np.linspace(0.2, 0.8, n_layers)
    np.random.shuffle(layer_hardness)
    
    for y in range(ny):
        for x in range(nx):
            norm_x = x / nx
            norm_y = y / ny
            
            rot_x = math.cos(angle_rad) * norm_x - math.sin(angle_rad) * norm_y
            rot_y = math.sin(angle_rad) * norm_x + math.cos(angle_rad) * norm_y
            
            # Smoother folding
            fold_offset = fold_amplitude * math.sin(2 * math.pi * rot_x / fold_wavelength)
            folded_y = (rot_y + fold_offset) % 1.0
            
            # Smooth layer interpolation
            layer_pos = folded_y * n_layers
            layer_idx = int(layer_pos) % n_layers
            layer_frac = layer_pos - int(layer_pos)
            
            next_idx = (layer_idx + 1) % n_layers
            hardness = (1 - layer_frac) * layer_hardness[layer_idx] + layer_frac * layer_hardness[next_idx]
            
            pattern[y, x] = hardness
    
    return pattern

def erodability_broad_zones(nx, ny, n_zones=4, zone_complexity=0.02, seed=42):
    """Broad geological zones - no fracture sticks"""
    # Use very low frequency Perlin noise for broad, continuous zones
    base_zones = perlin_noise_2d(nx, ny, scale=zone_complexity, octaves=2, 
                                persistence=0.8, seed=seed)
    
    # Add secondary variation
    secondary = perlin_noise_2d(nx, ny, scale=zone_complexity * 3, octaves=3,
                               persistence=0.4, seed=seed + 1000)
    
    combined = 0.8 * base_zones + 0.2 * secondary
    
    # Create smooth zones
    pattern = np.zeros_like(combined)
    zone_values = np.linspace(0.1, 0.9, n_zones)
    
    for y in range(ny):
        for x in range(nx):
            noise_val = combined[y, x]
            # Smooth zone assignment
            zone_pos = noise_val * (n_zones - 1)
            zone_idx = int(zone_pos)
            zone_frac = zone_pos - zone_idx
            
            if zone_idx >= n_zones - 1:
                pattern[y, x] = zone_values[-1]
            else:
                # Smooth interpolation between zones
                pattern[y, x] = ((1 - zone_frac) * zone_values[zone_idx] + 
                               zone_frac * zone_values[zone_idx + 1])
    
    return normalize_random(pattern, seed)

def erodability_contact_zones(nx, ny, n_contacts=2, contact_hardness=0.25, 
                             aureole_size=0.25, seed=42):
    """Broad contact metamorphism zones"""
    np.random.seed(seed)
    
    pattern = np.random.random((ny, nx)) * 0.3 + 0.5  # Base rock
    
    for i in range(n_contacts):
        center_x = 0.15 + np.random.random() * 0.7  # More central
        center_y = 0.15 + np.random.random() * 0.7
        
        # Larger, more continuous zones
        core_size = aureole_size * (0.4 + np.random.random() * 0.3)
        aureole_extent = aureole_size * (1.2 + np.random.random() * 0.5)
        
        for y in range(ny):
            for x in range(nx):
                dist = math.sqrt(((x/nx - center_x)**2 + (y/ny - center_y)**2))
                
                if dist < core_size:
                    pattern[y, x] = contact_hardness
                elif dist < aureole_extent:
                    # Very smooth transition
                    t = (dist - core_size) / (aureole_extent - core_size)
                    t_smooth = t * t * (3 - 2 * t)  # Smooth step
                    hardness = contact_hardness + t_smooth * (pattern[y, x] - contact_hardness)
                    pattern[y, x] = hardness
    
    return normalize_random(pattern, seed)

# =============================================================================
# 3. PRECIPITATION PATTERNS - GLOBAL, BLOBBY PATTERNS
# =============================================================================

def precipitation_large_cells(nx, ny, n_cells=6, seed=42):
    """Large-scale precipitation cells - blobby, global patterns"""
    np.random.seed(seed)
    
    # Fewer, larger cells
    cells = []
    for i in range(n_cells):
        x = np.random.random()
        y = np.random.random()
        intensity = 0.3 + np.random.random() * 0.6
        size = 0.8 + np.random.random() * 0.7  # Much larger
        cells.append((x, y, intensity, size))
    
    pattern = np.full((ny, nx), 0.2)  # Low base precipitation
    
    for y in range(ny):
        for x in range(nx):
            norm_x, norm_y = x / nx, y / ny
            
            total_weight = 0
            weighted_intensity = 0
            
            for cx, cy, intensity, size in cells:
                dist = math.sqrt((norm_x - cx)**2 + (norm_y - cy)**2)
                # Much larger influence radius
                weight = math.exp(-dist / (size * 0.6))
                weighted_intensity += intensity * weight
                total_weight += weight
            
            if total_weight > 0:
                pattern[y, x] = max(pattern[y, x], weighted_intensity / total_weight)
    
    return normalize_random(pattern, seed)

def precipitation_broad_bands(nx, ny, n_bands=2, band_angle=45, seed=42):
    """Broad precipitation bands - continental scale"""
    np.random.seed(seed)
    
    pattern = np.full((ny, nx), 0.25)  # Base precipitation
    angle_rad = math.radians(band_angle)
    cos_angle, sin_angle = math.cos(angle_rad), math.sin(angle_rad)
    
    for i in range(n_bands):
        # Much broader bands
        band_pos = (i + 0.5) / n_bands + np.random.normal(0, 0.05)
        intensity = 0.4 + np.random.random() * 0.5
        width = 0.4 + np.random.random() * 0.3  # Very wide bands
        
        for y in range(ny):
            for x in range(nx):
                proj = cos_angle * (x/nx) + sin_angle * (y/ny)
                dist_from_band = abs(proj - band_pos)
                
                if dist_from_band < width:
                    # Very smooth, broad profile
                    band_intensity = intensity * math.exp(-((dist_from_band/width)**2) * 1.5)
                    pattern[y, x] = max(pattern[y, x], band_intensity)
    
    return normalize_random(pattern, seed)

def precipitation_continental_gradient(nx, ny, direction=0, seed=42):
    """Continental-scale precipitation gradient"""
    np.random.seed(seed)
    
    pattern = np.zeros((ny, nx))
    
    # Add some large-scale variation
    base_noise = perlin_noise_2d(nx, ny, scale=0.008, octaves=2, 
                                persistence=0.7, seed=seed)
    
    for y in range(ny):
        for x in range(nx):
            if direction == 0:  # West to East
                t = x / (nx - 1)
            elif direction == 1:  # North to South
                t = y / (ny - 1)
            elif direction == 2:  # SW to NE
                t = (x + y) / (nx + ny - 2)
            else:  # NW to SE
                t = (x - y + ny) / (nx + ny - 2)
            
            # Smooth continental gradient with noise
            base_precip = 0.2 + 0.6 * (1 - t**1.5)
            noise_contrib = base_noise[y, x] * 0.3
            pattern[y, x] = base_precip + noise_contrib
    
    return normalize_random(pattern, seed)

def precipitation_monsoon_zones(nx, ny, n_zones=3, seed=42):
    """Large monsoon-like precipitation zones"""
    np.random.seed(seed)
    
    # Very broad base pattern
    base = perlin_noise_2d(nx, ny, scale=0.005, octaves=2, persistence=0.8, seed=seed)
    
    # Add broad secondary pattern
    secondary = perlin_noise_2d(nx, ny, scale=0.015, octaves=3, 
                               persistence=0.6, seed=seed + 1000)
    
    # Combine for broad, blobby patterns
    combined = 0.7 * base + 0.3 * secondary
    
    # Create zones with smooth transitions
    pattern = 0.2 + 0.6 * combined  # Scale to reasonable precip range
    
    return normalize_random(pattern, seed)

def precipitation_perlin_broad(nx, ny, seed=42):
    """Very broad-scale Perlin precipitation"""
    # Much larger scale Perlin noise
    base = perlin_noise_2d(nx, ny, scale=0.006, octaves=3, 
                          persistence=0.75, seed=seed)
    
    # Add some medium-scale variation
    medium = perlin_noise_2d(nx, ny, scale=0.02, octaves=2,
                            persistence=0.6, seed=seed + 1000)
    
    # Combine with emphasis on broad patterns
    pattern = 0.8 * base + 0.2 * medium
    pattern = 0.15 + 0.7 * pattern  # Scale to precip range
    
    return normalize_random(pattern, seed)

# =============================================================================
# 4. TECTONIC UPLIFT PATTERNS - MORE CONTINUOUS
# =============================================================================

def uplift_broad_regional(nx, ny, n_centers=2, max_amplitude=0.8, seed=42):
    """Broad, continuous regional uplift"""
    np.random.seed(seed)
    
    pattern = np.zeros((ny, nx))
    
    for i in range(n_centers):
        center_x = 0.15 + np.random.random() * 0.7
        center_y = 0.15 + np.random.random() * 0.7
        amplitude = np.random.random() * max_amplitude
        
        # Much larger, more continuous uplift zones
        a = 0.5 + np.random.random() * 0.3  # Larger major axis
        b = a * (0.7 + np.random.random() * 0.6)
        angle = np.random.random() * np.pi
        
        cos_angle, sin_angle = math.cos(angle), math.sin(angle)
        
        for y in range(ny):
            for x in range(nx):
                dx = x/nx - center_x
                dy = y/ny - center_y
                
                rx = cos_angle * dx - sin_angle * dy
                ry = sin_angle * dx + cos_angle * dy
                
                dist = math.sqrt((rx/a)**2 + (ry/b)**2)
                
                if dist < 1.2:  # Extend influence further
                    # Very smooth, broad uplift profile
                    uplift = amplitude * math.exp(-2 * dist**2)
                    pattern[y, x] = max(pattern[y, x], uplift)
    
    return normalize_random(pattern, seed)

def uplift_continuous_faults(nx, ny, n_faults=2, fault_throw=0.6, seed=42):
    """Continuous fault systems - no rectangular blocks"""
    np.random.seed(seed)
    
    # Start with smooth base
    pattern = perlin_noise_2d(nx, ny, scale=0.01, octaves=2, 
                             persistence=0.7, seed=seed) * 0.3 + 0.5
    
    for fault_id in range(n_faults):
        # Create continuous fault influence across entire domain
        angle = np.random.random() * np.pi
        fault_center = 0.3 + np.random.random() * 0.4
        throw = fault_throw * (0.6 + np.random.random() * 0.4)
        
        cos_angle, sin_angle = math.cos(angle), math.sin(angle)
        
        for y in range(ny):
            for x in range(nx):
                norm_x, norm_y = x / nx, y / ny
                
                # Distance from fault plane (continuous across domain)
                across_fault = -sin_angle * (norm_x - 0.5) + cos_angle * (norm_y - 0.5)
                
                # Smooth, continuous fault influence
                if across_fault > 0:  # One side of fault
                    fault_influence = throw * math.exp(-abs(across_fault) * 8)
                    pattern[y, x] += fault_influence
                else:  # Other side of fault
                    fault_influence = throw * math.exp(-abs(across_fault) * 8)
                    pattern[y, x] -= fault_influence
    
    return normalize_random(pattern, seed)

def uplift_broad_graben(nx, ny, graben_width=0.4, graben_angle=30, 
                       boundary_throw=0.5, seed=42):
    """Broad, continuous graben system"""
    np.random.seed(seed)
    
    pattern = np.ones((ny, nx)) * 0.7
    
    # Much broader graben
    center_x = 0.5
    center_y = 0.5
    angle_rad = math.radians(graben_angle)
    cos_angle, sin_angle = math.cos(angle_rad), math.sin(angle_rad)
    
    for y in range(ny):
        for x in range(nx):
            dx = x/nx - center_x  
            dy = y/ny - center_y
            
            across_graben = abs(-sin_angle * dx + cos_angle * dy)
            
            # Very smooth, broad graben profile
            if across_graben < graben_width/2:
                # Smooth drop in center
                drop_factor = 1 - (across_graben / (graben_width/2))**2
                pattern[y, x] -= boundary_throw * drop_factor
            elif across_graben < graben_width:
                # Smooth transition zone
                transition_dist = (across_graben - graben_width/2) / (graben_width/2)
                drop_factor = (1 - transition_dist)**2
                pattern[y, x] -= boundary_throw * drop_factor * 0.3
    
    return normalize_random(pattern, seed)

def uplift_broad_tilt(nx, ny, tilt_direction=45, tilt_amount=0.6, seed=42):
    """Broad regional tilting"""
    np.random.seed(seed)
    
    angle_rad = math.radians(tilt_direction)
    cos_angle, sin_angle = math.cos(angle_rad), math.sin(angle_rad)
    
    pattern = np.zeros((ny, nx))
    
    # Add some broad-scale variation to the tilt
    base_noise = perlin_noise_2d(nx, ny, scale=0.008, octaves=2, 
                                persistence=0.7, seed=seed)
    
    for y in range(ny):
        for x in range(nx):
            norm_x, norm_y = x / nx, y / ny
            tilt_coord = cos_angle * norm_x + sin_angle * norm_y
            
            # Smooth tilt with broad variation
            tilt_value = tilt_coord * tilt_amount
            noise_contrib = base_noise[y, x] * 0.2
            pattern[y, x] = tilt_value + noise_contrib
    
    return normalize_random(pattern, seed)

def uplift_mountain_chain(nx, ny, chain_direction=45, chain_width=0.3, seed=42):
    """Continuous mountain chain uplift"""
    np.random.seed(seed)
    
    pattern = np.zeros((ny, nx))
    angle_rad = math.radians(chain_direction)
    cos_angle, sin_angle = math.cos(angle_rad), math.sin(angle_rad)
    
    # Mountain chain along center
    for y in range(ny):
        for x in range(nx):
            norm_x, norm_y = x / nx, y / ny
            
            # Distance from chain centerline
            dx = norm_x - 0.5
            dy = norm_y - 0.5
            dist_from_chain = abs(-sin_angle * dx + cos_angle * dy)
            
            if dist_from_chain < chain_width:
                # Smooth mountain profile
                elevation = math.exp(-((dist_from_chain / chain_width) ** 2) * 3)
                # Add some along-chain variation
                along_chain = cos_angle * dx + sin_angle * dy
                variation = 0.8 + 0.4 * math.sin(along_chain * 8 * math.pi)
                pattern[y, x] = elevation * variation
    
    return normalize_random(pattern, seed)

# =============================================================================
# MASTER GENERATION FUNCTIONS - UPDATED
# =============================================================================

def generate_base_topography(nx, ny, seed=None):
    """Generate varied base topography with better scale distribution"""
    if seed is not None:
        np.random.seed(seed)
    
    generators = [
        lambda: base_topography_continental(nx, ny, np.random.randint(0, 10000)),
        lambda: base_topography_regional(nx, ny, np.random.randint(0, 10000)),
        lambda: base_topography_local(nx, ny, np.random.randint(0, 10000)),
        lambda: base_topography_mixed_scales(nx, ny, np.random.randint(0, 10000)),
        lambda: base_topography_ridged_broad(nx, ny,
                                           scale=np.random.uniform(0.008, 0.025),
                                           octaves=np.random.randint(2, 4),
                                           seed=np.random.randint(0, 10000)),
        lambda: base_topography_plateaus_broad(nx, ny,
                                             n_levels=np.random.randint(2, 5),
                                             smoothness=np.random.uniform(0.3, 0.7),
                                             seed=np.random.randint(0, 10000))
    ]
    
    result = np.random.choice(generators)()
    return normalize_random(result, seed)

def generate_erodability_pattern(nx, ny, seed=None):
    """Generate varied erodability patterns - continuous only"""
    if seed is not None:
        np.random.seed(seed)
    
    generators = [
        lambda: erodability_volcanic_dome(nx, ny,
                                        size_factor=np.random.uniform(0.3, 0.6),
                                        hardness=np.random.uniform(0.1, 0.4),
                                        deformation=np.random.uniform(0.6, 1.4),
                                        seed=np.random.randint(0, 10000)),
        
        lambda: erodability_continuous_bands(nx, ny,
                                           n_bands=np.random.randint(3, 8),
                                           band_angle=np.random.uniform(-60, 60),
                                           thickness_variation=np.random.uniform(0.2, 0.5),
                                           hardness_contrast=np.random.uniform(0.6, 0.9),
                                           seed=np.random.randint(0, 10000)),
        
        lambda: erodability_folded_continuous(nx, ny,
                                            fold_wavelength=np.random.uniform(0.1, 0.3),
                                            fold_amplitude=np.random.uniform(0.04, 0.12),
                                            n_layers=np.random.randint(4, 8),
                                            fold_angle=np.random.uniform(-90, 90),
                                            seed=np.random.randint(0, 10000)),
        
        lambda: erodability_broad_zones(nx, ny,
                                      n_zones=np.random.randint(3, 6),
                                      zone_complexity=np.random.uniform(0.01, 0.03),
                                      seed=np.random.randint(0, 10000)),
        
        lambda: erodability_contact_zones(nx, ny,
                                        n_contacts=np.random.randint(1, 3),
                                        contact_hardness=np.random.uniform(0.15, 0.4),
                                        aureole_size=np.random.uniform(0.2, 0.4),
                                        seed=np.random.randint(0, 10000))
    ]
    
    result = np.random.choice(generators)()
    return normalize_random(result, seed)

def generate_precipitation_pattern(nx, ny, seed=None):
    """Generate varied precipitation patterns - broad and blobby"""
    if seed is not None:
        np.random.seed(seed)
    
    generators = [
        lambda: precipitation_large_cells(nx, ny,
                                        n_cells=np.random.randint(4, 8),
                                        seed=np.random.randint(0, 10000)),
        
        lambda: precipitation_broad_bands(nx, ny,
                                        n_bands=np.random.randint(1, 3),
                                        band_angle=np.random.uniform(0, 180),
                                        seed=np.random.randint(0, 10000)),
        
        lambda: precipitation_continental_gradient(nx, ny,
                                                 direction=np.random.randint(0, 4),
                                                 seed=np.random.randint(0, 10000)),
        
        lambda: precipitation_monsoon_zones(nx, ny,
                                          n_zones=np.random.randint(2, 4),
                                          seed=np.random.randint(0, 10000)),
        
        lambda: precipitation_perlin_broad(nx, ny, seed=np.random.randint(0, 10000))
    ]
    
    result = np.random.choice(generators)()
    return normalize_random(result, seed)

def generate_tectonic_uplift(nx, ny, seed=None):
    """Generate varied tectonic uplift patterns - continuous"""
    if seed is not None:
        np.random.seed(seed)
    
    generators = [
        lambda: uplift_broad_regional(nx, ny,
                                    n_centers=np.random.randint(1, 3),
                                    max_amplitude=np.random.uniform(0.6, 1.0),
                                    seed=np.random.randint(0, 10000)),
        
        lambda: uplift_continuous_faults(nx, ny,
                                       n_faults=np.random.randint(1, 3),
                                       fault_throw=np.random.uniform(0.4, 0.8),
                                       seed=np.random.randint(0, 10000)),
        
        lambda: uplift_broad_graben(nx, ny,
                                  graben_width=np.random.uniform(0.3, 0.6),
                                  graben_angle=np.random.uniform(0, 90),
                                  boundary_throw=np.random.uniform(0.4, 0.7),
                                  seed=np.random.randint(0, 10000)),
        
        lambda: uplift_broad_tilt(nx, ny,
                                tilt_direction=np.random.uniform(0, 180),
                                tilt_amount=np.random.uniform(0.4, 0.8),
                                seed=np.random.randint(0, 10000)),
        
        lambda: uplift_mountain_chain(nx, ny,
                                    chain_direction=np.random.uniform(0, 180),
                                    chain_width=np.random.uniform(0.2, 0.4),
                                    seed=np.random.randint(0, 10000))
    ]
    
    result = np.random.choice(generators)()
    return normalize_random(result, seed)

# =============================================================================
# COMBINATION AND BLENDING FUNCTIONS
# =============================================================================

def combine_patterns(patterns, blend_mode='weighted', weights=None):
    """Combine multiple patterns with various blending modes"""
    patterns = np.array(patterns)
    
    if weights is None:
        weights = np.ones(len(patterns)) / len(patterns)
    else:
        weights = np.array(weights)
        weights = weights / weights.sum()
    
    if blend_mode == 'weighted':
        result = np.average(patterns, axis=0, weights=weights)
    
    elif blend_mode == 'max':
        result = np.max(patterns, axis=0)
    
    elif blend_mode == 'min':
        result = np.min(patterns, axis=0)
    
    elif blend_mode == 'multiply':
        result = np.prod(patterns, axis=0)
    
    elif blend_mode == 'geological_layering':
        # Simulate geological layering - stronger patterns override weaker ones
        result = patterns[0].copy()
        for i in range(1, len(patterns)):
            # Use pattern strength as a mask
            strength_mask = patterns[i] > 0.7
            transition_mask = (patterns[i] > 0.4) & (patterns[i] <= 0.7)
            
            result[strength_mask] = patterns[i][strength_mask]
            result[transition_mask] = (0.6 * result[transition_mask] + 
                                     0.4 * patterns[i][transition_mask])
    
    else:  # default to weighted
        result = np.average(patterns, axis=0, weights=weights)
    
    return normalize_random(result)

def add_noise_overlay(pattern, noise_type='perlin', noise_strength=0.1, seed=42):
    """Add noise overlay to existing pattern"""
    if noise_type == 'perlin':
        noise = perlin_noise_2d(pattern.shape[1], pattern.shape[0], 
                               scale=0.05, octaves=2, persistence=0.4, seed=seed)
    elif noise_type == 'white':
        np.random.seed(seed)
        noise = np.random.random(pattern.shape)
    else:  # gaussian
        np.random.seed(seed)
        noise = np.random.normal(0.5, 0.2, pattern.shape)
        noise = normalize_linear(noise)
    
    # Blend noise with original pattern
    result = (1 - noise_strength) * pattern + noise_strength * noise
    return normalize_random(result, seed)

# =============================================================================
# ULTIMATE PATTERN GENERATORS
# =============================================================================

def generate_landscape_dataset_pattern(nx, ny, pattern_type='mixed', 
                                     complexity='medium', seed=None):
    """
    Generate comprehensive landscape patterns for dataset creation
    
    Parameters:
    -----------
    nx, ny : int
        Pattern dimensions
    pattern_type : str
        'topography', 'erodability', 'precipitation', 'tectonic', 'mixed'
    complexity : str
        'simple', 'medium', 'complex'
    seed : int, optional
        Random seed
    """
    if seed is None:
        seed = np.random.randint(0, 2**31 - 1)

    np.random.seed(seed)
    
    if pattern_type == 'topography':
        if complexity == 'simple':
            result = generate_base_topography(nx, ny, seed)
        elif complexity == 'medium':
            # Combine 2 topographic patterns
            patterns = [generate_base_topography(nx, ny, seed + i) for i in range(2)]
            result = combine_patterns(patterns, 'weighted')
        else:  # complex
            # Combine 3 patterns with noise overlay
            patterns = [generate_base_topography(nx, ny, seed + i) for i in range(3)]
            result = combine_patterns(patterns, 'geological_layering')
            result = add_noise_overlay(result, noise_strength=0.05, seed=seed)
    
    elif pattern_type == 'erodability':
        if complexity == 'simple':
            result = generate_erodability_pattern(nx, ny, seed)
        elif complexity == 'medium':
            # Combine 2 erodability patterns
            patterns = [generate_erodability_pattern(nx, ny, seed + i) for i in range(2)]
            blend_mode = np.random.choice(['weighted', 'geological_layering'])
            result = combine_patterns(patterns, blend_mode)
        else:  # complex  
            # Complex geological history
            patterns = [generate_erodability_pattern(nx, ny, seed + i) for i in range(3)]
            result = combine_patterns(patterns, 'geological_layering')
            # Add some broad variation
            broad_overlay = erodability_broad_zones(nx, ny, 
                                                  n_zones=np.random.randint(2, 4),
                                                  seed=seed + 100)
            result = 0.7 * result + 0.3 * broad_overlay
    
    elif pattern_type == 'precipitation':
        if complexity == 'simple':
            result = generate_precipitation_pattern(nx, ny, seed)
        elif complexity == 'medium':
            # Combine different precipitation systems
            patterns = [generate_precipitation_pattern(nx, ny, seed + i) for i in range(2)]
            result = combine_patterns(patterns, 'max')  # Multiple weather systems
        else:  # complex
            # Multiple interacting weather systems
            patterns = [generate_precipitation_pattern(nx, ny, seed + i) for i in range(3)]
            result = combine_patterns(patterns, 'weighted')
            # Add very broad-scale variability
            result = add_noise_overlay(result, 'perlin', noise_strength=0.1, seed=seed)
    
    elif pattern_type == 'tectonic':
        if complexity == 'simple':
            result = generate_tectonic_uplift(nx, ny, seed)
        elif complexity == 'medium':
            # Combine different tectonic processes
            patterns = [generate_tectonic_uplift(nx, ny, seed + i) for i in range(2)]
            result = combine_patterns(patterns, 'weighted')
        else:  # complex
            # Multiple tectonic events
            patterns = [generate_tectonic_uplift(nx, ny, seed + i) for i in range(3)]
            result = combine_patterns(patterns, 'geological_layering')
    
    else:  # mixed - combine different types
        type_generators = [
            lambda: generate_base_topography(nx, ny, seed),
            lambda: generate_erodability_pattern(nx, ny, seed + 1000),
            lambda: generate_precipitation_pattern(nx, ny, seed + 2000),
            lambda: generate_tectonic_uplift(nx, ny, seed + 3000)
        ]
        
        if complexity == 'simple':
            result = np.random.choice(type_generators)()
        elif complexity == 'medium':
            # Combine 2 different types
            selected = np.random.choice(type_generators, 2, replace=False)
            patterns = [gen() for gen in selected]
            result = combine_patterns(patterns, 'weighted')
        else:  # complex
            # Combine 3-4 different types
            n_types = np.random.randint(3, 5)
            selected = np.random.choice(type_generators, n_types, replace=False)
            patterns = [gen() for gen in selected]
            result = combine_patterns(patterns, 'geological_layering')
            result = add_noise_overlay(result, noise_strength=0.05, seed=seed)
    
    return normalize_random(result, seed)

# =============================================================================
# BATCH GENERATION AND DATASET CREATION
# =============================================================================

def create_landscape_training_dataset(nx, ny, n_patterns_per_type=200, 
                                    output_dir=None, seed_start=0):
    """
    Create comprehensive training dataset for landscape modeling
    """
    pattern_types = ['topography', 'erodability', 'precipitation', 'tectonic']
    complexity_levels = ['simple', 'medium', 'complex']
    
    all_patterns = []
    metadata = []
    
    current_seed = seed_start
    pattern_id = 0
    
    for pattern_type in pattern_types:
        for complexity in complexity_levels:
            n_patterns = n_patterns_per_type // 3  # Divide among complexity levels
            
            print(f"Generating {n_patterns} {complexity} {pattern_type} patterns...")
            
            for i in range(n_patterns):
                pattern = generate_landscape_dataset_pattern(
                    nx, ny, pattern_type, complexity, seed=current_seed
                )
                
                all_patterns.append(pattern)
                
                metadata.append({
                    'id': pattern_id,
                    'type': pattern_type,
                    'complexity': complexity,
                    'seed': current_seed,
                    'filename': f"{pattern_type}_{complexity}_{i:04d}.npy" if output_dir else None
                })
                
                if output_dir:
                    subdir = f"{output_dir}/{pattern_type}/{complexity}"
                    import os
                    os.makedirs(subdir, exist_ok=True)
                    np.save(f"{subdir}/{pattern_type}_{complexity}_{i:04d}.npy", pattern)
                
                current_seed += 1
                pattern_id += 1
    
    # Add some mixed patterns for extra variety
    print(f"Generating {n_patterns_per_type} mixed complexity patterns...")
    for i in range(n_patterns_per_type):
        complexity = np.random.choice(complexity_levels)
        pattern = generate_landscape_dataset_pattern(
            nx, ny, 'mixed', complexity, seed=current_seed
        )
        
        all_patterns.append(pattern)
        metadata.append({
            'id': pattern_id,
            'type': 'mixed',
            'complexity': complexity,
            'seed': current_seed,
            'filename': f"mixed_{complexity}_{i:04d}.npy" if output_dir else None
        })
        
        if output_dir:
            subdir = f"{output_dir}/mixed/{complexity}"
            import os
            os.makedirs(subdir, exist_ok=True)
            np.save(f"{subdir}/mixed_{complexity}_{i:04d}.npy", pattern)
        
        current_seed += 1
        pattern_id += 1
    
    if output_dir:
        # Save metadata
        import json
        with open(f"{output_dir}/dataset_metadata.json", 'w') as f:
            json.dump({
                'total_patterns': len(metadata),
                'dimensions': [nx, ny],
                'pattern_types': pattern_types + ['mixed'],
                'complexity_levels': complexity_levels,
                'patterns_per_type': n_patterns_per_type,
                'patterns': metadata
            }, f, indent=2)
        
        print(f"\nDataset created successfully!")
        print(f"Total patterns: {len(metadata)}")
        print(f"Saved to: {output_dir}")
        
        return None, metadata
    
    return np.array(all_patterns), metadata

# =============================================================================
# QUICK GENERATION FUNCTIONS
# =============================================================================

def quick_topography(nx, ny, seed=None):
    """Quick topography generation"""
    return generate_landscape_dataset_pattern(nx, ny, 'topography', 'medium', seed)

def quick_erodability(nx, ny, seed=None):
    """Quick erodability generation"""
    return generate_landscape_dataset_pattern(nx, ny, 'erodability', 'medium', seed)

def quick_precipitation(nx, ny, seed=None):
    """Quick precipitation generation"""
    return generate_landscape_dataset_pattern(nx, ny, 'precipitation', 'medium', seed)

def quick_tectonic(nx, ny, seed=None):
    """Quick tectonic uplift generation"""
    return generate_landscape_dataset_pattern(nx, ny, 'tectonic', 'medium', seed)

def quick_mixed(nx, ny, seed=None):
    """Quick mixed pattern generation"""
    return generate_landscape_dataset_pattern(nx, ny, 'mixed', 'medium', seed)

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example usage
    print("Landscape Pattern Generator - Improved Version")
    print("=" * 55)
    
    # Generate individual patterns
    nx, ny = 1024, 1024
    
    topo = quick_topography(nx, ny, seed=42)
    erod = quick_erodability(nx, ny, seed=42)
    precip = quick_precipitation(nx, ny, seed=42)
    uplift = quick_tectonic(nx, ny, seed=42)
    
    print(f"Generated patterns with shape: {topo.shape}")
    print(f"Topography range: [{topo.min():.3f}, {topo.max():.3f}]")
    print(f"Erodability range: [{erod.min():.3f}, {erod.max():.3f}]")
    print(f"Precipitation range: [{precip.min():.3f}, {precip.max():.3f}]")
    print(f"Tectonic uplift range: [{uplift.min():.3f}, {uplift.max():.3f}]")
    
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    imtopo = axes[0].imshow(topo, cmap='terrain')
    axes[0].set_title('Topography (Broader Scales)')
    axes[0].axis('off')
    
    imerod = axes[1].imshow(erod, cmap='RdBu')
    axes[1].set_title('Erodability K (Continuous)')
    axes[1].axis('off')
    
    imprecip = axes[2].imshow(precip, cmap='viridis')
    axes[2].set_title('Precipitation (Broad & Blobby)')
    axes[2].axis('off')
    
    imuplift = axes[3].imshow(uplift, cmap='magma')
    axes[3].set_title('Tectonic Uplift (Continuous)')
    axes[3].axis('off')

    plt.tight_layout()
    fig.show()

    # Interactive loop for testing
    print("\nPress Enter to generate new patterns (Ctrl+C to exit)...")
    try:
        while True:
            # input()  # Wait for Enter key
            
            topo = quick_topography(nx, ny, seed=None)
            erod = quick_erodability(nx, ny, seed=None)
            precip = quick_precipitation(nx, ny, seed=None)
            uplift = quick_tectonic(nx, ny, seed=None)
            
            imtopo.set_data(topo)
            imerod.set_data(erod)
            imprecip.set_data(precip)
            imuplift.set_data(uplift)

            # Update colormaps
            imtopo.set_clim(topo.min(), topo.max())
            imerod.set_clim(erod.min(), erod.max())
            imprecip.set_clim(precip.min(), precip.max())
            imuplift.set_clim(uplift.min(), uplift.max())

            fig.canvas.draw_idle()
            fig.canvas.start_event_loop(0.1)
            
    except KeyboardInterrupt:
        print("\nExiting...")
    
    # print("\nExample dataset creation:")
    # print("patterns, metadata = create_landscape_training_dataset(128, 128, n_patterns_per_type=100)")
    # print("# or save to disk:")
    # print("create_landscape_training_dataset(128, 128, n_patterns_per_type=100, output_dir='./landscape_dataset')")