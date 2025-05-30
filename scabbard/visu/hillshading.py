'''
These scripts provide hillshading capabilities
'''


import matplotlib.pyplot as plt
import numpy as np
import numba as nb
import scabbard as scb


from scipy.ndimage import gaussian_filter

def generate_hillshade(grid, dx=1., azimuth=315, altitude=45, z_factor=1.0, 
                          blur_sigma=0.5, contrast_enhance=1.2, multi_directional=True):
   """
   Generate a fancy, pretty hillshade from elevation data with enhanced visual appeal.
   
   This function creates a sophisticated hillshade by combining multiple lighting angles,
   applying gaussian smoothing for aesthetic appeal, and enhancing contrast for better
   visual definition of terrain features.
   
   Args:
       grid (RasterGrid object, or np.ndarray): elevation values
       dx (float): Grid spacing/resolution in same units as elevation
       azimuth (float): Primary light source azimuth angle in degrees (0-360, 315=NW)
       altitude (float): Light source altitude angle in degrees (0-90, 45=optimal)
       z_factor (float): Vertical exaggeration factor (1.0=no exaggeration)
       blur_sigma (float): Gaussian blur sigma for smoothing (0=no blur, 0.5=subtle)
       contrast_enhance (float): Contrast enhancement factor (1.0=none, >1=more contrast)
       multi_directional (bool): Whether to blend multiple light directions for softer look
       
   Returns:
       np.ndarray: Normalized hillshade array with values between 0 and 1
   """
   
   elevation = grid.Z if isinstance(grid, scb.raster.RegularRasterGrid) else grid

   # Apply optional gaussian smoothing to reduce noise and create smoother hillshade
   if blur_sigma > 0:
       elevation_smooth = gaussian_filter(elevation.astype(np.float64), sigma=blur_sigma)
   else:
       elevation_smooth = elevation.astype(np.float64)
   
   # Calculate gradients (slope) in x and y directions using central differences
   # This gives us the rate of change in elevation across the terrain
   grad_x, grad_y = np.gradient(elevation_smooth, dx)
   
   # Calculate slope magnitude and aspect from gradients
   # Slope: steepness of terrain at each point
   # Aspect: direction the slope faces (compass direction)
   slope = np.sqrt(grad_x**2 + grad_y**2)
   aspect = np.arctan2(-grad_x, grad_y)  # Note: negative grad_x for proper aspect calculation
   
   # Convert light source angles from degrees to radians
   azimuth_rad = np.radians(azimuth)
   altitude_rad = np.radians(altitude)
   
   # Apply vertical exaggeration to enhance terrain features
   slope_enhanced = np.arctan(z_factor * slope)
   
   # Calculate primary hillshade using standard illumination model
   # This simulates how light hits terrain based on slope and aspect
   hillshade_primary = (np.cos(altitude_rad) * np.cos(slope_enhanced) + 
                       np.sin(altitude_rad) * np.sin(slope_enhanced) * 
                       np.cos(azimuth_rad - aspect))
   
   if multi_directional:
       # Create additional light sources for more natural, softer illumination
       # This reduces harsh shadows and creates more appealing visualization
       
       # Secondary light from opposite direction (softer, 30% intensity)
       azimuth_secondary = azimuth + 180
       azimuth_secondary_rad = np.radians(azimuth_secondary)
       hillshade_secondary = (np.cos(altitude_rad) * np.cos(slope_enhanced) + 
                             np.sin(altitude_rad) * np.sin(slope_enhanced) * 
                             np.cos(azimuth_secondary_rad - aspect))
       
       # Tertiary light from perpendicular angle (subtle, 20% intensity)
       azimuth_tertiary = azimuth + 90
       azimuth_tertiary_rad = np.radians(azimuth_tertiary)
       hillshade_tertiary = (np.cos(altitude_rad) * np.cos(slope_enhanced) + 
                            np.sin(altitude_rad) * np.sin(slope_enhanced) * 
                            np.cos(azimuth_tertiary_rad - aspect))
       
       # Blend multiple light sources with weighted combination
       # Primary light dominates, others provide fill lighting
       hillshade = (0.7 * hillshade_primary + 
                   0.2 * hillshade_secondary + 
                   0.1 * hillshade_tertiary)
   else:
       hillshade = hillshade_primary
   
   # Apply contrast enhancement using power function
   # Values > 1 increase contrast, making terrain features more pronounced
   if contrast_enhance != 1.0:
       # Normalize to 0-1 first, apply contrast, then re-normalize
       hillshade_norm = (hillshade - hillshade.min()) / (hillshade.max() - hillshade.min())
       hillshade = np.power(hillshade_norm, 1.0/contrast_enhance)
   
   # Final normalization to ensure output is strictly between 0 and 1
   # This handles any edge cases from the mathematical operations above
   hillshade_final = (hillshade - hillshade.min()) / (hillshade.max() - hillshade.min())
   
   # Ensure no NaN or infinite values (can occur with flat terrain)
   hillshade_final = np.nan_to_num(hillshade_final, nan=0.5, posinf=1.0, neginf=0.0)
   
   return hillshade_final