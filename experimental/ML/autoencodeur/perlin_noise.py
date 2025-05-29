import numpy as np
from numba import njit
import math

@njit
def fade(t):
	"""Fade function for smooth interpolation (6t^5 - 15t^4 + 10t^3)"""
	return t * t * t * (t * (t * 6 - 15) + 10)

@njit
def lerp(a, b, t):
	"""Linear interpolation"""
	return a + t * (b - a)

@njit
def grad(hash_val, x, y):
	"""Gradient function for 2D Perlin noise"""
	h = hash_val & 3
	if h == 0:
		return x + y
	elif h == 1:
		return -x + y
	elif h == 2:
		return x - y
	else:
		return -x - y

@njit
def perlin_2d_single(x, y, perm):
	"""Generate single Perlin noise value at coordinates (x, y)"""
	# Find unit grid cell containing point
	X = int(math.floor(x)) & 255
	Y = int(math.floor(y)) & 255
	
	# Get relative coordinates within cell
	x -= math.floor(x)
	y -= math.floor(y)
	
	# Compute fade curves for x and y
	u = fade(x)
	v = fade(y)
	
	# Hash coordinates of 4 grid corners
	A = perm[X] + Y
	AA = perm[A]
	AB = perm[A + 1]
	B = perm[X + 1] + Y
	BA = perm[B]
	BB = perm[B + 1]
	
	# Blend results from 4 corners
	res = lerp(
		lerp(grad(perm[AA], x, y), grad(perm[BA], x - 1, y), u),
		lerp(grad(perm[AB], x, y - 1), grad(perm[BB], x - 1, y - 1), u),
		v
	)
	
	return res

@njit
def generate_permutation_table(seed):
	"""Generate permutation table for Perlin noise"""
	np.random.seed(seed)
	p = np.arange(256, dtype=np.int32)
	np.random.shuffle(p)
	perm = np.zeros(512, dtype=np.int32)
	for i in range(256):
		perm[i] = p[i]
		perm[i + 256] = p[i]
	return perm

@njit
def perlin_noise_2d(nx, ny, scale=0.1, octaves=1, persistence=0.5, 
				   lacunarity=2.0, seed=0):
	"""
	Generate 2D Perlin noise array normalized between 0 and 1
	
	Parameters:
	-----------
	nx : int
		Number of columns (width)
	ny : int
		Number of rows (height)  
	scale : float
		Controls zoom level (smaller = more zoomed out)
	octaves : int
		Number of noise layers to combine
	persistence : float
		How much each octave contributes (amplitude multiplier)
	lacunarity : float
		How much detail is added each octave (frequency multiplier)
	seed : int
		Random seed for reproducible results
		
	Returns:
	--------
	numpy.ndarray
		2D array of Perlin noise values normalized between 0 and 1
	"""
	# Generate permutation table
	perm = generate_permutation_table(seed)
	
	# Initialize output array
	noise = np.zeros((ny, nx), dtype=np.float64)
	
	# Track min/max for normalization
	min_val = float('inf')
	max_val = float('-inf')
	
	for y in range(ny):
		for x in range(nx):
			amplitude = 1.0
			frequency = scale
			noise_val = 0.0

			max_amplitude = 0.0
			
			# Combine octaves
			for octave in range(octaves):
				sample_x = x * frequency
				sample_y = y * frequency
				
				perlin_val = perlin_2d_single(sample_x, sample_y, perm)
				noise_val += perlin_val * amplitude
				max_amplitude += amplitude
				
				amplitude *= persistence
				frequency *= lacunarity
			
			# Normalize by max possible amplitude for this point
			noise_val /= max_amplitude
			noise[y, x] = noise_val
			
			# Track global min/max
			if noise_val < min_val:
				min_val = noise_val
			if noise_val > max_val:
				max_val = noise_val
	
	# Normalize to [0, 1] range
	if max_val != min_val:
		noise = (noise - min_val) / (max_val - min_val)
	else:
		noise.fill(0.5)  # If all values are the same
	
	return noise

# Example usage and test function
def test_perlin_noise():
	"""Test the Perlin noise generator"""
	import matplotlib.pyplot as plt
	
	# Generate noise with different parameters
	noise1 = perlin_noise_2d(512, 512, scale=0.01, octaves=8, seed=42)
	noise2 = perlin_noise_2d(512, 512, scale=0.05, octaves=4, persistence=0.5, seed=42)
	noise3 = perlin_noise_2d(512, 512, scale=0.1, octaves=6, persistence=0.6, lacunarity=2.5, seed=42)
	
	fig, axes = plt.subplots(1, 3, figsize=(15, 5))
	
	axes[0].imshow(noise1, cmap='terrain', origin='lower')
	axes[0].set_title('Single Octave (scale=0.05)')
	axes[0].axis('off')
	
	axes[1].imshow(noise2, cmap='terrain', origin='lower')  
	axes[1].set_title('4 Octaves (scale=0.1)')
	axes[1].axis('off')
	
	axes[2].imshow(noise3, cmap='terrain', origin='lower')
	axes[2].set_title('6 Octaves (scale=0.2, lacunarity=2.5)')
	axes[2].axis('off')
	
	plt.tight_layout()
	plt.show()
	
	print(f"Noise1 range: [{noise1.min():.3f}, {noise1.max():.3f}]")
	print(f"Noise2 range: [{noise2.min():.3f}, {noise2.max():.3f}]")
	print(f"Noise3 range: [{noise3.min():.3f}, {noise3.max():.3f}]")

if __name__ == "__main__":
	test_perlin_noise()