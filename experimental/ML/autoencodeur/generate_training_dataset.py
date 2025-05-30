'''
Generates the training dataset using Analytical solution 2D to SPL
B.G. 05-2025
'''
import numpy as np
import matplotlib.pyplot as plt
import scabbard as scb
import dagger as dag
import click as cli
import numba as nb
from perlin_noise import perlin_noise_2d
from ldnoise import generate_landscape_dataset_pattern
from razorscape import generate_landscape
import random
import os
import glob
from multiprocessing import Pool
import functools

# Number of landscape to generate (in addition to the existing ones)
NGEN = 1000

def get_next_npy_number(folder_path):
   """
   Find the highest numbered .npy file in a folder and return the next number.
   
   Args:
       folder_path (str): Path to the folder containing .npy files
       
   Returns:
       int: Next available file number
   """
   npy_files = glob.glob(os.path.join(folder_path, "*.npy"))
   if not npy_files:
       return 0
   
   numbers = []
   for file in npy_files:
       name = os.path.basename(file)
       try:
           num = int(name.split('.')[0])
           numbers.append(num)
       except ValueError:
           continue
   
   return max(numbers) + 1 if numbers else 0

# Configuration flags
save = True  # Whether to save generated landscapes

def rng_complexity():
   """
   Randomly select complexity level for 2D pattern generation.
   
   Returns:
       str: Random complexity level ('simple', 'medium', or 'complex')
   """
   return random.choice(['simple','medium','complex'])

def rng_BCs():
   """
   Randomly select boundary conditions for landscape generation.
   
   Returns:
       str: Random boundary condition type
   """
   return random.choice(['periodic_EW','periodic_NS','4edges'])

def generate_single_landscape(args):
   """
   Generate a single landscape with random parameters.
   
   Args:
       args (tuple): Contains (iteration_number, base_iteration, nx, ny, dx, save_flag)
       
   Returns:
       str: Status message for the generated landscape
   """
   it, baseit, nx, ny, dx, save_flag = args
   
   # Generate initial topography using dataset pattern generator
   initopo = generate_landscape_dataset_pattern(nx, ny, pattern_type='topography', 
                                              complexity=rng_complexity(), seed=None)
   
   # Generate precipitation pattern
   precipitations = generate_landscape_dataset_pattern(nx, ny, pattern_type='precipitation', 
                                                     complexity=rng_complexity(), seed=None)
   # Scale precipitation with random range
   range_P = random.uniform(0.05, 0.8)
   precipitations *= range_P
   precipitations += (1 - range_P/2)
   
   # Generate erodability modifier (K modifier)
   Kmod = generate_landscape_dataset_pattern(nx, ny, pattern_type='erodability', 
                                           complexity=rng_complexity(), seed=None)
   # Scale K modifier with random range
   range_K = random.uniform(0.05, 1.9)
   Kmod *= range_K
   Kmod += (1 - range_K/2)
   
   # Generate tectonic uplift pattern
   UE = generate_landscape_dataset_pattern(nx, ny, pattern_type='tectonic', 
                                         complexity=rng_complexity(), seed=None)
   # Scale uplift with random range
   range_UE = random.uniform(0.5, 1.)
   UE *= range_UE
   UE += (1 - range_K/2)  # Note: Uses range_K, might be intentional
   UE *= 1e-3  # Scale to appropriate uplift rate
   
   # Generate random erosion parameters
   tm = random.uniform(0.3, 0.8)  # m parameter for stream power law
   tn = tm/random.uniform(0.1, 0.5)  # n parameter for stream power law
   
   # Select random boundary conditions
   tbc = rng_BCs()
   
   print(f'Generating landscape {it}...')
   
   # Generate the landscape using the razorscape library
   topo = generate_landscape(initopo, dx=dx, Urate=UE, base_K=1e-4, Kmod=Kmod, 
                           precipitations=precipitations, uplift=UE, m=tm, n=tn, 
                           boundary_type=tbc, minimum_size=32)
   
   print(f'Landscape {it} generation complete')
   
   # Save the normalized landscape if save flag is enabled
   if save_flag:
       # Normalize topography to [0,1] range
       topo_normalized = (topo - topo.min())/(topo.max() - topo.min())
       np.save(f'./dataset/{str(baseit + it)}.npy', topo_normalized)
   
   return f"Landscape {it} processed successfully"

def main():
   """
   Main function to generate landscape dataset using multiprocessing.
   """
   # Landscape geometry parameters
   nx, ny = 2048, 2048  # Grid dimensions
   dx = 100.  # Grid spacing in meters
   
   # Generate initial topography for visualization setup
   initopo = perlin_noise_2d(nx, ny, scale=0.01, octaves=5, persistence=0.5, lacunarity=2.0, seed=0)
   
   # Get the starting file number for the dataset
   baseit = get_next_npy_number("./dataset")
   print(f"Starting dataset generation from file number: {baseit}")
   
   # Prepare arguments for multiprocessing
   # Each process will receive: (iteration, base_iteration, nx, ny, dx, save_flag,)
   args_list = [(it, baseit, nx, ny, dx, save) for it in range(NGEN)]
   
   # Create process pool with 8 workers
   print("Starting multiprocessed landscape generation with 8 processes...")
   with Pool(processes=8) as pool:
       # Map the landscape generation function across all iterations
       results = pool.map(generate_single_landscape, args_list)
   
   # Print completion status
   print("All landscapes generated successfully!")
   for result in results[:5]:  # Print first 5 results as sample
       print(result)
   
   print(f"Generated {len(results)} landscapes total")

# Execute main function when script is run directly
if __name__ == "__main__":
   main()

