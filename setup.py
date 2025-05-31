"""
Setup script for PyScabbard - A suite of hydrodynamic, topographic analysis, 
landscape evolution modeling and visualization tools.

This setup script handles package installation, dependencies, and post-installation
configuration for the Scabbard scientific computing package.


Authors:
	- B.G. (last modification 05/2025)
"""

import os
from setuptools import setup, find_packages
from setuptools.command.install import install

# Package metadata
VERSION = '0.0.16'
PACKAGE_NAME = 'pyscabbard'
AUTHOR = 'Boris Gailleton'
AUTHOR_EMAIL = 'boris.gailleton@univ-rennes.fr'
DESCRIPTION = ("Suite of Hydrodynamic, topographic analysis, "
               "Landscape Evolution model and visualisation tools")
URL = 'https://github.com/bgailleton/scabbard'
LICENSE = 'MIT license'

# Python version requirements
PYTHON_REQUIRES = '>=3.10'

# Package dependencies
INSTALL_REQUIRES = [
    'Click>=7.0',           # Command line interface creation
    'numpy>=2',             # Numerical computing
    'numba>=0.60',          # JIT compilation for performance
    'daggerpy>=0.0.14',     # c++ engine DAGGER - behind graphflood and all
    'matplotlib',           # Plotting and visualization
    'taichi',               # High-performance GPU framework
    'rasterio',             # Geospatial raster I/O
    'scipy',                # Scientific computing
    'cmcrameri',            # Scientific colormaps
    'h5py',                 # HDF5 file format support
    'topotoolbox',          # Topographic analysis tools - libtopotoolbox
]

# Development and testing dependencies
TEST_REQUIRES = [
    'pytest>=6.0',          # Testing framework
    'pytest-cov',           # Coverage reporting
    # Add other test dependencies as needed
]

# Console script entry points - command line tools provided by the package
CONSOLE_SCRIPTS = [
    'scb-baseplot=scabbard.visu.nice_terrain:cli_nice_terrain',
    'scb-crop=scabbard.raster.std_raster_cropper:std_crop_raster',
    'scb-graphflood=scabbard.phineas:graphflood_basic',
    'scb-reset-config=scabbard.config:defaultConfig',
    'scb-visu2D=scabbard.phineas:visu2Dnpy',
    'scb-quick-hydro=scabbard.phineas:GPUgraphflood',
    'scb-reachBC=scabbard.flow.preprocess_bc_reach_wizard:_cli_reach_wizard',
    'scb-genexr=scabbard._utils.converter_exr:cli_convert_to_EXR',
    'scbftgd-load=scabbard.gdcom:simple_load',
]

# Package classifiers for PyPI
CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Hydrology',
    'Topic :: Scientific/Engineering :: Visualization',
    'License :: OSI Approved :: MIT License',
    'Natural Language :: English',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3.12',
    'Operating System :: OS Independent',
]


class PostInstallCommand(install):
    """
    Custom post-installation command to run configuration setup.
    
    This command runs after the package is installed to perform necessary
    initialization tasks such as resetting configuration files.
    """
    
    def run(self):
        """Execute the post-installation steps."""
        # Call the parent class installation method first
        install.run(self)
        
        try:
            # Reset configuration to default values
            print("Running post-installation configuration...")
            exit_code = os.system("scb-reset-config")
            
            if exit_code == 0:
                print("✓ Scabbard configuration initialized successfully!")
            else:
                print("⚠ Warning: Configuration initialization may have failed")
                print("  You can manually run 'scb-reset-config' later if needed")
                
        except Exception as e:
            print(f"⚠ Warning: Post-installation setup encountered an issue: {e}")
            print("  The package should still work, but you may need to run")
            print("  'scb-reset-config' manually to complete setup")


def read_file(filename):
    """Read and return the contents of a file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: {filename} not found")
        return ""
    except Exception as e:
        print(f"Warning: Could not read {filename}: {e}")
        return ""


def main():
    """Main setup function."""
    # Read long description from README and HISTORY files
    readme = read_file('README.md')
    history = read_file('HISTORY.rst')
    long_description = readme + '\n\n' + history if readme and history else readme
    
    setup(
        # Basic package information
        name=PACKAGE_NAME,
        version=VERSION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        description=DESCRIPTION,
        long_description=long_description,
        long_description_content_type='text/markdown',
        url=URL,
        license=LICENSE,
        
        # Package discovery and requirements
        packages=find_packages(),
        python_requires=PYTHON_REQUIRES,
        install_requires=INSTALL_REQUIRES,
        tests_require=TEST_REQUIRES,
        
        # Package data and resources
        include_package_data=True,
        package_data={
            'scabbard': [
                'data/*.json',      # Configuration files
                # 'steenbok/*.cu',  # CUDA files (uncomment if needed)
            ],
        },
        
        # Entry points for command-line tools
        entry_points={
            'console_scripts': CONSOLE_SCRIPTS,
        },
        
        # Custom installation commands
        cmdclass={
            'install': PostInstallCommand,
        },
        
        # PyPI metadata
        classifiers=CLASSIFIERS,
        keywords=['hydrology', 'topography', 'landscape-evolution', 'scientific-computing'],
        
        # Testing configuration
        test_suite='tests',
        zip_safe=False,  # Required for packages with data files
    )


if __name__ == '__main__':
    main()