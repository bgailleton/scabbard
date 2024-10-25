"""Top-level package for scabbard."""

__author__ = """Boris Gailleton"""
__email__ = 'boris.gailleton@univ-rennes.fr'

import sys
import platform

# Dictionary to map the OS and Python version to the wheel file
wheel_urls = {
	'Linux': {
		'310': 'https://github.com/bgailleton/scabbard/raw/refs/heads/main/wheels/pytopotoolbox/topotoolbox-3.0.1-cp310-cp310-linux_x86_64.whl',
		'311': 'https://github.com/bgailleton/scabbard/raw/refs/heads/main/wheels/pytopotoolbox/topotoolbox-3.0.1-cp311-cp311-linux_x86_64.whl',
		'312': 'https://github.com/bgailleton/scabbard/raw/refs/heads/main/wheels/pytopotoolbox/topotoolbox-3.0.1-cp312-cp312-linux_x86_64.whl'
	},
	'Darwin': {  # macOS
		'310': 'https://github.com/bgailleton/scabbard/raw/refs/heads/main/wheels/pytopotoolbox/topotoolbox-3.0.1-cp310-cp310-macosx_14_0_arm64.whl',
		'311': 'https://github.com/bgailleton/scabbard/raw/refs/heads/main/wheels/pytopotoolbox/topotoolbox-3.0.1-cp311-cp311-macosx_14_0_arm64.whl',
		'312': 'https://github.com/bgailleton/scabbard/raw/refs/heads/main/wheels/pytopotoolbox/topotoolbox-3.0.1-cp312-cp312-macosx_14_0_arm64.whl'
	},
	'Windows': {
		'310': 'https://github.com/bgailleton/scabbard/raw/refs/heads/main/wheels/pytopotoolbox/topotoolbox-3.0.1-cp310-cp310-win_amd64.whl',
		'311': 'https://github.com/bgailleton/scabbard/raw/refs/heads/main/wheels/pytopotoolbox/topotoolbox-3.0.1-cp311-cp311-win_amd64.whl',
		'312': 'https://github.com/bgailleton/scabbard/raw/refs/heads/main/wheels/pytopotoolbox/topotoolbox-3.0.1-cp312-cp312-win_amd64.whl'
	}
}

def suggest_installation():
	python_version = f"{sys.version_info.major}{sys.version_info.minor}"
	os_name = platform.system()

	print('As of today, the "official" installation procedure is Work In Progress, so I precompiled the binaries but you need to install them (couple of command lines max)')

	# Match OS names to dictionary keys
	if os_name == 'Darwin':
		os_key = 'Darwin'
	elif os_name == 'Windows':
		os_key = 'Windows'
	elif os_name == 'Linux':
		os_key = 'Linux'
	else:
		print(f"Unsupported OS: {os_name}")
		return

	# Get the appropriate wheel link
	wheel_link = wheel_urls.get(os_key, {}).get(python_version)

	if wheel_link:
		print(f"To install the appropriate wheel, run:")
		print(f"pip install {wheel_link}")
	else:
		print(f"Sorry, no wheel available for Python {python_version} on {os_name}.")





TTB_AVAILABLE = False

try:
	# Common import centralised
	import topotoolbox as ttb
	TTB_AVAILABLE = True
except:
	import os
	# Call the function when the module is imported
	suggest_installation()
	raise RuntimeError('''
pytopotoolbox is required for using scabbard. 
See above for installation instruction.
''')

# Legacy imports
from .config import *
from .enumeration import *
from .utils import *
from .shape_functions import *
from .lio import *
from .fastflood import *
from .geography import *
from .grid import *
from .Dax import *
from .Dfig import *
from .Dplot import *
from .graphflood import *
from .phineas import *
from .graphflood_helper import *
from .environment import *
from .blendplot import *
# from .local_file_picker import *


# New module-type import system
from . import _utils as ut
from . import raster
from . import riverdale
from . import steenbok
from . import riverdale as rvd
from . import steenbok as ste
from . import filters
from . import flow
from . import visu
from . import io
from . import graphflood_ui as graphflood



