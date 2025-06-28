import json
import click
import pkg_resources
from importlib import resources 
import os

# Determine the path to the configuration file
try:
	config_fname = os.path.join(os.path.dirname(__file__), 'data', 'config.json')
except:
	print("Unable to determine config file path.")
	pass

def read_json(file_path):
	"""
	Reads a JSON file and returns its content.

	If the file is empty or not a valid JSON, it returns an empty dictionary.

	Args:
		file_path (str): The path to the JSON file.

	Returns:
		dict: The content of the JSON file, or an empty dictionary if an error occurs.

	Author: B.G.
	"""
	try:
		with open(file_path, 'r') as file:
			# Try to load the JSON content
			return json.load(file)
	except json.JSONDecodeError:
		# File is empty or not a valid JSON, return a default value
		# Return {} or [] depending on what your application expects
		return {}


def load_config():
	"""
	Loads the configuration settings from the `config.json` file.

	If the file cannot be read or is empty, it returns an empty dictionary.

	Returns:
		dict: A dictionary containing the configuration settings.

	Author: B.G.
	"""
	try:
		with open(config_fname,'r') as config_file:
			try:
				config = json.load(config_file)
			except json.JSONDecodeError:
				config = {}
	except:
		print("Unable to read config")
		config = {}
	
	return config

@click.command()
def defaultConfig():
	"""
	Resets the configuration to default values and saves them to `config.json`.

	This function is a Click command that reinitializes the 'blender' path in the
	configuration file to a default value. It prints messages indicating its actions.

	Returns:
		None

	Author: B.G.
	"""

	print("Reintialising config to default")

	# Reading the JSON file
	data = load_config()

	# Edit the data: Set default Blender path
	data['blender'] = '/home/bgailleton/code/blender/blender-4.0.1-linux-x64/blender'  # Modify this line according to your needs


	try:
		# Saving the modified data back to the JSON file
		with open(config_fname, 'w+') as file:
			json.dump(data, file, indent=4)
	except:
		print('unable to write config')

def query(paramstr = 'blender'):
	"""
	Queries a specific parameter from the configuration settings.

	Args:
		paramstr (str, optional): The name of the parameter to query. Defaults to 'blender'.

	Returns:
		Any: The value of the queried parameter.

	Author: B.G.
	"""

	# Reading the JSON file
	data = load_config()

	return data[paramstr]


