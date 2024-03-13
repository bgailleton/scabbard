import json
import click
import pkg_resources
from importlib import resources 
import os

try:
	# Path to your JSON file
	# with resources.open_text('scabbard', 'config.json') as config_file:
		# config_fname = config_file.name
	# config_fname = pkg_resources.resource_filename('scabbard', 'data/config.json')
	config_fname = os.path.join(os.path.dirname(__file__), 'data', 'config.json')

except:
	print("Unable to read config")
	pass

# Path to your JSON file
def read_json(file_path):
	try:
		with open(file_path, 'r') as file:
			# Try to load the JSON content
			return json.load(file)
	except json.JSONDecodeError:
		# File is empty or not a valid JSON, return a default value
		# Return {} or [] depending on what your application expects
		return {}


def load_config():
	# Access the resource within the 'data' package/directory
	try:
		with open(config_fname,'r') as config_file:
			# print("config_file",config_file)
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

	print("Reintialising config to default")

	# data = {}
	# Reading the JSON file
	data = load_config()

	# Edit the data
	data['blender'] = '/home/bgailleton/code/blender/blender-4.0.1-linux-x64/blender'  # Modify this line according to your needs
	# data['blender'] = 'afslkdglasdfgsjldf'  # Modify this line according to your needs


	try:
		# Saving the modified data back to the JSON file
		with open(config_fname, 'w+') as file:
			json.dump(data, file, indent=4)
	except:
		print('unable to write config')

def query(paramstr = 'blender'):

	# Reading the JSON file
	data = load_config()

	return data[paramstr]


