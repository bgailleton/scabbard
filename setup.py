#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages
from setuptools.command.install import install


class PostInstallCommand(install):
	"""Post-installation for installation mode."""
	def run(self):
		# Call superclass's run method
		install.run(self)
		
		# Your post-installation logic here
		os.system("scb-reset-config")
		print("KDSFDSJKFSLDKHF")



with open('README.rst') as readme_file:
		readme = readme_file.read()

with open('HISTORY.rst') as history_file:
		history = history_file.read()

requirements = ['Click>=7.0', ]

test_requirements = [ ]

setup(
	author="Boris Gailleton",
	author_email='boris.gailleton@univ-rennes1.fr',
	python_requires='>=3.6',
	classifiers=[
			'Development Status :: 2 - Pre-Alpha',
			'Intended Audience :: Developers',
			'License :: OSI Approved :: MIT License',
			'Natural Language :: English',
			'Programming Language :: Python :: 3',
			'Programming Language :: Python :: 3.6',
			'Programming Language :: Python :: 3.7',
			'Programming Language :: Python :: 3.8',
	],
	description="high-level python package for the DAGGER suite",
	entry_points={
		'console_scripts': [
				'scb-baseplot=scabbard.phineas:simplemapwizard' ,
				'scb-graphflood=scabbard.phineas:graphflood_basic' ,
				'scb-debugger=scabbard.phineas:_debug_1' ,
				'scb-reset-config=scabbard.config:defaultConfig' ,
		],
	},
	install_requires=requirements,
	license="MIT license",
	long_description=readme + '\n\n' + history,
	include_package_data=True,
	keywords='scabbard',
	name='pyscabbard',
	packages=find_packages(include=['scabbard', 'scabbard.*']),
	package_data={
		'': ['data/config.json'],  # Include your config.json file
	},
	cmdclass={
		'install': PostInstallCommand,
	},
	test_suite='tests',
	tests_require=test_requirements,
	url='https://github.com/bgailleton/scabbard',
	version='0.0.6',
	zip_safe=False,
)
