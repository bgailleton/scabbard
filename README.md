# Scabbard

<!-- 
.. image:: https://img.shields.io/pypi/v/scabbard.svg
        :target: https://pypi.python.org/pypi/scabbard

.. image:: https://readthedocs.org/projects/scabbard/badge/?version=latest
        :target: https://scabbard.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status -->



Python package for topographic analysis, landscape evolution modelling and hydro/morphodynamics simulations


* Free software: MIT license
<!-- * Documentation: https://scabbard.readthedocs.io. -->

## How to install

I need to clean/properly set the dependencies. So far:

```
mamba install numpy scipy matplotlib ipympl jupyterlab rasterio numba cmcrameri plotly nicegui daggerpy
pip install taichi pyscabbard
```


## Usage

TODO

## Features

* TODO


## Experimental

Among the experimental features is a tentative link to Blender to make nice 3D plots. The latter will only get activated if called from blender.
Only tested on Ubuntu, it requires the following steps:

1. identify the `python` version of your blender version
2. create a new env with this python version
3. install all the packages you need either with conda or pip.
4. AFTER EVERY NEW ADDITIONS (of one or multiple packages at once) YOU'LL NEED TO LINK THE PACKAGES TO BLENDER:

`ln -s /path/to/your/environment/site-packages/* ~/.config/blender/3.X/scripts/addons/modules/`

Where you need to adapts the paths and version number

## Credits


This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
