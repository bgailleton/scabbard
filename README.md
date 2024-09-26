# Scabbard

<!-- 
.. image:: https://img.shields.io/pypi/v/scabbard.svg
        :target: https://pypi.python.org/pypi/scabbard

.. image:: https://readthedocs.org/projects/scabbard/badge/?version=latest
        :target: https://scabbard.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status -->



Python package to design, use or visualise topographic analysis, landscape evolution models and hydro/morphodynamics simulations. Among other, home of [GraphFlood](https://egusphere.copernicus.org/preprints/2024/egusphere-2024-1239/) (`CPU` and `GPU`) and [`CHONK`](https://gmd.copernicus.org/articles/17/71/2024/).


### Built on the shoulder of giants

`scabbard` started as a personal codebase and evolved toward a fully-fledge numerical Framework. It uses the robust and battle-tested scientific `python` stack (e.g. `numpy, scipy, numba, ...`) and geospatial `python` (e.g. `rasterio`, `geopandas`, `libgdal`, ...). It also relies on multiple domain-specific libraries to built a future-proof and community-compatible backend: [`libtopotoolbox` and `pytopotoolbox`](https://github.com/TopoToolbox), [`fastscapelib`](https://fastscapelib.readthedocs.io/en/latest/), [`LSDTopoTool`](https://lsdtopotools.github.io/) as well as my own collection of `c++` and `numba` routines. All the GPU backends use [`taichi`](https://docs.taichi-lang.org/) to ensure cross-platform/hardware compatibility and user-friendly code.


* Free software: MIT license
<!-- * Documentation: https://scabbard.readthedocs.io. -->

## How to install

I am currently working on an easier installation process. In the meantime, using a `conda` environment:

```
mamba install numpy scipy matplotlib ipympl jupyterlab rasterio numba cmcrameri plotly nicegui daggerpy pip geopandas
pip install taichi pyscabbard
```

You also need to follow the installation procedure of [`pytopotoolbox`](https://github.com/TopoToolbox/pytopotoolbox).  


## Usage

TODO

## Features

* TODO

## Credits


This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
