# Source code for `Scabbard`

In this folder, you can find the source code of `scabbard`, splitted into multiple modules:

* Miscellaneous scripts in this folder are mostly there for legacy compatibility, they will eventually get deprecated

* utils: contains general utilities function (e.g. general statistics or array generation)

* archives: contains old script

* data: persistent configuration data

* filters: general 1D and 2D filters for topography or signal processing (e.g., Gaussian, spectral, ...)

* flow: everything related to topographic analysis of river/drainage area, ...

* geometry: objects to manage the coordinate, index, extent, ... anything related to the spatial indexing of rasters/vectors

* io: functions to load/write data

* raster: objects to manage raster-type data (or grids)

* riverdale: GPU version of `graphflood` plus a number of GPU tools for topography

* steenbok: Numba backend for processing topography/grid

* visu: Routines for visualisation