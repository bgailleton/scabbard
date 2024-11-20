# Source code for `Scabbard`

In this folder, you can find the source code of `scabbard`, splitted into multiple modules:

* Miscellaneous scripts in this folder are mostly there for legacy compatibility, they will eventually get deprecated

* utils: contains general utilities function (e.g. general statistics, array generation, OOP manipulation)

* archives: I am slowly moving my old script there as they get fully deprecated, for example, it contains the CUDA version of graphflood now moved to `taichi`

* data: persistent configuration data

* filters: general 1D and 2D filters for topography or signal processing (e.g., Gaussian, spectral, ...)

* flow: everything related to topographic analysis of river/drainage area, ...

* graphflood_ui: User interface for `graphflood`, aims to provide a universal, backend agnostic interface for graphflood (cpu,gpu,TTB,dagger,...)

* geometry: objects to manage the coordinate, index, extent, ... anything related to the spatial indexing of rasters/vectors

* io: functions to load/write data

* raster: objects to manage raster-type data (or grids)

* riverdale: GPU version of `graphflood` plus a number of GPU tools for topography

* steenbok: Numba backend for processing topography/grid

* visu: Routines for visualisation