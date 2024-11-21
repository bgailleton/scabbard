=======
History
=======


0.0.13 (2024-11-20)
-----------------------
* Adding an option for changing the exponent in Riverdale friction equation (e.g. migrating from manning to chezy)

0.0.12 (2024-11-20)
-----------------------

* Adapting the install process to pip topotoolbox
* Migrating to new ttb
* Minor fixes and additions
* Adding the start of a multi-backend unified graphflood interface


0.0.11 (2024-11-05)
-----------------------

* Loads of minor fixes in the install process
* Adding an hdf5 io/helper
* Refactor part of the ray tracing 3D plotter
* Adding orthographic projection

0.0.9 (2024-10-25)
--------------------------

* More fixes on the setup process
* Added a simple GPU ray tracing engine for 3D plots
* Started a unified interface for graphflood 

0.0.8 (2024-10-15)
--------------------------

* Cleaning the install process
* Removing problematic legacy imports
* Trimming requirements
* Deprecating older tools relying on packages I do not use anymore

0.0.5 - 0.0.7 (2024-10-15)
--------------------------

* Total refactoring. As Simple as that.
* `scabbard` is the main tool now, it uses multiple backends `DAGGER`, `pytopotoolbox`, `fastscapelib`, `numba` and `taichi`


0.0.3 - 0.0.4 (2023-10-23)
--------------------------

* Fixing couple of bugs on the grid
* Experimental support for DAGGER's experimental stuff
* Adding environment object (WIP, future main structure)

0.0.2 (2023-07-31)
------------------

* Adding drainage divide quick extraction tools
* Fixing sea_level/Z0 stuff 

0.0.1 (2023-07-25)
------------------

* First release on PyPI.
* Adding tools for quick river extraction
* Started a big behind-the-scene refactoring and standardisation (invisible at top level)
* Maintenance and bug fixes
