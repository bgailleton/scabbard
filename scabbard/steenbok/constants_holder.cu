/*
Contains all the constant required code-wise
B.G. 03/2024
*/
// #include "STEENBOKPATH/includer.cu"

#ifndef CONSTANT_HOLDER
#define CONSTANT_HOLDER

// Constants for managing the neighbours and spatial dimensions
__constant__ float DXS[NNEIGHBOURS];
__constant__ float DYS[NNEIGHBOURS];
__constant__ int NEIGHBOURERS[9][NNEIGHBOURS];
__constant__ float CELLAREA;
__constant__ float DX;
__constant__ float DY;
__constant__ int NX;
__constant__ int NY;
__constant__ int NXY;
__constant__ int NODATA;

// Constants to manage time steps
__constant__ float DT_HYDRO;
__constant__ float DT_MORPHO;

// Hydrodynamic constants
__constant__ float MANNING;
__constant__ float RHO_WATER;
__constant__ float RHO_SEDIMENT;

// Morphodynamic constants
__constant__ float GRAVITY;
__constant__ float E_MPM;
__constant__ float TAU_C;

// Other
__constant__ float QSOUT_MULT;


#endif