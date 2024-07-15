/*
Defines the different macros
B.G. 03/2024
*/
// #include "STEENBOKPATH/includer.cu"

#ifndef MACRO_HOLDER
#define MACRO_HOLDER

// Defines the number of neighbours in every constant neighbourer
#define NNEIGHBOURS MACRO2SETNNEIGHBOURS

// Reads if I have 8 neighbours and am in D8 or not
#if NNEIGHBOURS == 8
#define ISD8
#endif




#endif