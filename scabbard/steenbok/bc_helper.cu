/*
This files contians routine to read and interpret boundary conditions
*/
// #include "STEENBOKPATH/includer.cu"

#ifndef BC_HELPER
#define BC_HELPER

/*
Reminder, I am using the DAGGER convention
// Cannot flow at all = nodata
NO_FLOW = 0,

// Internal Node (can flow in every directions)
FLOW = 1,

// Internal Node (can flow in every directions) BUT neighbours a special flow
// condition and may need specific care
FLOW_BUT = 2,

// flow can out there but can also flow to downstream neighbours
CAN_OUT = 3,

// flow can only out from this cell
OUT = 4,

// Not only flow HAS to out there: neighbouring flows will be drained there no
// matter what
FORCE_OUT = 5,

// Flows through the cell is possible, but the cell CANNOT out fluxes from
// this boundary (reserved to model edges, internal boundaries wont give to
// nodata anyway)
CANNOT_OUT = 6,

// Flow can only flow to potential receivers
IN = 7,

// Forced INFLOW: flow will flow to all neighbours (except other FORCE_IN)
FORCE_IN = 8,

// periodic border
PERIODIC_BORDER = 9
*/

namespace BC{

// determines if a node can receive fluxes
__device__ bool can_receive(unsigned char tbc){
	if(tbc == 0 || tbc == 7 || tbc == 8) return false;
	return true;

}


// determines if a node can give fluxes
__device__ bool can_give(unsigned char tbc){
	if(tbc == 0 || tbc == 5 || tbc == 4) return false;
	return true;

}

__device__ bool can_out(unsigned char tbc){
	if(tbc == 3 || tbc == 4) return true;
	return false;
}



__device__ bool nodata(unsigned char tbc){
	if(tbc == 0) return true;
	return false;
}








































}; // end of namespace


#endif