/*
This files contians routine to read and interpret boundary conditions
*/
// #include "STEENBOKPATH/includer.cu"

#ifndef NEIGHBOURER
#define NEIGHBOURER


// return false if indice is incorrect, otherwise set idx to the 1D index of the grid
__device__ bool get_index_raw(int x, int y, int& idx){
	if (x >= NX || y >= NY) return false;
	idx = y * NX + x;
	return true;
}

// return false if indice is incorrect or if nodata, otherwise set idx to the 1D index of the grid
__device__ bool get_index_check(int x, int y, int& idx, unsigned char *BC){
	if (x >= NX || y >= NY) return false;
	idx = y * NX + x;
	if(BC[idx] == 0) return false;
	return true;
}


// return false if indice is incorrect, otherwise set idx to the 1D index of the grid
__device__ bool get_index(int x, int y, int& idx, int& adder, unsigned char *BC){

	if(get_index_check( x,  y,  idx, BC) == false) return false;

	unsigned char tbc = BC[idx];

    if(tbc == 1){
        adder = 0;
        return true;
    }else if(tbc == 0)
        return false;
    else if(x == 0 && y == 0){
        adder = 1;
        return true;
    }else if(idx < NX-1){
        adder = 2;
        return true;
    }else if(idx == NX-1){
        adder = 3;
        return true;
    }else if(x == 0 && y != NY-1){
        adder = 4;
        return true;
    }else if(x == NX-1 && y != NY-1){
        adder = 5;
        return true;
    }else if(x == 0 && y == NY-1){
        adder = 6;
        return true;
    }else if(x != NX-1 && y == NY-1){
        adder = 7;
        return true;
    }else if(x == NX-1 && y == NY-1){
        adder = 8;
        return true;
    }

	return true;
}


__device__ bool get_neighbour(int idx, int adder, int j, int& nidx){
	int tadd = NEIGHBOURERS[adder][j];
	if( tadd == NODATA) return false;
	nidx = idx + tadd;
	return true;
}





















#endif