/*
General grid functions
*/



// Calculate grid-wise steepest slope
__global__ void calculate_SS(float *Z, float *res, unsigned char *BC) {

    // Getting the right index
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx,adder;
    if(get_index(x, y, idx, adder, BC) == false) return;

    float SS = 0.;
    for(int j=0; j<NNEIGHBOURS; ++j){
    	int nidx;
    	if(get_neighbour(idx, adder, j, nidx) == false) continue;
    	float tS = Z[idx] - Z[nidx];
        if (tS < 0) continue;
    	tS /= DXS[j];
    	if(tS > SS) SS = tS;
    }

    res[idx] = SS;

    return;

}


// Calculate grid-wise steepest slope
__global__ void grid2val(float *arr, float val) {

    // Getting the right index
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx;
    if(get_index_raw(x, y, idx) == false) return;

    arr[idx] = val;

    return;

}







