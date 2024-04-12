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
    	if(get_neighbour(idx, adder, j, nidx, BC) == false) continue;
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

// Calculate grid-wise steepest slope
__global__ void copygrid(float *arr, float* arr2) {

    // Getting the right index
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx;
    if(get_index_raw(x, y, idx) == false) return;

    arr[idx] = arr2[idx];

    return;

}


// Calculate grid-wise steepest slope
__global__ void grid2val_i32(int *arr, float val) {

    // Getting the right index
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx;
    if(get_index_raw(x, y, idx) == false) return;

    arr[idx] = val;

    return;

}



// Calculate grid-wise steepest slope
__global__ void uplift(float *arr, unsigned char *BC, float val) {

    // Getting the right index
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx,adder;
    if(get_index(x, y, idx, adder, BC) == false) return;
    if(BC::can_out(BC[idx])) return;

    arr[idx] += val;

    return;

}

// Calculate grid-wise steepest slope
__global__ void uplift_nolastrow(float *arr, unsigned char *BC, float val) {

    // Getting the right index
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx,adder;
    if(get_index(x, y, idx, adder, BC) == false) return;
    if(BC::can_out(BC[idx])) return;
    if(y >= NY - 2) return;

    arr[idx] += val;

    return;

}


__global__ void checkGrid4nan(float *arr, int* nans){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx;
    if(get_index_raw(x, y, idx) == false) return;
    if(isnan(arr[idx])) atomicAdd(&nans[0],1);
}



