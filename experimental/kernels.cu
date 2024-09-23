// kernels.cu
#define NNEIGHBOURS MACROTOREPLACE_D4D8
#if NNEIGHBOURS == 8
#define ISD8
#endif
#include <stdint.h>

__constant__ float dXs[NNEIGHBOURS];
__constant__ float dYs[NNEIGHBOURS];
__constant__ int neighbourers[9][NNEIGHBOURS];
__constant__ float dt_hydro;
__constant__ float dt_morpho;
__constant__ float cellarea;
__constant__ float manning;
__constant__ float rho_water;
__constant__ float gravity;
__constant__ float E_MPM;
__constant__ float tau_c;
__constant__ float alphalat;
__constant__ int nx;
__constant__ int ny;
__constant__ int nodata;
__constant__ float Qsout_mult;

__device__ float generate_random(unsigned int seed) {
    unsigned int random = seed;
    random ^= random << 13;
    random ^= random >> 17;
    random ^= random << 5;
    return float(random) / 4294967295.0f; // Divide by 2^32-1
}


__device__ bool can_out(uint8_t tbc){
    if(tbc == 3 || tbc == 4 || tbc == 5) return true;
    return false;
}


// Internal function managing the neighbouring function of the boundary conditions
__device__ bool get_indices(int& idx, int& idx_neighbourer, int col, int row, uint8_t *BC){

    if(idx >= nx*ny || col >= nx || row >= ny) return false;

    uint8_t tbc = BC[idx];

    if(tbc == 1){
        idx_neighbourer = 0;
        return true;
    }else if(tbc == 0)
        return false;
    else if(col == 0 && row == 0){
        idx_neighbourer = 1;
        return true;
    }else if(idx < nx-1){
        idx_neighbourer = 2;
        return true;
    }else if(idx == nx-1){
        idx_neighbourer = 3;
        return true;
    }else if(col == 0 && row != ny-1){
        idx_neighbourer = 4;
        return true;
    }else if(col == nx-1 && row != ny-1){
        idx_neighbourer = 5;
        return true;
    }else if(col == 0 && row == ny-1){
        idx_neighbourer = 6;
        return true;
    }else if(col != nx-1 && row == ny-1){
        idx_neighbourer = 7;
        return true;
    }else if(col == nx-1 && row == ny-1){
        idx_neighbourer = NNEIGHBOURS;
        return true;
    }


}


//#####################################################################
// Graphflood - Hydrodynamic 
//#####################################################################


// Add specific input points of water
__global__ void add_Qwin_local(int *indices, float *values, float *QwA, float *QwB, const int sizid) {
    // Getting the right IF
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    // int y = threadIdx.y + blockIdx.y * blockDim.y;


    if(x >= sizid) return;

    QwA[indices[x]] += values[x];
    QwB[indices[x]] += values[x];

}

// Add specific input points of water
__global__ void add_Qwin_global(float *QwA, float val) {
    // Getting the right IF
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = y * nx + x;
    if(x >= nx || y >= ny || idx >= nx * ny) return;

    QwA[idx] += val;
}


// Add specific input points of water
__global__ void multiply_array(float mult, float *values, const int sizid) {
    // Getting the right IF
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = y * nx + x;

    if(idx >= sizid) return;

    values[idx] *= mult;

}


// transfers Qwin on a node to node basis
__global__ void compute_Qwin(float *hw, float *Z, float *QwA, float *QwB, uint8_t *BC) {

    // Getting the right IF
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = y * nx + x;

    int idnei;
    if(get_indices(idx, idnei, x, y, BC) == false) return;

    float surface_idx = hw[idx] + Z[idx];

    float weights[NNEIGHBOURS];

    for(int i=0;i<NNEIGHBOURS;++i){
        weights[i] = 0.;
    }

    float sumWeights = 0.;

    while(sumWeights == 0.){

        for(int i=0;i<NNEIGHBOURS;++i){

            // idx of the neighbours
            int adder = neighbourers[idnei][i];
            if(adder == nodata) continue;
            int nidx = idx + adder;
            
            // calculating local weight (i.e. Sw * dy)
            float tw = surface_idx - (hw[nidx] + Z[nidx]);

            if(tw<0) continue; // aborting if higher neighbour
            
            // finishing slope calc
            tw *= dYs[i];
            tw /= dXs[i];

            // saving weight
            weights[i] = tw;

            // summing all of em'
            sumWeights += tw;
        }

        if(sumWeights == 0.) {
            hw[idx] += 0.0001;
            surface_idx += 0.0001;
        }


    }

    // if(sumWeights <= 0) {
    //     atomicAdd(&QwB[idx], QwA[idx]);
    //     return;
    // }

    for(int i=0;i<NNEIGHBOURS;++i){
        if(weights[i] <= 0) continue;
        int adder = neighbourers[idnei][i];
        if(adder == nodata) continue;
        int nidx = idx + adder;
        atomicAdd(&QwB[nidx], QwA[idx] * weights[i]/sumWeights);
    }

}

// intermediate function required tofinalise the new Qwin
__global__ void swapQwin(float *QwA, float *QwB) {

    // Getting the right IF
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = y * nx + x;

    if(idx >= nx * ny || x >= nx || y >= ny) return;
    
    QwA[idx] = QwB[idx];
    QwB[idx] = 0.;

}


#ifdef ISD8
// compute Qwout using Gailleton et al 2024
__global__ void compute_Qwout(float *hw, float *Z, float *QwB, uint8_t *BC) {

    // Getting the right IF
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = y * nx + x;

    int idnei;
    if(get_indices(idx, idnei, x, y, BC) == false) return;

    float surface_idx = hw[idx] + Z[idx];

    float SS = -1;
    float SSdy = 1.;

    for(int i=0;i<NNEIGHBOURS;++i){

        // idx of the neighbours
        int adder = neighbourers[idnei][i];
        if(adder == nodata) continue;
        int nidx = idx + adder;
        
        // calculating local weight (i.e. Sw * dy)
        float ts = surface_idx - (hw[nidx] + Z[nidx]);

        if(ts<0) continue; // aborting if higher neighbour
        
        // finishing slope calc
        ts /= dXs[i];

        if(ts > SS){
            SS = ts;
            SSdy = dYs[i];
        }
    }

    if(SS == -1) return;

    QwB[idx] = SSdy/manning * hw[idx] * std::sqrt(SS);

}
#else
// compute Qwout using Gailleton et al 2024
__global__ void compute_Qwout(float *hw, float *Z, float *QwB, uint8_t *BC) {

    // Getting the right IF
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = y * nx + x;

    int idnei;
    if(get_indices(idx, idnei, x, y, BC) == false) return;

    float surface_idx = hw[idx] + Z[idx];

    QwB[idx] = 0.;


    for(int i=0;i<NNEIGHBOURS;++i){

        // idx of the neighbours
        int adder = neighbourers[idnei][i];
        if(adder == nodata) continue;
        int nidx = idx + adder;
        
        // calculating local weight (i.e. Sw * dy)
        float ts = surface_idx - (hw[nidx] + Z[nidx]);

        if(ts<0) continue; // aborting if higher neighbour
        
        // finishing slope calc
        ts /= dXs[i];

        QwB[idx] += dYs[i]/manning * hw[idx] * std::sqrt(ts);;
    }

}
#endif


// Increment water function of the divergence of the fluxes
__global__ void increment_hw(float *hw, float *Z,float *QwA, float *QwB, uint8_t *BC) {

    // Getting the right IF
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = y * nx + x;

    if(idx >= nx*ny || x >= nx || y >= ny) return;

    if(BC[idx]  != 1) {
        QwA[idx] = 0.;
        return;
    };

    float dhw = (QwA[idx] - QwB[idx])/ cellarea * dt_hydro;
    hw[idx] = max(0., hw[idx] + dhw);

    

}



//###############################################################################################
// Graphflood - Morphodynamics - pure MPM euqations: infinite pile of sand with mono grain size
//###############################################################################################



// Add specific input points of sediments
__global__ void add_Qsin_local(int *indices, float *values, float *QsA, float *QsB, const int sizid) {
    // Getting the right IF
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    // int y = threadIdx.y + blockIdx.y * blockDim.y;

    if(x >= sizid) return;

    QsA[indices[x]] += values[x];
    QsB[indices[x]] += values[x];

}

#ifdef ISD8
// computes MPM equations
__global__ void compute_MPM(float *hw, float *Z, float *QsA, float *QsB, uint8_t *BC) {

    // Getting the right IF
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = y * nx + x;

    int idnei;
    if(get_indices(idx, idnei, x, y, BC) == false) return;

    float surface_idx = hw[idx] + Z[idx];

    float weights[NNEIGHBOURS];
    for(int i=0;i<NNEIGHBOURS;++i){
        weights[i] = 0.;
    } 
    float sumWeights = 0.;

    float capacity = 0.;
    float SSdy = 1.;
    float SS = -1;


    for(int i=0;i<NNEIGHBOURS;++i){

        // idx of the neighbours
        int adder = neighbourers[idnei][i];
        if(adder == nodata) continue;
        int nidx = idx + adder;
        
        // calculating local weight (i.e. Sw * dy)
        float ts = surface_idx - (hw[nidx] + Z[nidx]);

        if(ts<0) continue; // aborting if higher neighbour
        
        // finishing slope calc
        ts /= dXs[i];

        if(ts > SS){
            SS = ts;
            SSdy = dYs[i];
        }


        float tau = rho_water * gravity * hw[nidx] * ts;
        if(tau <= tau_c) continue;

        float tcapacity = pow(tau - tau_c, 1.5);

        capacity += tcapacity * dYs[i];

        // tcapacity *= generate_random(round(idx * hw[idx]));

        // saving weight
        weights[i] = tcapacity;

        // summing all of em'
        sumWeights += tcapacity;



    }

    if(sumWeights <= 0 || capacity <= 0) return;

    if(rho_water * gravity * SS * hw[idx] <= tau_c) return;

    float correction_factor = pow(rho_water * gravity * SS * hw[idx] - tau_c, 1.5) * SSdy / capacity;
    capacity *= correction_factor * E_MPM;

    QsB[idx] = capacity;

    for(int i=0;i<NNEIGHBOURS;++i){
        if(weights[i] <= 0) continue;
        int adder = neighbourers[idnei][i];
        if(adder == nodata) continue;
        int nidx = idx + adder;
        atomicAdd(&QsA[nidx], capacity * weights[i]/sumWeights);
    }

}
#else
// computes MPM equations
__global__ void compute_MPM(float *hw, float *Z, float *QsA, float *QsB, uint8_t *BC) {

    // Getting the right IF
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = y * nx + x;

    int idnei;
    if(get_indices(idx, idnei, x, y, BC) == false) return;

    float surface_idx = hw[idx] + Z[idx];

    // float weights[NNEIGHBOURS];
    // for(int i=0;i<NNEIGHBOURS;++i){
    //     weights[i] = 0.;
    // } 
    // float sumWeights = 0.;

    // float capacity = 0.;

    for(int i=0;i<NNEIGHBOURS;++i){

        // idx of the neighbours
        int adder = neighbourers[idnei][i];
        if(adder == nodata) continue;
        int nidx = idx + adder;
        
        // calculating local weight (i.e. Sw * dy)
        float ts = surface_idx - (hw[nidx] + Z[nidx]);

        if(ts<0) continue; // aborting if higher neighbour
        
        // finishing slope calc
        ts /= dXs[i];

        float tau = rho_water * gravity * hw[nidx] * ts;
        if(tau <= tau_c) continue;

        float capacity = E_MPM * pow(tau - tau_c, 1.5) * dYs[i];
        atomicAdd(&QsA[nidx], capacity);
        QsB[idx] += capacity;

        // tcapacity *= generate_random(round(idx * hw[idx]));

        // saving weight
        // weights[i] = tcapacity;

        // summing all of em'
        // sumWeights += tcapacity;



    }

    // if(sumWeights <= 0 || capacity <= 0) return;

    // capacity *= E_MPM;

    // QsB[idx] = capacity;

    // for(int i=0;i<NNEIGHBOURS;++i){
    //     if(weights[i] <= 0) continue;
    //     int adder = neighbourers[idnei][i];
    //     if(adder == nodata) continue;
    //     int nidx = idx + adder;
    //     atomicAdd(&QsA[nidx], capacity * weights[i]/sumWeights);
    // }

}
#endif


// Increment sediment height function of the divergence of the fluxes
__global__ void increment_hs(float *hw, float *Z,float *QsA, float *QsB, uint8_t *BC) {

    // Getting the right IF
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = y * nx + x;

    if(idx >= nx*ny || x >=nx || y >=ny) return;
    if(BC[idx] != 1) return;

    float dhs = (QsA[idx] - QsB[idx] * Qsout_mult)/ cellarea * dt_morpho;
    Z[idx] += dhs;
    // hw[idx] += dhs;

    // if(hw[idx] < 0){
        // Z[idx] -= hw[idx];
    //     hw[idx] = 0.;
    // }

}

//###########################################
// Other morphodynamics
//###########################################




__global__ void diffuse(float *Z, float *Z_new, float D, float dx, float dy) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * nx + x;
    if(idx < nx*ny) Z_new[idx] = Z[idx];

    if (x > 4 && x < nx - 1 && y > 0 && y < ny - 1) {
        
        // Compute the Laplacian
        float laplacian = (Z[idx - 1] - 2.0 * Z[idx] + Z[idx + 1]) / (dx * dx) +
                          (Z[idx - nx] - 2.0 * Z[idx] + Z[idx + nx]) / (dy * dy);

        // Update the value using the diffusion equation
        Z_new[idx] += D * laplacian;
    }
}



//###########################################
// Other operations
//###########################################



__global__ void equal(float *A, float *B, int size){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * nx + x;

    if(idx>= size) return;
    
    A[idx] = B[idx];
}

__global__ void add_constant(float *A, uint8_t *BC, float val, int width, int height){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = y * nx + x;
    
    if(x >= width || y >= height) return;

    if(can_out(BC[idx])) return;

    A[idx] += val;

}

__global__ void grid_to(float *A, float val){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = y * nx + x;
    
    if(idx >= nx*ny || x >= nx || y >= ny) return ;    


    A[idx] = val;

}



































// end of file