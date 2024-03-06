// kernels.cu

__constant__ float dXs[8];
__constant__ float dYs[8];
__constant__ int neighbourers[9][8];
__constant__ float dt;
__constant__ float cellarea;
__constant__ float manning;
__constant__ int nx;
__constant__ int ny;
__constant__ int nodata;


// __global__ void sum_positive_gradients(float *img, float *output, const int width, const int height) {
//     int x = threadIdx.x + blockIdx.x * blockDim.x;
//     int y = threadIdx.y + blockIdx.y * blockDim.y;

//     if (x >= width || y >= height) return;

//     int idx = y * width + x;
//     float sum = 0.0;
//     float center_value = img[idx];

//     // Check the 8 neighbors
//     for (int dy = -1; dy <= 1; dy++) {
//         for (int dx = -1; dx <= 1; dx++) {
//             if (dx == 0 && dy == 0) continue; // Skip the center pixel itself

//             int nx = x + dx;
//             int ny = y + dy;

//             // Check boundaries
//             if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
//                 int neighbor_idx = ny * width + nx;
//                 float gradient = img[neighbor_idx] - center_value;
//                 if (gradient > 0) {
//                     sum += gradient;
//                 }
//             }
//         }
//     }

//     output[idx] = sum;
// }


__device__ bool get_indices(int& idx, int& idx_neighbourer, int col, int row, unsigned char *BC){

    if(idx >= nx*ny) return false;
    unsigned char tbc = BC[idx];

    if(tbc == 1){
        idx_neighbourer = 0;
        return true;
    }else if(tbc == 0)
        return false;
    else if(col == 0 && row == 0){
        idx_neighbourer = 1;
        return true;
    }else if(idx < nx-1 && row == 0){
        idx_neighbourer = 2;
        return true;
    }else if(row == 0){
        idx_neighbourer = 3;
        return true;
    }else if(col == 0 && row != ny-1){
        idx_neighbourer = 4;
        return true;
    }else if(col == ny-1 && row != ny-1){
        idx_neighbourer = 5;
        return true;
    }else if(col == 0 && row == ny-1){
        idx_neighbourer = 6;
        return true;
    }else if(col != nx-1 && row == ny-1){
        idx_neighbourer = 7;
        return true;
    }else if(col == nx-1 && row == ny-1){
        idx_neighbourer = 8;
        return true;
    }


}




__global__ void add_Qwin_local(int *indices, float *values, float *QwA, float *QwB, const int sizid) {
    // Getting the right IF
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = y * nx + x;
    if(idx >= nx * ny) return;

    QwB[idx] = 0;

    if(idx >= sizid) return;

    QwA[indices[idx]] += values[idx];
    QwB[indices[idx]] += values[idx];

}


__global__ void compute_Qwin(float *hw, float *Z, float *QwA, float *QwB, unsigned char *BC) {

    // Getting the right IF
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = y * nx + x;

    int idnei;
    if(get_indices(idx, idnei, x, y, BC) == false) return;

    float surface_idx = hw[idx] + Z[idx];

    float weights[8] = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f}; 
    float sumWeights = 0.;

    for(int i=0;i<8;++i){

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

    if(sumWeights <= 0) return;

    for(int i=0;i<8;++i){
        if(weights[i] <= 0) continue;
        int adder = neighbourers[idnei][i];
        if(adder == nodata) continue;
        int nidx = idx + adder;
        atomicAdd(&QwB[nidx], QwA[idx] * weights[i]/sumWeights);
    }

}

__global__ void swapQwin(float *QwA, float *QwB) {

    // Getting the right IF
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = y * nx + x;

    if(idx >= nx * ny) return;
    
    QwA[idx] = QwB[idx];
    QwB[idx] = 0.;

}

__global__ void compute_Qwout(float *hw, float *Z, float *QwB, unsigned char *BC) {

    // Getting the right IF
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = y * nx + x;

    int idnei;
    if(get_indices(idx, idnei, x, y, BC) == false) return;

    float surface_idx = hw[idx] + Z[idx];

    float SS = -1;
    float SSdy = 1.;

    for(int i=0;i<8;++i){

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

__global__ void increment_hw(float *hw, float *Z,float *QwA, float *QwB, unsigned char *BC) {

    // Getting the right IF
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = y * nx + x;

    if(idx >= nx*ny) return;
    if(BC[idx] != 1) return;

    float dhw = (QwA[idx] - QwB[idx])/ cellarea * dt;
    hw[idx] = max(0., hw[idx] + dhw);

    

}