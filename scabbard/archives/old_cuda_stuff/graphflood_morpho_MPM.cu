/*


*/





#ifdef ISD8
// #if True
// computes MPM equations
__global__ void compute_MPM(float *hw, float *Z, float *QsA, float *QsB, uint8_t *BC) {

    // Getting the right IF
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx,adder;
	if(get_index(x, y, idx, adder, BC) == false) return;
	if(BC::can_out(BC[idx]) == true){return;};

    float surface_idx = hw[idx] + Z[idx];

    float weights[NNEIGHBOURS];
    for(int j=0;j<NNEIGHBOURS;++j){
        weights[j] = 0.;
    } 
    float sumWeights = 0.;

    float capacity = 0.;
    float SSdy = 1.;
    float SS = -1;


    for(int j=0;j<NNEIGHBOURS;++j){

		int nidx;
		if(get_neighbour(idx, adder, j, nidx) == false) continue;
		
        
        // calculating local weight (i.e. Sw * dy)
        float ts = surface_idx - (hw[nidx] + Z[nidx]);

        if(ts<0) continue; // aborting if higher neighbour
        
        // finishing slope calc
        ts /= DXS[j];

        if(ts > SS){
            SS = ts;
            SSdy = DYS[j];
        }


        float tau = RHO_WATER * GRAVITY * hw[nidx] * ts;
        if(tau <= TAU_C) continue;

        float tcapacity = pow(tau - TAU_C, 1.5);

        capacity += tcapacity * DYS[j];

        // tcapacity *= generate_random(round(idx * hw[idx]));

        // saving weight
        weights[j] = tcapacity;

        // summing all of em'
        sumWeights += tcapacity;



    }

    if(sumWeights <= 0 || capacity <= 0) return;

    if(RHO_WATER * GRAVITY * SS * hw[idx] <= TAU_C) return;

    float correction_factor = pow(RHO_WATER * GRAVITY * SS * hw[idx] - TAU_C, 1.5) * SSdy / capacity;
    capacity *= correction_factor * E_MPM;

    QsB[idx] = capacity;

    for(int j=0;j<NNEIGHBOURS;++j){
        if(weights[j] <= 0) continue;
		int nidx;
		if(get_neighbour(idx, adder, j, nidx) == false) continue;

		atomicAdd(&QsA[nidx], capacity * weights[j]/sumWeights);
    }

}
#else
// computes MPM equations
__global__ void compute_MPM(float *hw, float *Z, float *QsA, float *QsB, uint8_t *BC) {

    // Getting the right IF
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx,adder;
	if(get_index(x, y, idx, adder, BC) == false) return;
	if(BC::can_out(BC[idx]) == true){return;};

    float surface_idx = hw[idx] + Z[idx];
    for(int j=0;j<NNEIGHBOURS;++j){

		int nidx;
		if(get_neighbour(idx, adder, j, nidx) == false) continue;
        
        // calculating local weight (i.e. Sw * dy)
        float ts = surface_idx - (hw[nidx] + Z[nidx]);

        if(ts<0) continue; // aborting if higher neighbour
        
        // finishing slope calc
        ts /= DXS[j];

        float tau = RHO_WATER * GRAVITY * hw[idx] * ts;
        if(tau <= TAU_C) continue;
        // printf("%f\n",tau );

        float capacity = E_MPM * pow(tau - TAU_C, 1.5) * DYS[j];
        // printf("%f\n", capacity );

        atomicAdd(&QsA[nidx], capacity);

        QsB[idx] += capacity;
        // QsA[idx] += capacity;
        // printf("%f\n", QsB[idx] );

        // printf("%d , %d", QsA[idx],  QsB[idx]);

    }

}
#endif





__global__ void compute_MPM_SS(float *hw, float *Z, float *QsA, float *QsB, uint8_t *BC) {

    // Getting the right IF
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx,adder;
	if(get_index(x, y, idx, adder, BC) == false) return;
	if(BC::can_out(BC[idx]) == true){return;};

    float surface_idx = hw[idx] + Z[idx];

    float SSdy = 1.;
    float SS = -1;
    int SSnidx = -1;

    for(int j=0;j<NNEIGHBOURS;++j){

		int nidx;
		if(get_neighbour(idx, adder, j, nidx) == false) continue;
		
        
        // calculating local weight (i.e. Sw * dy)
        float ts = surface_idx - (hw[nidx] + Z[nidx]);

        if(ts<0) continue; // aborting if higher neighbour
        
        // finishing slope calc
        ts /= DXS[j];

        if(ts > SS){
            SS = ts;
            SSdy = DYS[j];
        	SSnidx = nidx;
        }

    }

    if(SSnidx == -1) return;

    if(RHO_WATER * GRAVITY * SS * hw[idx] <= TAU_C) return;

    float capacity = E_MPM * pow(RHO_WATER * GRAVITY * SS * hw[idx] - TAU_C, 1.5) * SSdy;

    QsB[idx] = capacity;
	atomicAdd(&QsA[SSnidx], capacity );

}



























































































// end of file