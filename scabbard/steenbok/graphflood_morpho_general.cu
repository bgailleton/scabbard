/*


*/




__global__ void add_Qs_global(float* QsA, float val, unsigned char *BC){
	// Getting the right index
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	int idx;
	if(get_index_check( x,  y, idx, BC) == false) return;
	QsA[idx] += val;
}


__global__ void add_Qs_local(int *indices, float *values, float *QsA, float *QsB, const int sizid) {
	// Getting the right IF
	int x = threadIdx.x + blockIdx.x * blockDim.x;

	if(x >= sizid) return;

	QsA[indices[x]] += values[x];
	QsB[indices[x]] += values[x];

}



// Increment water function of the divergence of the fluxes
__global__ void increment_hs(float *hw, float *Z,float *QsA, float *QsB, float *QsD, unsigned char *BC) {

	// Getting the right IF
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int idx;
	if(get_index_check(x, y, idx, BC) == false) return;

	if(BC::can_out(BC[idx]) == true){return;}; 
	if(BC::can_receive(BC[idx]) == false){return;}; 
	
    // if(QsA[idx]>0) printf("A%f \n", QsA[idx]);
    // if(QsB[idx]>0) printf("B%f \n", QsB[idx]);

	double dhs = (double(QsA[idx]) - double(QsB[idx]) - double(QsD[idx]))/ CELLAREA * DT_MORPHO;
	// if(dhs > 0) printf("dhs = %f", dhs);
    // if(QsA[idx]>0) printf(" %f / %f  => %f\n", QsA[idx] -  QsB[idx], CELLAREA, dhs);
	// dhs = min(dhs,1e-2);
	
	Z[idx] += float(dhs);


}

// Increment water function of the divergence of the fluxes
__global__ void increment_hs_noQD(float *hw, float *Z,float *QsA, float *QsB, unsigned char *BC) {

	// Getting the right IF
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int idx;
	if(get_index_check(x, y, idx, BC) == false) return;

	if(BC::can_out(BC[idx]) == true){return;}; 
	
    // if(QsA[idx]>0) printf("A%f \n", QsA[idx]);
    // if(QsB[idx]>0) printf("B%f \n", QsB[idx]);

	double dhs = (double(QsA[idx]) - double(QsB[idx]))/ CELLAREA * DT_MORPHO;
	// if(dhs > 0) printf("dhs = %f", dhs);
    // if(QsA[idx]>0) printf(" %f / %f  => %f\n", QsA[idx] -  QsB[idx], CELLAREA, dhs);
	// dhs = min(dhs,1e-2);
	
	Z[idx] += float(dhs);


}

// intermediate function required tofinalise the new Qwin
__global__ void swapQsin(float *QsA, float *QsB) {

	// Getting the right IF
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int idx;
	if(get_index_raw(x, y, idx) == false) return;
	
	QsA[idx] = QsB[idx];
	QsB[idx] = 0.;

}


__global__ void bedslip(float *hw, float *Z, float* QsA, float* QsD, unsigned char *BC) {

	// Getting the right IF
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int idx,adder;
	if(get_index(x, y, idx, adder, BC) == false) return;
	if(BC::can_out(BC[idx]) == true){return;}; 

	if(hw[idx] < BS_MINHW) return;

	



	float SS = -1;
	float SSdy = 1.;

	for(int j=0;j<NNEIGHBOURS;++j){

		int nidx;
		if(get_neighbour(idx, adder, j, nidx) == false) continue;
		
		// calculating local weight (i.e. Sw * dy)
		float ts = Z[idx] - Z[nidx];

		if(ts<0) continue; // aborting if higher neighbour
		
		// finishing slope calc
		ts /= DXS[j];

		float terr = BS_K * pow(ts,BS_EXP);
		if(BC::can_out(BC[nidx])){
			continue;
		}

		QsD[idx] += terr * CELLAREA;
		atomicAdd(&QsA[nidx], terr * CELLAREA);		
	}

	if(SS == -1) return;

	// QwB[idx] = SSdy/MANNING * pow(hw[idx],5./3.) * sqrt(SS);

}




// end of file