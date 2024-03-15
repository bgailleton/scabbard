/*


*/




__global__ void add_Qw_global(float* QwA, float val, unsigned char *BC){
	// Getting the right index
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	int idx;
	if(get_index_check( x,  y, idx, BC) == false) return;
	QwA[idx] += val;
}


__global__ void add_Qw_local(int *indices, float *values, float *QwA, float *QwB, const int sizid) {
	// Getting the right IF
	int x = threadIdx.x + blockIdx.x * blockDim.x;

	if(x >= sizid) return;

	QwA[indices[x]] += values[x];
	QwB[indices[x]] += values[x];

}

// transfers Qwin on a node to node basis
__global__ void compute_Qwin(float *hw, float *Z, float *QwA, float *QwB, unsigned char *BC) {

	// Getting the right IF
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int idx,adder;
	if(get_index(x, y, idx, adder, BC) == false) return;
	if(BC::can_out(BC[idx])) return;

	float surface_idx = hw[idx] + Z[idx];

	float weights[NNEIGHBOURS];

	for(int j=0;j<NNEIGHBOURS;++j){
		weights[j] = 0.;
	}

	float sumWeights = 0.;

	while(sumWeights == 0.){

		for(int j=0;j<NNEIGHBOURS;++j){

			// idx of the neighbours
			int nidx;
			if(get_neighbour(idx, adder, j, nidx) == false) continue;
			
			// calculating local weight (i.e. Sw * dy)
			float tw = surface_idx - (hw[nidx] + Z[nidx]);

			if(tw<0) continue; // aborting if higher neighbour
			
			// finishing slope calc
			tw *= DYS[j];
			tw /= DXS[j];

			// saving weight
			weights[j] = tw;

			// summing all of em'
			sumWeights += tw;
		}

		if(sumWeights == 0.) {
			hw[idx] += 0.0001;
			surface_idx += 0.0001;
		}


	}

	for(int j=0;j<NNEIGHBOURS;++j){
		if(weights[j] <= 0) continue;
		int nidx;
		if(get_neighbour(idx, adder, j, nidx) == false) continue;
		
		atomicAdd(&QwB[nidx], QwA[idx] * weights[j]/sumWeights);
	}

}

// intermediate function required tofinalise the new Qwin
__global__ void swapQwin(float *QwA, float *QwB) {

	// Getting the right IF
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int idx;
	if(get_index_raw(x, y, idx) == false) return;
	
	QwA[idx] = QwB[idx];
	QwB[idx] = 0.;

}


// #ifdef ISD8
// #if true
#if true // Forcing stencil approach whatever
// compute Qwout using Gailleton et al 2024
__global__ void compute_Qwout(float *hw, float *Z, float *QwB, unsigned char *BC) {

	// Getting the right IF
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int idx,adder;
	if(get_index(x, y, idx, adder, BC) == false) return;
	if(BC::can_out(BC[idx]) == true){return;}; 

	float surface_idx = hw[idx] + Z[idx];

	float SS = -1;
	float SSdy = 1.;

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
	}

	if(SS == -1) return;

	QwB[idx] = SSdy/MANNING * pow(hw[idx],5./3.) * sqrt(SS);

}
#else
// compute Qwout using Gailleton et al 2024
__global__ void compute_Qwout(float *hw, float *Z, float *QwB, unsigned char *BC) {

	// Getting the right IF
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int idx,adder;
	if(get_index(x, y, idx, adder, BC) == false) return;
	if(BC::can_out(BC[idx]) == true){return;}; 

	float surface_idx = hw[idx] + Z[idx];

	QwB[idx] = 0.;


	for(int j=0;j<NNEIGHBOURS;++j){

		int nidx;
		if(get_neighbour(idx, adder, j, nidx) == false) continue;
		
		// calculating local weight (i.e. Sw * dy)
		float ts = surface_idx - (hw[nidx] + Z[nidx]);

		if(ts<0) continue; // aborting if higher neighbour
		
		// finishing slope calc
		ts /= DXS[j];

		QwB[idx] += DYS[j]/MANNING * pow(hw[idx],5./3.) * sqrt(ts);;
	}

}
#endif


// #ifdef ISD8
#if true // Forcing stencil approach whatever
// compute Qwout using Gailleton et al 2024
__global__ void compute_Qw_dyn(float *hw, float *Z, float *QwA, float *QwB, unsigned char *BC) {

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

	float SS = -1;
	float SSdy = 1.;

	for(int j=0;j<NNEIGHBOURS;++j){

		int nidx;
		if(get_neighbour(idx, adder, j, nidx) == false) continue;
		
		// calculating local weight (i.e. Sw * dy)
		float ts = surface_idx - (hw[nidx] + Z[nidx]);

		if(ts<0) continue; // aborting if higher neighbour
		
		// finishing slope calc
		ts /= DXS[j];

		float tw = ts * DYS[j];

		// saving weight
		weights[j] = tw;

		// summing all of em'
		sumWeights += tw;

		if(ts > SS){
			SS = ts;
			SSdy = DYS[j];
		}
	}

	if(SS == -1 || sumWeights <= 0 ) return;

	QwB[idx] = SSdy/MANNING * pow(hw[idx],5./3.) * sqrt(SS);

	for(int j=0;j<NNEIGHBOURS;++j){
		if(weights[j] <= 0) continue;
		int nidx;
		if(get_neighbour(idx, adder, j, nidx) == false) continue;
		
		atomicAdd(&QwA[nidx], QwB[idx] * weights[j]/sumWeights);
	}

}
#else
// compute Qwout using Gailleton et al 2024
__global__ void compute_Qw_dyn(float *hw, float *Z, float *QwA, float *QwB, unsigned char *BC) {

	// Getting the right IF
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int idx,adder;
	if(get_index(x, y, idx, adder, BC) == false) return;
	if(BC::can_out(BC[idx]) == true){return;}; 

	float surface_idx = hw[idx] + Z[idx];

	QwB[idx] = 0.;


	for(int j=0;j<NNEIGHBOURS;++j){

		int nidx;
		if(get_neighbour(idx, adder, j, nidx) == false) continue;
		
		// calculating local weight (i.e. Sw * dy)
		float ts = surface_idx - (hw[nidx] + Z[nidx]);

		if(ts<0) continue; // aborting if higher neighbour
		
		// finishing slope calc
		ts /= DXS[j];

		QwB[idx] += DYS[j]/MANNING * hw[idx] * sqrt(ts);;
	}

}
#endif


// Increment water function of the divergence of the fluxes
__global__ void increment_hw(float *hw, float *Z,float *QwA, float *QwB, unsigned char *BC) {

	// Getting the right IF
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int idx;
	if(get_index_check(x, y, idx, BC) == false) return;

	if(BC::can_out(BC[idx]) == true){return;}; 
	
	float dhw = (QwA[idx] - QwB[idx])/ CELLAREA * DT_HYDRO;

	// float mult = (BC::can_out(BC[idx])) ? 0.: 1.;

	hw[idx] = max(0., hw[idx] + dhw);// * float(BC::can_out(BC[idx])) ;
	
	// else{
	// 	hw[idx] = 0.;
	// }
	

}

// Increment water function of the divergence of the fluxes
__global__ void HwBC20(float *hw, unsigned char *BC) {

	// Getting the right IF
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int idx;
	if(get_index_check(x, y, idx, BC) == false) return;

	if(BC::can_out(BC[idx])){
		hw[idx] = 0.; 
		return;
	} 

}







// end of file