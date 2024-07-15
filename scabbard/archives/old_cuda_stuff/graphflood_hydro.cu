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
	// QwB[indices[x]] += values[x];

}


// transfers Qwin on a node to node basis
__global__ void compute_Qws(float *hw, float *Z, float *QwA, float *QwB, float *QwC, unsigned char *BC) {

	// Getting the right IF
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int idx,adder;
	if(get_index(x, y, idx, adder, BC) == false) return;
	if(BC::can_out(BC[idx])) return;

	QwC[idx] = 0.;

	float surface_idx = hw[idx] + Z[idx];

	float weights[NNEIGHBOURS];

	for(int j=0;j<NNEIGHBOURS;++j){
		weights[j] = 0.;
	}

	float sumWeights = 0., SSx = 0., SSy = 0.;

	bool outs = false;

	while(sumWeights == 0.){

		for(int j=0;j<NNEIGHBOURS;++j){
			// if((j == 1 || j == 2)) continue;

			// idx of the neighbours
			int nidx;
			if(get_neighbour(idx, adder, j, nidx, BC) == false) continue;

			
			// calculating local weight (i.e. Sw * dy)
			float tw = surface_idx - (hw[nidx] + Z[nidx]);

			if(tw<0) continue; // aborting if higher neighbour
			
			// finishing slope calc
			tw /= DXS[j];

			// if(BC::can_out(BC[nidx])) tw = BOUND_SLOPE;

			if(BC::can_out(BC[nidx])) outs = true;


			// saving weight
			weights[j] = tw;

			// summing all of em'
			sumWeights += tw;

			if(j == 0 || j == 3){
				if(tw > SSy) SSy = tw;
			} else{
				if(tw > SSx) SSx = tw;
			}
		}

		if(sumWeights == 0.) {
			hw[idx] += 0.0001;
			surface_idx += 0.0001;
		}


	}

	float gradSw = sqrt(SSx * SSx + SSy * SSy);
	if(outs) gradSw = BOUND_SLOPE;


	for(int j=0;j<NNEIGHBOURS;++j){
		if(weights[j] <= 0) continue;
		int nidx;
		if(get_neighbour(idx, adder, j, nidx, BC) == false) continue;
		
		atomicAdd(&QwB[nidx], QwA[idx] * weights[j]/sumWeights);
	}

	QwC[idx] = DX/MANNING * pow(hw[idx], 5./3.) * sumWeights/sqrt(gradSw);

}

// transfers Qwin on a node to node basis
__global__ void compute_QwsQss(float *hw, float *Z, float *QwA, float *QwB, float *QwC, float* QsA, float* QsB, float* QsC, float* QsD, unsigned char *BC) {

	// Getting the right IF
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int idx,adder;
	if(get_index(x, y, idx, adder, BC) == false) return;
	if(BC::can_out(BC[idx])) return;

	QwC[idx] = 0.;
	// QsC[idx] = 0.;

	float surface_idx = hw[idx] + Z[idx];

	float weights[NNEIGHBOURS];
	float weightsSed[NNEIGHBOURS];

	for(int j=0;j<NNEIGHBOURS;++j){
		weights[j] = 0.;
		weightsSed[j] = 0.;
	}

	float sumWeights = 0., SSx = 0., SSy = 0.;
	float sumWeightsSed = 0., SSxZ = 0., SSyZ = 0.;
	float SSxPart = 0., SSyPart = 0.;

	float partsurf = (KZ * Z[idx] + KH * hw[idx] * (Z[idx] + hw[idx]))/(KZ + hw[idx] * KH);

	bool outs = false;

	while(sumWeights == 0.){
		sumWeightsSed = 0.;
		for(int j=0;j<NNEIGHBOURS;++j){

			// idx of the neighbours
			int nidx;
			if(get_neighbour(idx, adder, j, nidx, BC) == false) continue;
			
			// calculating local weight (i.e. Sw * dy)
			float tw = surface_idx - (hw[nidx] + Z[nidx]);
			tw /= DX;
			float S_part = (partsurf - ((KZ * Z[nidx] + KH * hw[nidx] * (Z[nidx] + hw[nidx]))/(KZ + hw[nidx] * KH)))/DX;
			// S_part = S_part * sqrt(S_part );
			float SZ = (Z[idx] - Z[nidx])/DX;
			
			// if(BC::can_out(BC[nidx])) {tw = BOUND_SLOPE; S_part = KZ * BOUND_SLOPE; SZ = BOUND_SLOPE;}

			if(BC::can_out(BC[nidx])) outs = true;

			if(j == 0 || j == 3){
				if(SZ > SSyZ) SSyZ = SZ;
			}else{
				if(SZ > SSxZ) SSxZ = SZ;
			}

			if(j == 0 || j == 3){
				if(S_part > SSyPart) SSyPart = S_part;
			}else{
				if(S_part > SSxPart) SSxPart = S_part;
			}

			weightsSed[j] = max(S_part,0.);
			sumWeightsSed += max(S_part,0.);
			
			// saving weight
			weights[j] = max(0.,tw);

			// summing all of em'
			sumWeights += max(0.,tw);

			if(j == 0 || j == 3){
				if(tw > SSy) SSy = tw;
			} else{
				if(tw > SSx) SSx = tw;
			}
		}

		if(sumWeights == 0.) {
			hw[idx] += 0.0001;
			surface_idx += 0.0001;
		}


	}

	float gradSw = sqrt(SSx * SSx + SSy * SSy);
	float gradSZ = sqrt(SSxZ * SSxZ + SSyZ * SSyZ);
	float gradSPart = sqrt(SSxPart * SSxPart + SSyPart * SSyPart);
	if(outs) {
		gradSw = BOUND_SLOPE;
		gradSZ = BOUND_SLOPE;
		gradSPart = KZ * BOUND_SLOPE;
	}


	// NORMAL
	// float edot = max(K_EROS * pow( hw[idx] * GRAVITY * RHO_WATER * gradSw - TAU_C,1.5), 0.);
	// GUILLAUME
	float edot = max(K_EROS * pow( GRAVITY * (KH * RHO_WATER * hw[idx] * gradSw + (RHO_SEDIMENT - RHO_WATER) * KZ * gradSZ) - TAU_C,1.5), 0.);
	// if(edot>0) printf("%f\n", edot);
	// float edot = max(K_EROS * pow( (KH * hw[idx] * gradSw + KZ * gradSZ) - TAU_C/(GRAVITY * RHO_WATER),1.5), 0.);
	// float edot = max(K_EROS * pow( GRAVITY * RHO_WATER * gradSPart - TAU_C,1.5), 0.);
	// if(hw[idx] <1e-2) edot = 0;
	// float edot = max(K_EROS * pow( hw[idx] * GRAVITY * RHO_WATER * gradSw - TAU_C,1.5), 0.) / (1 + exp(10 * (hw[idx] - 0.5)));

	// float locerr = 0., laterr = 0.;
	// float locQ = 0.;
	// QsC[idx] = (QsA[idx] + pow(DX,2) * (edot))/(1 + pow(DX,2) * (gradSw/(sumWeights * L_EROS)));
	float depcof = pow(DX,2) * (gradSPart/(sumWeightsSed * L_EROS));
	if(sumWeightsSed == 0. || gradSPart == 0.) depcof = 0.;

	// depcof = 0.;
	
	QsC[idx] = (QsA[idx] + pow(DX,2) * (edot))/(1 + depcof);
	// if(isnan(QsC[idx])) QsC[idx] = QsA[idx];
	if(outs) QsC[idx] = QsA[idx];

	// if(sumWeightsSed == 0.) sumWeightsSed = 1;


	// if(KL_EROS > 0){
	// 	for(int j=0;j<NNEIGHBOURS;++j){
	// 		int nidx;
	// 		if(get_neighbour(idx, adder, j, nidx) == false) continue;
			
	// 		// if(Z[nidx] < Z[idx]){
	// 		// 	float tlocQ = - KL_EROS * QsC[idx];
	// 		// 	atomicAdd(&QsD[nidx], tlocQ);
	// 		// 	locQ += tlocQ;
	// 		// 	// QsC[idx] -=
	// 		// }


	// 		// KEEP DAT: lateral diffusion of the bed, but expressed on a cell basis
	// 		if(Z[nidx] > Z[idx]){
	// 			// float tlocerr = KL_EROS * (Z[idx] - Z[nidx])/DX * edot ;
	// 			float tlocerr = KL_EROS * min(Z[nidx] - Z[idx], hw[idx])/DX * edot ;
	// 			atomicAdd(&QsD[nidx], tlocerr * CELLAREA);
	// 			laterr += tlocerr; 
	// 		}


	// 	}
	// 	QsA[idx] += laterr * CELLAREA;
	// }
	// QsC[idx] = (QsA[idx] + pow(DX,2) * (edot))/(1 + pow(DX,2) * (gradSw/(sumWeights * L_EROS)));

	// float sumsum = 0;
	for(int j=0;j<NNEIGHBOURS;++j){
		int nidx;
		if(get_neighbour(idx, adder, j, nidx, BC) == false) continue;
		// sumsum += weightsSed[j]/sumWeightsSed;
		atomicAdd(&QwB[nidx], QwA[idx] * weights[j]/sumWeights);
		if(sumWeightsSed > 0)
			atomicAdd(&QsB[nidx], (QsC[idx]) * weightsSed[j]/sumWeightsSed);
	}
	// if(sumsum >0.1 && sumsum < 0.9) printf("huh %f\n",sumsum );

	QwC[idx] = DX/MANNING * pow(hw[idx], 5./3.) * sumWeights/sqrt(gradSw);
	// KEEP DAT: lateral diffusion of the bed, but expressed on a cell basis
	// QsC[idx] += locerr * CELLAREA;
	// QsC[idx] += laterr * CELLAREA;
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
	// if(isnan(Z[idx]))
		// printf("NANBEEF\n");
	Z[idx] += float(dhs);
	// if(isnan(Z[idx]))
	// 	printf("NANAFFF, %f, %f at %i,%i\n",  QsA[idx],  QsB[idx],x,y);


}

// transfers Qwin on a node to node basis
__global__ void ARCHIVE_compute_QwsQss(float *hw, float *Z, float *QwA, float *QwB, float *QwC, float* QsA, float* QsB, float* QsC, float* QsD, unsigned char *BC) {

	// Getting the right IF
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int idx,adder;
	if(get_index(x, y, idx, adder, BC) == false) return;
	if(BC::can_out(BC[idx])) return;

	QwC[idx] = 0.;
	// QsC[idx] = 0.;

	float surface_idx = hw[idx] + Z[idx];

	float weights[NNEIGHBOURS];

	for(int j=0;j<NNEIGHBOURS;++j){
		weights[j] = 0.;
	}

	float sumWeights = 0., SSx = 0., SSy = 0.;
	float SSxZ = 0., SSyZ = 0.;

	while(sumWeights == 0.){

		for(int j=0;j<NNEIGHBOURS;++j){

			// idx of the neighbours
			int nidx;
			if(get_neighbour(idx, adder, j, nidx, BC) == false) continue;
			
			// calculating local weight (i.e. Sw * dy)
			float tw = surface_idx - (hw[nidx] + Z[nidx]);

			float SZ = (Z[idx] - Z[nidx])/DX;
			if(j == 0 || j == 3){
				if(SZ > SSyZ) SSyZ = SZ;
			}else{
				if(SZ > SSxZ) SSxZ = SZ;
			}
			
			if(tw<0) continue; // aborting if higher neighbour
			
			// finishing slope calc
			tw /= DXS[j];

			// saving weight
			weights[j] = tw;

			// summing all of em'
			sumWeights += tw;

			if(j == 0 || j == 3){
				if(tw > SSy) SSy = tw;
			} else{
				if(tw > SSx) SSx = tw;
			}
		}

		if(sumWeights == 0.) {
			hw[idx] += 0.0001;
			surface_idx += 0.0001;
		}


	}

	float gradSw = sqrt(SSx * SSx + SSy * SSy);
	float gradSZ = sqrt(SSxZ * SSxZ + SSyZ * SSyZ);

	// NORMAL
	// float edot = max(K_EROS * pow( hw[idx] * GRAVITY * RHO_WATER * gradSw - TAU_C,1.5), 0.);
	// GUILLAUME
	float edot = max(K_EROS * pow( hw[idx] * GRAVITY * RHO_WATER * gradSw + 0.7/TAU_C * gradSZ - TAU_C,1.5), 0.);
	// if(hw[idx] <1e-2) edot = 0;
	// float edot = max(K_EROS * pow( hw[idx] * GRAVITY * RHO_WATER * gradSw - TAU_C,1.5), 0.) / (1 + exp(10 * (hw[idx] - 0.5)));

	float locerr = 0., laterr = 0.;
	float locQ = 0.;
	// QsC[idx] = (QsA[idx] + pow(DX,2) * (edot))/(1 + pow(DX,2) * (gradSw/(sumWeights * L_EROS)));
	QsC[idx] = (QsA[idx] + pow(DX,2) * (edot))/(1 + pow(DX,2) * (gradSw/(sumWeights * L_EROS))) ;


	// if(KL_EROS > 0){
	// 	for(int j=0;j<NNEIGHBOURS;++j){
	// 		int nidx;
	// 		if(get_neighbour(idx, adder, j, nidx) == false) continue;
			
	// 		// if(Z[nidx] < Z[idx]){
	// 		// 	float tlocQ = - KL_EROS * QsC[idx];
	// 		// 	atomicAdd(&QsD[nidx], tlocQ);
	// 		// 	locQ += tlocQ;
	// 		// 	// QsC[idx] -=
	// 		// }


	// 		// KEEP DAT: lateral diffusion of the bed, but expressed on a cell basis
	// 		if(Z[nidx] > Z[idx]){
	// 			// float tlocerr = KL_EROS * (Z[idx] - Z[nidx])/DX * edot ;
	// 			float tlocerr = KL_EROS * min(Z[nidx] - Z[idx], hw[idx])/DX * edot ;
	// 			atomicAdd(&QsD[nidx], tlocerr * CELLAREA);
	// 			laterr += tlocerr; 
	// 		}


	// 	}
	// 	QsA[idx] += laterr * CELLAREA;
	// }
	// QsC[idx] = (QsA[idx] + pow(DX,2) * (edot))/(1 + pow(DX,2) * (gradSw/(sumWeights * L_EROS)));


	for(int j=0;j<NNEIGHBOURS;++j){
		if(weights[j] <= 0) continue;
		int nidx;
		if(get_neighbour(idx, adder, j, nidx, BC) == false) continue;
		
		atomicAdd(&QwB[nidx], QwA[idx] * weights[j]/sumWeights);
		atomicAdd(&QsB[nidx], (QsC[idx]) * weights[j]/sumWeights);
	}

	QwC[idx] = DX/MANNING * pow(hw[idx], 5./3.) * sumWeights/sqrt(gradSw);
	// KEEP DAT: lateral diffusion of the bed, but expressed on a cell basis
	// QsC[idx] += locerr * CELLAREA;
	// QsC[idx] += laterr * CELLAREA;
}

// transfers Qwin on a node to node basis
__global__ void compute_QwsQss_dyn(float *hw, float *Z, float *QwA, float *QwB, float *QwC, float* QsA, float* QsB, float* QsC, unsigned char *BC) {

	// Getting the right IF
	return;
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
			// if((j == 1 || j == 2)) continue;

			// idx of the neighbours
			int nidx;
			if(get_neighbour(idx, adder, j, nidx, BC) == false) continue;
			
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
		if(get_neighbour(idx, adder, j, nidx, BC) == false) continue;
		
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
		if(get_neighbour(idx, adder, j, nidx, BC) == false) continue;
		
		// calculating local weight (i.e. Sw * dy)
		float ts = surface_idx - (hw[nidx] + Z[nidx]);

		if(ts<0) continue; // aborting if higher neighbour
		
		// finishing slope calc
		ts /= DXS[j];

		if(BC::can_out(BC[nidx])){
			ts = BOUND_SLOPE;
			SS = ts;
			SSdy = DYS[j];
			break;
		}
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
		if(get_neighbour(idx, adder, j, nidx, BC) == false) continue;
		
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
		if(get_neighbour(idx, adder, j, nidx, BC) == false) continue;
		
		// calculating local weight (i.e. Sw * dy)
		float ts = surface_idx - (hw[nidx] + Z[nidx]);

		if(ts<0) continue; // aborting if higher neighbour
		
		// finishing slope calc
		ts /= DXS[j];

		if(BC::can_out(BC[nidx])) ts = BOUND_SLOPE;


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
		if(get_neighbour(idx, adder, j, nidx, BC) == false) continue;
		
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
		if(get_neighbour(idx, adder, j, nidx, BC) == false) continue;
		
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