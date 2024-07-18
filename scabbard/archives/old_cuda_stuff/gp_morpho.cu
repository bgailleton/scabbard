

// transfers Qwin on a node to node basis
__global__ void compute_Qsin_v1(float *hw, float *Z, float *QsA, float *QsB, float* QsC, unsigned char *BC) {

	// Getting the right Index  and checking boundary conditions
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int idx,adder;
	if(get_index(x, y, idx, adder, BC) == false) return;
	if(BC::can_out(BC[idx])) return;

	// Local hydraulic surface elevation Zw
	float Zw = hw[idx] + Z[idx];

	// local array of weight partioning
	float weights[NNEIGHBOURS];
	for(int j=0;j<NNEIGHBOURS;++j) weights[j] = 0.;

	// Summing the Zw gradients	
	float sum_grad_Zw = 0.;
	float SSx = 0.;	
	float SSy = 0.;
	float thw = 0.;
	int tnrec = 0;
	float elat = 0;
	
	// neighbour loop
	for(int j=0; j<NNEIGHBOURS; ++j){

		// if((j == 1 || j == 2)) continue;
		// if((j == 3 || j == 0)) continue;

		// idx of the neighbours
		int nidx;
		if(get_neighbour(idx, adder, j, nidx) == false) continue;
		if(BC::can_receive(BC[nidx]) == false) continue;
		
		// calculating local weight (i.e. grad Zw)
		float tw = Zw - (hw[nidx] + Z[nidx]);
		
		// finishing local gradient
		tw /= DXS[j];

		if(j == 0 || j == 3){
			if(tw > SSy) SSy = tw;
		} else{
			if(tw > SSx) SSx = tw;
		}

		if(tw <= 0) continue;

		// regulating local flow depth
		float tthw = hw[idx];
		
		if(Z[nidx] > Z[idx]) tthw -= Z[nidx] - Z[idx];
		thw += tthw;
		++tnrec;

		// Sw is already tw
		tw *= tthw * GRAVITY * RHO_WATER;  
		tw -= TAU_C;


		if(tw <= 0){
			
			continue;
		}

		// ++tnrec;

		// tw = pow(tw,1.5) * (j == 0 || j == 3) ? 1. : sqrt(2.);
		tw = pow(tw,1.5);

		// saving weight
		weights[j] = tw;

		// summing the weight
		sum_grad_Zw += tw;

		

		

		// SSx = max(SSx,tw);

		// next
	}

	// mean hflow for tau calculation
	thw /= tnrec;

	// float tau = RHO_WATER * hw[idx] * sqrt(SSx * SSx + SSy * SSy) * GRAVITY;
	float tau = RHO_WATER *  hw[idx] * sqrt(SSx * SSx + SSy * SSy) * GRAVITY;
	float edot = max(0., tau - TAU_C);
	edot = K_EROS * pow(edot,1.5);
	// foatt ddot
	
	// QsC[idx] = max((QsA[idx]/(1 + DX * DY/L_EROS) + edot * DX * DY),0.);

	// double K = (1. / L_EROS);
	// double edotpsy = (edot) / K;
	// double C1 = (QsA[idx] + elat * CELLAREA) / DX - edotpsy;
	// QsC[idx] = DX * (edotpsy + C1 * std::exp(-DX * K));
	QsC[idx] = DX * (edot * L_EROS + (QsA[idx]/DX - (edot * L_EROS)) * exp(-DX/L_EROS));


	float sum = 0.;
	// Transfer of Qsin
	for(int j=0;j<NNEIGHBOURS;++j){
		// Checking weight and indices
		if(weights[j] <= 0) continue;
		int nidx;
		if(get_neighbour(idx, adder, j, nidx) == false) continue;
		// sum += weights[j]/sum_grad_Zw;
		
		// actual transfer			
		atomicAdd(&QsB[nidx], QsC[idx] * weights[j]/sum_grad_Zw);

	}

	// if(sum > 0.1 && sum<0.9) printf("%f\n", sum);


}



__global__ void increment_morpho(float *hw, float *Z, float *QsA, float* QsC, unsigned char *BC){
	// Getting the right Index  and checking boundary conditions
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int idx,adder;
	if(get_index(x, y, idx, adder, BC) == false) return;
	if(BC::can_out(BC[idx])) return;

	Z[idx] += (QsA[idx] - QsC[idx])/ CELLAREA * DT_MORPHO;

}



__global__ void diffuse_bed(float *hw, float *Z, float *Z_new, float *QsA, unsigned char *BC) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int idx,adder;
	if(get_index(x, y, idx, adder, BC) == false) return;
	if(BC::can_out(BC[idx])) return;
	if(BC::can_receive(BC[idx]) == false) return;

	// Neighbors' indices
	int upIdx; if(top(idx, adder, upIdx) == false){upIdx = idx;};
	int downIdx; if(bottom(idx, adder, downIdx) == false){downIdx = idx;};
	int leftIdx; if(left(idx, adder, leftIdx) == false){leftIdx = idx;};
	int rightIdx; if(right(idx, adder, rightIdx) == false){rightIdx = idx;};

	
	// float meanhw = (hw[upIdx] + hw[downIdx] + hw[leftIdx] + hw[rightIdx] + hw[idx])/5.;
	float thw = max( max( max( max(hw[upIdx], hw[idx]), hw[leftIdx]), hw[rightIdx]), hw[downIdx]);
	float laplacian =  (Z[upIdx] + Z[downIdx] + Z[leftIdx] + Z[rightIdx] - 4 * Z[idx]) / (DX * DX);

	// if(laplacian > 0) return;

	// Apply the diffusion equation
	laplacian *= thw * BS_K * DT_MORPHO;
	

	Z_new[idx] = Z[idx] + laplacian;

	return;
	int maxid = idx;
	float maxh = hw[idx];
	if(hw[upIdx] > maxh){
		maxh = hw[upIdx];
		maxid = upIdx;
	}
	if(hw[downIdx] > maxh){
		maxh = hw[downIdx];
		maxid = downIdx;
	}
	if(hw[leftIdx] > maxh){
		maxh = hw[leftIdx];
		maxid = leftIdx;
	}
	if(hw[rightIdx] > maxh){
		maxh = hw[rightIdx];
		maxid = rightIdx;
	}

	atomicAdd(&QsA[maxid], abs(laplacian) * CELLAREA);

	
}



__global__ void diffuse_bed_v3(float *hw, float *Z, float *Z_new, float *QsA, unsigned char *BC) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int idx,adder;
	if(get_index(x, y, idx, adder, BC) == false) return;
	if(BC::can_out(BC[idx])) return;
	if(BC::can_receive(BC[idx]) == false) return;

	// Neighbors' indices
	// float thhw = 1e-3;
	// int upIdx; if(top(idx, adder, upIdx) == false){upIdx = idx;} if(hw[upIdx] < thhw && hw[idx] < thhw) upIdx = idx;
	// int downIdx; if(bottom(idx, adder, downIdx) == false){downIdx = idx;} if(hw[downIdx] < thhw && hw[idx] < thhw) downIdx = idx;
	// int leftIdx; if(left(idx, adder, leftIdx) == false){leftIdx = idx;} if(hw[leftIdx] < thhw && hw[idx] < thhw) leftIdx = idx;
	// int rightIdx; if(right(idx, adder, rightIdx) == false){rightIdx = idx;} if(hw[rightIdx] < thhw && hw[idx] < thhw) rightIdx = idx;


	// Neighbors' indices
	int upIdx; if(top(idx, adder, upIdx) == false){upIdx = idx;};
	int downIdx; if(bottom(idx, adder, downIdx) == false){downIdx = idx;};
	int leftIdx; if(left(idx, adder, leftIdx) == false){leftIdx = idx;};
	int rightIdx; if(right(idx, adder, rightIdx) == false){rightIdx = idx;};

	// Calculate new value for Z based on diffusion equation
	// float diffusion = BS_K * pow((Z[upIdx] + Z[downIdx] + Z[leftIdx] + Z[rightIdx] - 4 * Z[idx])/(DX*DX), BS_EXP) * pow(1 + hw[idx], BS_HW_EXP);
 // Calculate average value of neighbors
	float sumhw = (hw[upIdx] + hw[downIdx] + hw[leftIdx] + hw[rightIdx] + hw[idx]);
	if(sumhw == 0) return;
	float avg_neighbors =  (hw[upIdx]*Z[upIdx] + hw[downIdx]*Z[downIdx] + hw[leftIdx]*Z[leftIdx] + hw[rightIdx]*Z[rightIdx] + hw[idx]*Z[idx]) / sumhw;

	float dZ = (avg_neighbors - Z[idx]);
	if(dZ > 0) return;

	// Apply the diffusion equation
	dZ = sumhw/5. * BS_K * dZ;
	

	Z_new[idx] = Z[idx] + dZ * DT_MORPHO;

	// return;
	int maxid = idx;
	float maxh = hw[idx];
	if(hw[upIdx] > maxh){
		maxh = hw[upIdx];
		maxid = upIdx;
	}
	if(hw[downIdx] > maxh){
		maxh = hw[downIdx];
		maxid = downIdx;
	}
	if(hw[leftIdx] > maxh){
		maxh = hw[leftIdx];
		maxid = leftIdx;
	}
	if(hw[rightIdx] > maxh){
		maxh = hw[rightIdx];
		maxid = rightIdx;
	}

	atomicAdd(&QsA[maxid], abs(dZ) * CELLAREA);


	// if(diffusion > 0) printf("%f\n", diffusion );
	// if(isnan(diffusion)) printf("%f\n", BS_EXP);
	// if()
	// float Zw = Z[idx] + hw[idx];

	// // Transfer of Qsin
	// for(int j=0;j<NNEIGHBOURS;++j){
	// 	// Checking weight and indices
	// 	if(weights[j] <= 0) continue;
	// 	int nidx;
	// 	if(get_neighbour(idx, adder, j, nidx) == false) continue;
	// 	float tZw = Z[nidx] + hw[nidx];
	// 	if(Zw > tZw)

	// }
	
}

__global__ void diffuse_bed_v2(float *hw, float *Z, float *Z_new, unsigned char *BC) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int idx,adder;
	if(get_index(x, y, idx, adder, BC) == false) return;
	if(BC::can_out(BC[idx])) return;
	if(BC::can_receive(BC[idx]) == false) return;

	// Neighbors' indices
	// float thhw = 1e-3;
	// int upIdx; if(top(idx, adder, upIdx) == false){upIdx = idx;} if(hw[upIdx] < thhw && hw[idx] < thhw) upIdx = idx;
	// int downIdx; if(bottom(idx, adder, downIdx) == false){downIdx = idx;} if(hw[downIdx] < thhw && hw[idx] < thhw) downIdx = idx;
	// int leftIdx; if(left(idx, adder, leftIdx) == false){leftIdx = idx;} if(hw[leftIdx] < thhw && hw[idx] < thhw) leftIdx = idx;
	// int rightIdx; if(right(idx, adder, rightIdx) == false){rightIdx = idx;} if(hw[rightIdx] < thhw && hw[idx] < thhw) rightIdx = idx;


	// Neighbors' indices
	int upIdx; if(top(idx, adder, upIdx) == false){upIdx = idx;};
	int downIdx; if(bottom(idx, adder, downIdx) == false){downIdx = idx;};
	int leftIdx; if(left(idx, adder, leftIdx) == false){leftIdx = idx;};
	int rightIdx; if(right(idx, adder, rightIdx) == false){rightIdx = idx;};

	float thw = max( max( max( max(hw[upIdx], hw[idx]), hw[leftIdx]), hw[rightIdx]), hw[downIdx]);

	// Calculate new value for Z based on diffusion equation
	// float diffusion = BS_K * pow((Z[upIdx] + Z[downIdx] + Z[leftIdx] + Z[rightIdx] - 4 * Z[idx])/(DX*DX), BS_EXP) * pow(1 + hw[idx], BS_HW_EXP);
 // Calculate average value of neighbors
	float avg_neighbors =  (Z[upIdx] + Z[downIdx] + Z[leftIdx] + Z[rightIdx]) / 4.0;

	// Apply the diffusion equation
	Z_new[idx] = Z[idx] + pow(thw,1) * DT_MORPHO * BS_K * (avg_neighbors - Z[idx]);


	// if(diffusion > 0) printf("%f\n", diffusion );
	// if(isnan(diffusion)) printf("%f\n", BS_EXP);
	// if()
	// float Zw = Z[idx] + hw[idx];

	// // Transfer of Qsin
	// for(int j=0;j<NNEIGHBOURS;++j){
	// 	// Checking weight and indices
	// 	if(weights[j] <= 0) continue;
	// 	int nidx;
	// 	if(get_neighbour(idx, adder, j, nidx) == false) continue;
	// 	float tZw = Z[nidx] + hw[nidx];
	// 	if(Zw > tZw)

	// }
	
}

__global__ void diffuse_bed_v1(float *hw, float *Z, float *Z_new, unsigned char *BC) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int idx,adder;
	if(get_index(x, y, idx, adder, BC) == false) return;
	if(BC::can_out(BC[idx])) return;
	if(BC::can_receive(BC[idx]) == false) return;

	// Neighbors' indices
	int upIdx; if(top(idx, adder, upIdx) == false){upIdx = idx;};
	int downIdx; if(bottom(idx, adder, downIdx) == false){downIdx = idx;};
	int leftIdx; if(left(idx, adder, leftIdx) == false){leftIdx = idx;};
	int rightIdx; if(right(idx, adder, rightIdx) == false){rightIdx = idx;};

	// Calculate new value for Z based on diffusion equation
	// float diffusion = BS_K * pow((Z[upIdx] + Z[downIdx] + Z[leftIdx] + Z[rightIdx] - 4 * Z[idx])/(DX*DX), BS_EXP) * pow(1 + hw[idx], BS_HW_EXP);
 // Calculate average value of neighbors
	float avg_neighbors = (Z[upIdx] + Z[downIdx] + Z[leftIdx] + Z[rightIdx]) / 4.0;

	// Apply the diffusion equation
	Z_new[idx] = Z[idx] + DT_MORPHO * BS_K * (avg_neighbors - Z[idx]);


	// if(diffusion > 0) printf("%f\n", diffusion );
	// if(isnan(diffusion)) printf("%f\n", BS_EXP);
	// if()
	// float Zw = Z[idx] + hw[idx];

	// // Transfer of Qsin
	// for(int j=0;j<NNEIGHBOURS;++j){
	// 	// Checking weight and indices
	// 	if(weights[j] <= 0) continue;
	// 	int nidx;
	// 	if(get_neighbour(idx, adder, j, nidx) == false) continue;
	// 	float tZw = Z[nidx] + hw[nidx];
	// 	if(Zw > tZw)

	// }
	
}


__global__ void diffuse_bed_simple(float *hw, float *Z, float *Z_new, unsigned char *BC) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int idx,adder;
	if(get_index(x, y, idx, adder, BC) == false) return;
	if(BC::can_out(BC[idx])) return;
	if(BC::can_receive(BC[idx]) == false) return;

	// Neighbors' indices
	int upIdx; if(top(idx, adder, upIdx) == false){upIdx = idx;};
	int downIdx; if(bottom(idx, adder, downIdx) == false){downIdx = idx;};
	int leftIdx; if(left(idx, adder, leftIdx) == false){leftIdx = idx;};
	int rightIdx; if(right(idx, adder, rightIdx) == false){rightIdx = idx;};

	// Calculate new value for Z based on diffusion equation
	// float diffusion = BS_K * pow((Z[upIdx] + Z[downIdx] + Z[leftIdx] + Z[rightIdx] - 4 * Z[idx])/(DX*DX), BS_EXP) * pow(1 + hw[idx], BS_HW_EXP);
	float diffusion = BS_K * pow((Z[upIdx] + Z[downIdx] + Z[leftIdx] + Z[rightIdx] - 4 * Z[idx])/(DX*DX), BS_EXP); // simple diffusion, Works but meh
	Z_new[idx] = Z[idx] + diffusion * DT_MORPHO;


	// if(diffusion > 0) printf("%f\n", diffusion );
	// if(isnan(diffusion)) printf("%f\n", BS_EXP);
	// if()
	// float Zw = Z[idx] + hw[idx];

	// // Transfer of Qsin
	// for(int j=0;j<NNEIGHBOURS;++j){
	// 	// Checking weight and indices
	// 	if(weights[j] <= 0) continue;
	// 	int nidx;
	// 	if(get_neighbour(idx, adder, j, nidx) == false) continue;
	// 	float tZw = Z[nidx] + hw[nidx];
	// 	if(Zw > tZw)

	// }
	
}

// __global__ void diffuse(float *Z, float *Z_new, float alpha, int width, int height) {
//     int idx = get_index();
//     int i = idx / width;  // Row index
//     int j = idx % width;  // Column index
    
//     // Skip boundary cells if necessary
//     if(i == 0 || j == 0 || i == height - 1 || j == width - 1) return;

//     // Calculate indices of neighbors
//     int upIdx = up(idx, width, height);
//     int downIdx = down(idx, width, height);
//     int leftIdx = left(idx, width, height);
//     int rightIdx = right(idx, width, height);

//     // Calculate average value of neighbors
//     float avg_neighbors = (Z[upIdx] + Z[downIdx] + Z[leftIdx] + Z[rightIdx]) / 4.0;

//     // Apply the diffusion equation
//     Z_new[idx] = Z[idx] + alpha * (avg_neighbors - Z[idx]);
// }



// __global__ void diffuse_bed_simple(float *hw, float *Z, float *Z_new, unsigned char *BC) {
// 	int x = threadIdx.x + blockIdx.x * blockDim.x;
// 	int y = threadIdx.y + blockIdx.y * blockDim.y;
// 	int idx,adder;
// 	if(get_index(x, y, idx, adder, BC) == false) return;
// 	if(BC::can_out(BC[idx])) return;

// 	// Neighbors' indices
// 	int upIdx; if(top(idx, adder, upIdx) == false){upIdx = idx;};
// 	int downIdx; if(bottom(idx, adder, downIdx) == false){downIdx = idx;};
// 	int leftIdx; if(left(idx, adder, leftIdx) == false){leftIdx = idx;};
// 	int rightIdx; if(right(idx, adder, rightIdx) == false){rightIdx = idx;};

// 	// Calculate new value for Z based on diffusion equation
// 	float diffusion = BS_K * (Z[upIdx] + Z[downIdx] + Z[leftIdx] + Z[rightIdx] - 4 * Z[idx])/(DX*DX) * (1 + hw[idx]);
// 	Z_new[idx] = Z[idx] + diffusion;
// }