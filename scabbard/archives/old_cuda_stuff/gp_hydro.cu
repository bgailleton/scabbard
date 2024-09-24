


// transfers Qwin on a node to node basis
__global__ void compute_static_Qwin(float *hw, float *Z, float *QwA, float *QwB, float *normgradZ, float *sum_grad_Zw, unsigned char *BC) {

	// Getting the right Index  and checking boundary conditions
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int idx,adder;
	if(get_index(x, y, idx, adder, BC) == false) return;
	if(BC::can_out(BC[idx])) return;

	// Local hydraulic surface elevation Zw
	float Zw = hw[idx] + Z[idx];

	// local array of weight partioning
	float slopes[NNEIGHBOURS];
	for(int j=0;j<NNEIGHBOURS;++j) slopes[j] = 0.;

	float SSx = 0.;	
	float SSy = 0.;

	// Summing the Zw gradients	
	sum_grad_Zw[idx] = 0.;
	normgradZ[idx] = 0.;
	
	// Cheap local minima solver: if I have no neighbours I add water until it spills out
	while(sum_grad_Zw[idx] == 0.){

		// neighbour loop
		for(int j=0;j<NNEIGHBOURS;++j){

			// idx of the neighbours
			int nidx;
			if(get_neighbour(idx, adder, j, nidx) == false) continue;
			
			// calculating local weight (i.e. grad Zw)
			float ts = Zw - (hw[nidx] + Z[nidx]);

			if(ts<0) continue; // aborting if higher neighbour
			
			// finishing local gradient
			ts /= DXS[j];

			// saving weight
			slopes[j] = ts;

			// summing the weight
			sum_grad_Zw[idx] += ts;

			if(j == 0 || j == 3){
				if(ts > SSy) SSy = ts;
			} else{
				if(ts > SSx) SSx = ts;
			}

			// next
		}

		// CHeap local minima solver: incrementing water surface
		if(sum_grad_Zw[idx] == 0.) {
			return;
			hw[idx] += 0.0001;
			Zw += 0.0001;
		}

		// end
	}

	if(sum_grad_Zw[idx] == 0) return;

	normgradZ[idx] = sqrt(pow(SSx,2.) + pow(SSy,2.));

	// Transfer of Qwin
	for(int j=0;j<NNEIGHBOURS;++j){
		// Checking weight and indices
		if(slopes[j] <= 0) continue;
		int nidx;
		if(get_neighbour(idx, adder, j, nidx) == false) continue;
		
		// actual transfer			
		atomicAdd(&QwB[nidx], QwA[idx] * slopes[j]/sum_grad_Zw[idx]);
	}

}

// transfers Qwin on a node to node basis
__global__ void compute_static_Qwin_v2(float *hw, float *Z, float *QwA, float* normgradZ, float* sum_grad_Zw, unsigned char *BC) {

	// Getting the right Index  and checking boundary conditions
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int idx,adder;
	if(get_index(x, y, idx, adder, BC) == false) return;
	if(BC::can_out(BC[idx])) return;

	// Local hydraulic surface elevation Zw
	float Zw = hw[idx] + Z[idx];

	// local array of weight partioning
	float slopes[NNEIGHBOURS];
	for(int j=0;j<NNEIGHBOURS;++j) slopes[j] = 0.;

	float SSx = 0.;	
	float SSy = 0.;

	// Summing the Zw gradients	
	sum_grad_Zw[idx] = 0.;
	normgradZ[idx] = 0.;
	
	// Cheap local minima solver: if I have no neighbours I add water until it spills out
	while(sum_grad_Zw[idx] == 0.){

		// neighbour loop
		for(int j=0;j<NNEIGHBOURS;++j){

			// idx of the neighbours
			int nidx;
			if(get_neighbour(idx, adder, j, nidx) == false) continue;
			
			// calculating local weight (i.e. grad Zw)
			float ts = Zw - (hw[nidx] + Z[nidx]);

			if(ts<0) continue; // aborting if higher neighbour
			
			// finishing local gradient
			ts /= DXS[j];

			// saving weight
			slopes[j] = ts;

			// summing the weight
			sum_grad_Zw[idx] += ts;

			if(j == 0 || j == 3){
				if(ts > SSy) SSy = ts;
			} else{
				if(ts > SSx) SSx = ts;
			}

			// next
		}

		// CHeap local minima solver: incrementing water surface
		if(sum_grad_Zw[idx] == 0.) {
			return;
			hw[idx] += 0.0001;
			Zw += 0.0001;
		}

		// end
	}

	if(sum_grad_Zw[idx] == 0) return;

	normgradZ[idx] = sqrt(pow(SSx,2.) + pow(SSy,2.));

	// Transfer of Qwin
	for(int j=0;j<NNEIGHBOURS;++j){
		// Checking weight and indices
		if(slopes[j] <= 0) continue;
		int nidx;
		if(get_neighbour(idx, adder, j, nidx) == false) continue;
		float transfer =  DX/MANNING  * pow(hw[idx],5./3.) * slopes[j]/sqrt(normgradZ[idx]);
		// float transfer =  DX/MANNING  * pow(hw[idx],5./3.) * slopes[j]/sqrt(normgradZ[idx]);
		// actual transfer			
		atomicAdd(&QwA[nidx],transfer);
	}

}

// transfers Qwin on a node to node basis
__global__ void compute_static_Qwin_v3(float *hw, float *Z, float *QwA, float *QwB, float* QwC, unsigned char *BC) {

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
	
	// Cheap local minima solver: if I have no neighbours I add water until it spills out
	while(sum_grad_Zw == 0.){

		// neighbour loop
		for(int j=0;j<NNEIGHBOURS;++j){

			// idx of the neighbours
			int nidx;
			if(get_neighbour(idx, adder, j, nidx) == false) continue;
			
			// calculating local weight (i.e. grad Zw)
			float tw = Zw - (hw[nidx] + Z[nidx]);

			if(tw<0) continue; // aborting if higher neighbour
			
			// finishing local gradient
			tw /= DXS[j];

			// saving weight
			weights[j] = tw;

			// summing the weight
			sum_grad_Zw += tw;

			// if(j == 0 || j == 3){
			// 	if(tw > SSy) SSy = tw;
			// } else{
			// 	if(tw > SSx) SSx = tw;
			// }

			SSx = max(SSx,tw);

			// next
		}

		// CHeap local minima solver: incrementing water surface
		if(sum_grad_Zw == 0.) {
			hw[idx] += 0.0001;
			Zw += 0.0001;
		}

		// end
	}
	
	QwC[idx] = pow(hw[idx], 5./3.) * sqrt(SSx) * DX/MANNING;

	// Transfer of Qwin
	for(int j=0;j<NNEIGHBOURS;++j){
		// Checking weight and indices
		if(weights[j] <= 0) continue;
		int nidx;
		if(get_neighbour(idx, adder, j, nidx) == false) continue;
		
		// actual transfer			
		atomicAdd(&QwB[nidx], QwA[idx] * weights[j]/sum_grad_Zw);

	}

}

__global__ void increment_hw_v3(float *hw, float *Z, float *QwA, float* QwC, unsigned char *BC){
	// Getting the right Index  and checking boundary conditions
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int idx,adder;
	if(get_index(x, y, idx, adder, BC) == false) return;
	if(BC::can_out(BC[idx])) return;

	hw[idx] += (QwA[idx] - QwC[idx])/ CELLAREA * DT_HYDRO;
	if(hw[idx] < 0) hw[idx] = 0;
}






/// transfers Qwin on a node to node basis
__global__ void compute_static_h(float *hw, float *Z, float *QwA, float* normgradZ, float* sum_grad_Zw, unsigned char *BC) {

	// Getting the right Index  and checking boundary conditions
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int idx,adder;
	if(get_index(x, y, idx, adder, BC) == false) return;
	if(BC::can_out(BC[idx])) return;



	if(sum_grad_Zw[idx] <= 0 || normgradZ[idx] <= 0) return;


	// FULL ANALYTICAL
	float A = QwA[idx] * MANNING * sqrt(normgradZ[idx]);
	float B = DX * sum_grad_Zw[idx];
	hw[idx] = pow( A/B,3./5.); 

	// MEAN ANALYTICAL
	// float A = QwA[idx] * MANNING * sqrt(normgradZ);
	// float B = DX * sum_grad_Zw;
	// hw[idx] =  STABIL_GPHYDRO * hw[idx] + (1 - STABIL_GPHYDRO) * pow(A/B,3./5.); 


	// // FULL ITERATIVE
	// hw[idx] += (QwA[idx] - (DX * sqrt(normgradZ) * pow(hw[idx], 5./3.) / MANNING) )/ CELLAREA * DT_HYDRO;
	// if(hw[idx] < 0) hw[idx] = 0;
}


// transfers Qwin on a node to node basis
__global__ void compute_static_Qwin_linear_test(float *hw, float *Z, float *QwA, float *QwB, int *Nrecs, float *sum_dz, unsigned char *BC) {

	// Getting the right Index  and checking boundary conditions
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int idx,adder;
	if(get_index(x, y, idx, adder, BC) == false) return;
	if(BC::can_out(BC[idx])) return;

	// Local hydraulic surface elevation Zw
	float Zw = hw[idx] + Z[idx];

	// local array of weight partioning
	float slopes[NNEIGHBOURS];
	for(int j=0;j<NNEIGHBOURS;++j) slopes[j] = 0.;

	float SSx = 0.;	
	float SSy = 0.;

	// Summing the Zw gradients	
	float sum_grad_Zw = 0.;
	sum_dz[idx] = 0.;
	Nrecs[idx] = 0;

	
	// Cheap local minima solver: if I have no neighbours I add water until it spills out
	while(sum_grad_Zw == 0.){

		// neighbour loop
		for(int j=0;j<NNEIGHBOURS;++j){

			// idx of the neighbours
			int nidx;
			if(get_neighbour(idx, adder, j, nidx) == false) continue;
			
			// calculating local weight (i.e. grad Zw)
			float ts = Zw - (hw[nidx] + Z[nidx]);

			if(ts<0) continue; // aborting if higher neighbour

			Nrecs[idx]++;
			
			// finishing local gradient
			ts /= DXS[j];

			// saving weight
			slopes[j] = ts;

			// summing the weight
			sum_grad_Zw += ts;

			sum_dz[idx] += Z[idx] - Z[nidx] - hw[idx];

		}

		// CHeap local minima solver: incrementing water surface
		if(sum_grad_Zw == 0.) {
			// return;
			hw[idx] += 0.0001;
			Zw += 0.0001;
		}

		// end
	}

	if(sum_grad_Zw == 0) return;

	// Transfer of Qwin
	for(int j=0;j<NNEIGHBOURS;++j){
		// Checking weight and indices
		if(slopes[j] <= 0) continue;
		int nidx;
		if(get_neighbour(idx, adder, j, nidx) == false) continue;
		
		// actual transfer			
		atomicAdd(&QwB[nidx], QwA[idx] * slopes[j]/sum_grad_Zw);
	}

}



/// transfers Qwin on a node to node basis
__global__ void compute_static_h_linear_test(float *hw, float *Z, float *QwA, int* Nrecs, float* sum_grad_Zw, unsigned char *BC) {

	// Getting the right Index  and checking boundary conditions
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int idx,adder;
	if(get_index(x, y, idx, adder, BC) == false) return;
	if(BC::can_out(BC[idx])) return;



	if(Nrecs[idx] <= 0){
		hw[idx] = 0.;
		return;
	} 

	float B = sum_grad_Zw[idx];
	float B2 = pow(B,2);
	float A = Nrecs[idx];
	float C = -DX * QwA[idx];
	float D = B2 - 4*A*C;
	hw[idx] = (-B + sqrt(D))/(2*A); 

	// MEAN ANALYTICAL
	// float A = QwA[idx] * MANNING * sqrt(normgradZ);
	// float B = DX * sum_grad_Zw;
	// hw[idx] =  STABIL_GPHYDRO * hw[idx] + (1 - STABIL_GPHYDRO) * pow(A/B,3./5.); 


	// // FULL ITERATIVE
	// hw[idx] += (QwA[idx] - (DX * sqrt(normgradZ) * pow(hw[idx], 5./3.) / MANNING) )/ CELLAREA * DT_HYDRO;
	// if(hw[idx] < 0) hw[idx] = 0;
}





__global__ void add_Qw_local_linear_test_v2(int *indices, float *values, float *hw, float *hwTemp, const int sizid) {
	// Getting the right IF
	int x = threadIdx.x + blockIdx.x * blockDim.x;

	if(x >= sizid) return;

	float B = BOUND_SLOPE * DX;
	float B2 = pow(B,2);
	float A = 1;
	float C = - values[x];
	float D = B2 - 4*A*C;
	hw[indices[x]] = (-B + sqrt(D))/(2*A); 	
	hwTemp[indices[x]] = hw[indices[x]];

}


// transfers Qwin on a node to node basis
__global__ void linear_test_v2(float *hw, float *Z, float *hwTemp, unsigned char *BC) {

	// Getting the right Index  and checking boundary conditions
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int idx,adder;
	if(get_index(x, y, idx, adder, BC) == false) return;
	if(BC::can_out(BC[idx])) return;
	if(BC::can_receive(BC[idx]) == false) return;

	float A = 0;
	float B = 0;
	float Br = 0;
	float Bd = 0;
	int Nd = 0;
	int Nr = 0;
	float C = 0;
	float C1 = 0;
	float C2 = 0;

	// Cheap local minima solver: if I have no neighbours I add water until it spills out
	while(true){

		A = 0;
		B = 0;
		Br = 0;
		Bd = 0;
		Nd = 0;
		C1 = 0;
		C2 = 0;

		// neighbour loop
		for(int j=0;j<NNEIGHBOURS;++j){

			// idx of the neighbours
			int nidx;
			if(get_neighbour(idx, adder, j, nidx) == false) continue;

			// CASE RECEIVER
			if(Z[idx] + hw[idx] > Z[nidx] + hw[nidx]){
				A -= 1.;
				Br += - Z[idx] + hw[nidx] + Z[nidx];
			}
			// CASE DONOR
			else{
				++Nd;
				Bd += hw[nidx]; 
				C1 += hw[nidx];
				C2 += hw[nidx] + Z[nidx];
			}

		}

		if(Nd == 0) {
			// if(hw[idx]>0) printf("HWD0\n" );
			if(hw[idx] > 0) hwTemp[idx] = hw[idx];
			return;
		}

		// CHeap local minima solver: incrementing water surface
		if(A == 0. && Nd == 0) {
			// if(hw[idx]>0) hwTemp[idx] = hw[idx] + 0.001;
			hw[idx] += 0.0001;
			// return;
			// Z[idx] += 0.0001;
		}
		else{
			if(A == 0){
				A = -1;
			} 
			// if(hw_d == 0){
			// 	++Nd;
			// }
			break;
		}

		// end
	}

	B = -Nd * Bd + Br;
	C2 *= C1;
	C = -Nd * C1 * Z[idx] + C2;


	// ANALYTICAL SOLUTION
	// float Delta = pow(B,2) -  4 * A * C;
	// if( Delta < 0) {
	// 	printf("DELTA<0\n" );
	// 	return;
	// }

	//NEWTON RHAPSON

	float thw = hw[idx];
	float tol = 1;
	while(abs(tol) > 1e-6){
		tol = (A * pow(thw,2) + B * thw + C)/(2*A*thw + B);
		thw -= tol;
	}

	// float thw = (-B + sqrt(Delta))/(2*A);
	// if(thw < 0)
	// 	thw = (-B - sqrt(Delta))/(2*A);
	// if(thw < 0)
	// 	return;

	hwTemp[idx] = thw;
}


















// transfers Qwin on a node to node basis
__global__ void compute_static_Qwin_v4(float *hw, float *Z, float *QwA, float *QwB, int *Nrecs, float *sum_dz_r, float* SSx, float* SSy, unsigned char *BC) {

	// Getting the right Index  and checking boundary conditions
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int idx,adder;
	if(get_index(x, y, idx, adder, BC) == false) return;
	if(BC::can_out(BC[idx])) return;

	// Local hydraulic surface elevation Zw
	float Zw = hw[idx] + Z[idx];

	// local array of weight partioning
	float slopes[NNEIGHBOURS];
	for(int j=0;j<NNEIGHBOURS;++j) slopes[j] = 0.;

	SSx[idx] = 0.;	
	SSy[idx] = 0.;

	// Summing the Zw gradients	
	Nrecs[idx] = 0;
	sum_dz_r[idx] = 0.;

	float sum_grad_Zw = 0;
	
	// Cheap local minima solver: if I have no neighbours I add water until it spills out
	while(sum_grad_Zw == 0.){

		// neighbour loop
		for(int j=0;j<NNEIGHBOURS;++j){

			// idx of the neighbours
			int nidx;
			if(get_neighbour(idx, adder, j, nidx) == false) continue;
			
			
			// calculating local weight (i.e. grad Zw)
			float ts = Zw - (hw[nidx] + Z[nidx]);

			if(ts<0) continue; // aborting if higher neighbour

			Nrecs[idx]++;

			
			// finishing local gradient
			// ts /= DXS[j];

			// saving weight
			slopes[j] = ts;

			// summing the weight
			sum_grad_Zw += ts;

			sum_dz_r[idx] += hw[nidx] + Z[nidx]; 

			if(j == 0 || j == 3){
				if(ts > SSy[idx]) SSy[idx] = hw[idx] + Z[idx] - hw[nidx] - Z[nidx];
			} else{
				if(ts > SSx[idx]) SSx[idx] = hw[idx] + Z[idx] - hw[nidx] - Z[nidx];
			}

			// next
		}

		// CHeap local minima solver: incrementing water surface
		if(sum_grad_Zw == 0.) {
			// return;
			hw[idx] += 0.0001;
			Zw += 0.0001;
		}

		// end
	}

	if(sum_grad_Zw == 0) return;
	// sum_dz_r[idx] /= Nrecs[idx];

	// Transfer of Qwin
	for(int j=0;j<NNEIGHBOURS;++j){
		// Checking weight and indices
		if(slopes[j] <= 0) continue;
		int nidx;
		if(get_neighbour(idx, adder, j, nidx) == false) continue;
		
		// actual transfer			
		atomicAdd(&QwB[nidx], QwA[idx] * slopes[j]/sum_grad_Zw);
	}

}




// transfers Qwin on a node to node basis
__global__ void compute_hw_v4(float *hw, float *Z, float *QwA, int *Nrecs, float *sum_dz_r, float* SSx, float* SSy, unsigned char *BC) {

	// Getting the right Index  and checking boundary conditions
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int idx,adder;
	if(get_index(x, y, idx, adder, BC) == false) return;
	if(BC::can_out(BC[idx])) return;
	if(QwA[idx] == 0 && hw[idx] == 0) return;

	if(hw[idx] == 0){
		float normgradZ = pow(hw[idx] + SSx[idx],2) + pow(hw[idx] + SSy[idx],2);
		float A = QwA[idx] * MANNING * sqrt(normgradZ);
		float B = DX * (Nrecs[idx] * Z[idx] - sum_dz_r[idx]);
		hw[idx] = max(pow( A/B,3./5.), 0.); 
		return;
	}

	if(Nrecs[idx] == 0) return;
	// return;

	float B = Z[idx] - 1/Nrecs[idx] * sum_dz_r[idx];
	float C = MANNING/(Nrecs[idx] * pow(DX,0.5)) * QwA[idx];

	float thw = abs(sum_dz_r[idx]/Nrecs[idx] - Z[idx]);
	// float thw = hw[idx];
	float tol = 1;
	// while(abs(tol) > 1e-6){
		float fhw = ( pow(thw,8./3.) + B * pow(thw, 5./3.)) * pow(pow(SSx[idx],2) + pow(SSy[idx],2), -1./4.) - C;

		// float gabul =(2*pow( pow(SSx[idx],2) + pow(SSy[idx],2), 5./4.));
		float derfhw = (8./3. * pow(thw,5./3.) + 5./3. * B * pow(thw,2./3.) ) * pow(pow(SSx[idx],2) + pow(SSy[idx],2), -1./4.) - (pow(thw,8./3.) + B * pow(thw, 5./3.)) * (SSx[idx] + SSy[idx]) / (2*pow( pow(SSx[idx],2) + pow(SSy[idx],2), 5./4.));
		// if(isnan(fhw)) printf("%s, %f\n", "NANFW",thw);
		// if(isnan(derfhw)) printf("%s, %f\n", "NANderfhw",thw);
		// // if(derfhw == 0) return;

		tol = fhw/derfhw;
		thw -= tol;
	// }

	hw[idx] = STABIL_GPHYDRO * hw[idx] + (1-STABIL_GPHYDRO) * max(thw,0.);


}




















































// end of file