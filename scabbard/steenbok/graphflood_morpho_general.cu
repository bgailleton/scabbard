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
__global__ void increment_hs(float *hw, float *Z,float *QsA, float *QsB, unsigned char *BC) {

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

	Z[idx] += float(dhs);


}






// end of file