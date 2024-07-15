/*


*/



__global__ void set2val(float* arr, float val, int sizzla){
	// Getting the right index
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    if(x >= sizzla) return;
    arr[x] = val;
}
