/*


*/


__global__ void compute_EROS_SS(float *hw, float *Z, float *QsA, float *QsB,  float *QsC,  float *QsD, uint8_t *BC) {

    // Getting the right IF
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx,adder;
	if(get_index(x, y, idx, adder, BC) == false) return;
	if(BC::can_out(BC[idx]) == true){return;};

    float surface_idx = hw[idx] + Z[idx];

    float SSdy = 1.;
    float SSdx = 1.;
    float SS = -1;
    int SSnidx = -1;
    int SSj = -1;

    bool bound = false;

    for(int j=0;j<NNEIGHBOURS;++j){

		int nidx;
		if(get_neighbour(idx, adder, j, nidx) == false) continue;
		
        
        // calculating local weight (i.e. Sw * dy)
        float ts = surface_idx - (hw[nidx] + Z[nidx]);

        if(ts <= 1e-6) continue; // aborting if higher neighbour
        
        // finishing slope calc
        ts /= DXS[j];

        if(BC::can_out(BC[nidx]))
        {
            bound = true;
            ts = BOUND_SLOPE;
            SS = ts;
            SSdy = DYS[j];
            SSdx = DXS[j];
            SSnidx = nidx;
            SSj = j;
            break;
        }

        if(ts > SS){
            SS = ts;
            SSdy = DYS[j];
            SSdx = DXS[j];
        	SSnidx = nidx;
            SSj = j;
        }

    }

    if(SSnidx == -1) return;

    float edot = 0.;
    float eldot = 0.;


    // float tau = RHO_WATER * GRAVITY * SS * min(hw[idx],1.);
    float tau = RHO_WATER * GRAVITY * SS * hw[idx];
    if(tau > TAU_C){
        edot = K_EROS * pow(tau - TAU_C,1.5);

        int onei;
        if(bound == false){
            if(get_oneighbourA(idx, adder, SSj, onei)){
                float tlatslope = (Z[onei] - Z[idx])/SSdy;
                if(tlatslope > 0){
                    float teldot =  tlatslope * KL_EROS * edot;
                    if(teldot * DT_MORPHO > (Z[onei] - Z[idx]) ) teldot = (Z[onei] - Z[idx])/DT_MORPHO;
                    eldot += teldot;
                    atomicAdd(&QsD[onei], teldot * SSdy * SSdx);
                }
            }
            if(get_oneighbourB(idx, adder, SSj, onei)){
                float tlatslope = (Z[onei] - Z[idx])/SSdy;
                if(tlatslope > 0){
                    float teldot =  tlatslope * KL_EROS * edot;
                    if(teldot * DT_MORPHO > (Z[onei] - Z[idx]) ) teldot = (Z[onei] - Z[idx])/DT_MORPHO;
                    eldot += teldot;
                    atomicAdd(&QsD[onei], teldot * SSdy * SSdx);
                }
            }
        }

    }

    double K = (1. / L_EROS);
    double edotpsy = (edot + eldot) / K;
    double C1 = QsA[idx] / SSdy - edotpsy;
    QsB[idx] = SSdy * (edotpsy + C1 * std::exp(-SSdx * K));
    
    // if(QsB[idx] > 0)
    //     printf("%f \n", QsB[idx]);

	atomicAdd(&QsC[SSnidx], QsB[idx] );

}



























































































// end of file