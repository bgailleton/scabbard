import taichi as ti


@ti.kernel
def A_equals_B(A:ti.template(),B:ti.template()):
	for i,j in A:
		A[i,j] = B[i,j]