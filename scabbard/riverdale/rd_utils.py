import taichi as ti


@ti.kernel
def A_equals_B(A:ti.template(),B:ti.template()):
	for i,j in A:
		A[i,j] = B[i,j]


# Declare a function to calculate epsilon for a given value
@ti.func
def compute_epsilon_f32(value):
	eps = ti.cast(1.0, ti.f32)
	while ti.cast(value + eps, ti.f32) > value:
		eps /= 2.0
	return eps * 2.0


# Declare a function to calculate epsilon for a given value
@ti.func
def compute_epsilon_f64(value):
	eps = ti.cast(1.0, ti.f64)
	while ti.cast(value + eps, ti.f64) > value:
		eps /= 2.0
	return eps * 2.0


	