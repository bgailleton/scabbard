'''
This module provides utility functions for the Riverdale model, primarily focusing on Taichi operations.

Author: B.G.
'''

import taichi as ti


@ti.kernel
def A_equals_B(A:ti.template(),B:ti.template()):
	"""
	Taichi kernel to copy the values from one Taichi field (B) to another (A).

	Args:
		A (ti.template()): The destination Taichi field.
		B (ti.template()): The source Taichi field.

	Returns:
		None: The values of A are updated in-place.

	Author: B.G.
	"""
	for i,j in A:
		A[i,j] = B[i,j]


@ti.func
def compute_epsilon_f32(value):
	"""
	Computes the machine epsilon for a float32 value.

	Machine epsilon is the smallest number that, when added to 1.0,
	results in a value greater than 1.0 in floating-point arithmetic.

	Args:
		value (float): A float32 value (used for type casting).

	Returns:
		float: The machine epsilon for float32.

	Author: B.G.
	"""
	eps = ti.cast(1.0, ti.f32)
	while ti.cast(value + eps, ti.f32) > value:
		eps /= 2.0
	return eps * 2.0


@ti.func
def compute_epsilon_f64(value):
	"""
	Computes the machine epsilon for a float64 value.

	Machine epsilon is the smallest number that, when added to 1.0,
	results in a value greater than 1.0 in floating-point arithmetic.

	Args:
		value (float): A float64 value (used for type casting).

	Returns:
		float: The machine epsilon for float64.

	Author: B.G.
	"""
	eps = ti.cast(1.0, ti.f64)
	while ti.cast(value + eps, ti.f64) > value:
		eps /= 2.0
	return eps * 2.0


	