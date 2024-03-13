import numpy as np





def str2type(tinput):
	tinput = tinput.lower()

	if(tinput == 'u8'):
		return np.uint8
	if(tinput == 'i32'):
		return np.int32
	if(tinput == 'f32'):
		return np.float32
	raise TypeError(f'type {tinput} not of u8, i or f') ;