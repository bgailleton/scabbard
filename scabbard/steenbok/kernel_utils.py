import pycuda
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np
import scabbard.steenbok.dtype_helper as typh

def get_current_function_name():
    return inspect.currentframe().f_back.f_code.co_name


def set_constant(mod, val, ref, dtype):
	'''
	Take a source module and set a global constant from it
	'''
	# Forcing a type
	ttype = typh.str2type(dtype)
	val_cpu = ttype(val)
	# getting the gpu value
	val_gpu = mod.get_global(ref)[0]
	drv.memcpy_htod(val_gpu, val_cpu)
	return val_cpu, val_gpu

def set_array(mod, val, ref, dtype):
	'''
	Take a source module and set a given array in gpu side
	'''
	# Forcing a type
	ttype = typh.str2type(dtype)
	val_cpu = ttype(val)

	val_gpu = drv.mem_alloc(val_cpu.nbytes)
	drv.memcpy_htod(val_gpu, val_cpu)
	return val_cpu, val_gpu


def get_array(mod, arrcpu, arrgpu):
	'''
	transfer an array from gpu to cpu
	'''
	drv.memcpy_dtoh(arrcpu, arrgpu)



class arrayHybrid:
	'''
		Small data structure helpinh wiht array management from gpu to cpu and all
	'''
	def __init__(self, mod, val, ref, ttype):
		
		'''
			mod: the SourceModule
			val: the array
			ref: the string name
			ttype: the type in homemade format
		'''
		self.dtype = ttype		
		self._dtype = typh.str2type(ttype)
		self.mod = mod
		self.ref = ref
		self._cpu, self._gpu = set_array(self.mod, val, self.ref, self.dtype)
		# print("HHH",self._cpu)
		self.nn = self._cpu.shape[0]

	def cpu2gpu(self):
		drv.memcpy_dtoh(self._cpu, self._gpu)

	def delete(self, cpu = False, gpu = True):
		if(gpu):
			self._gpu.free()
		if(cpu):
			self._cpu = None

	def set(self,val, gpu = True, cpu = True):
		if(gpu and cpu):
			self._cpu, self._gpu = set_array(self.mod, val, self.ref, self.dtype)
		else:
			if(cpu):
				self._cpu = val
			if(gpu):
				a, self._gpu = set_array(self.mod, val, self.ref, self.dtype)

	def get(self, cpu = False, gpu = True):
		if(cpu == gpu):
			print('you need to pick either cpu or gpu in the getter, not both or any')
			return
		if(cpu):
			return self._cpu

		get_array(self.mod, self._cpu, self._gpu)

		return self._cpu



def aH_zeros_like(mod, arr, ttype, ref = "temp"):

	val = np.zeros_like(arr, dtype = typh.str2type(ttype))

	return arrayHybrid(mod, val, ref, ttype)

def aH_zeros(mod, sizzla, ttype, ref = "temp"):

	val = np.zeros(sizzla, dtype = typh.str2type(ttype))

	return arrayHybrid(mod, val, ref, ttype)

