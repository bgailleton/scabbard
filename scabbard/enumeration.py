from enum import Enum


class Flooder(Enum):
	"""
	Enumeration of available flood routing methods.

	Attributes:
		SFD_STATIC (int): Single Flow Direction (SFD) - Static.
		MFD_STATIC (int): Multiple Flow Direction (MFD) - Static.
		SFD_DYNAMIC (int): Single Flow Direction (SFD) - Dynamic.
		MFD_DYNAMIC (int): Multiple Flow Direction (MFD) - Dynamic.
		CAESAR_LS (int): CAESAR-Lisflood model.
		CAESAR_LS_OMP (int): CAESAR-Lisflood model with OpenMP parallelization.
		FLOODOS (int): FLOODOS model.
	"""
	SFD_STATIC = 0
	MFD_STATIC = 1
	SFD_DYNAMIC = 2
	MFD_DYNAMIC = 3
	CAESAR_LS = 4
	CAESAR_LS_OMP = 5
	FLOODOS = 6


class Topology(Enum):
	"""
	Enumeration of available flow routing topologies.

	Attributes:
		D8 (int): D8 (eight-direction) flow routing.
		D4 (int): D4 (four-direction) flow routing.
	"""
	D8 = 1
	D4 = 2