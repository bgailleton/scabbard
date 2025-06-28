"""
This module provides functions for computing drainage area metrics within the Riverdale model.

It includes Taichi kernels for calculating single-flow receivers and incrementing
drainage area, with options for handling local minima and precipitation input.

Author: B.G.
"""

import taichi as ti
import numpy as np
from enum import Enum
import scabbard.utils as scaut 
from scabbard.riverdale.rd_grid import GRID
import scabbard.riverdale.rd_grid as gridfuncs
import scabbard.riverdale.rd_LM as lmfuncs
import scabbard.riverdale.rd_helper_surfw as hsw
import scabbard.riverdale.rd_utils as rut

@ti.kernel
def compute_Sreceivers_Zw(receivers: ti.template(), Z: ti.template(), hw: ti.template(), BCs: ti.template()):
    """
    Computes the single-flow receivers for each node based on water surface elevation (Zw).

    Args:
        receivers (ti.field): 2D Taichi field to store the flat index of the steepest receiver.
        Z (ti.field): 2D Taichi field of topographic elevation.
        hw (ti.field): 2D Taichi field of flow depth.
        BCs (ti.field): 2D Taichi field of boundary condition codes.
    """
    for i, j in Z:
        # Assign the receiver to itself by default (for pits or outlets)
        receivers[i, j] = i * GRID.nx + j

        # If the node cannot give flow or is inactive, skip it
        if not gridfuncs.can_give(i, j, BCs) or not gridfuncs.is_active(i, j, BCs):
            continue

        SS = 0.0  # Steepest slope found so far

        # Iterate over neighbors (D4 connectivity)
        for k in range(4):
            ir, jr = gridfuncs.neighbours(i, j, k, BCs)

            # Skip if neighbor is invalid (e.g., outside grid or NoData)
            if ir == -1:
                continue

            # Calculate local hydraulic slope (Zw difference / distance)
            tS = hsw.Zw(Z, hw, i, j) - hsw.Zw(Z, hw, ir, jr)
            tS /= GRID.dx

            # If slope is non-positive, neighbor is not a downstream receiver
            if tS <= 0:
                continue

            # If this slope is steeper than the current steepest, update receiver
            if tS > SS:
                receivers[i, j] = ir * GRID.nx + jr
                SS = tS

@ti.kernel
def compute_Sreceivers_Zw_rand(receivers: ti.template(), Z: ti.template(), hw: ti.template(), BCs: ti.template()):
    """
    Computes single-flow receivers with a stochastic component in slope selection.

    Similar to `compute_Sreceivers_Zw`, but introduces randomness to the slope
    comparison, which can be useful for exploring flow path variability.

    Args:
        receivers (ti.field): 2D Taichi field to store the flat index of the steepest receiver.
        Z (ti.field): 2D Taichi field of topographic elevation.
        hw (ti.field): 2D Taichi field of flow depth.
        BCs (ti.field): 2D Taichi field of boundary condition codes.
    """
    for i, j in Z:
        receivers[i, j] = i * GRID.nx + j

        if not gridfuncs.can_give(i, j, BCs) or not gridfuncs.is_active(i, j, BCs):
            continue

        SS = 0.0

        for k in range(4):
            ir, jr = gridfuncs.neighbours(i, j, k, BCs)

            if ir == -1:
                continue

            tS = hsw.Zw(Z, hw, i, j) - hsw.Zw(Z, hw, ir, jr)
            tS /= GRID.dx
            tS *= ti.random()  # Introduce stochasticity

            if tS <= 0:
                continue

            if tS > SS:
                receivers[i, j] = ir * GRID.nx + jr
                SS = tS

@ti.kernel
def increment_DAD4(receivers: ti.template(), ptrrec: ti.template(), DA: ti.template()):
    """
    Increments the drainage area for each node in a D4 single-flow network.

    This function propagates drainage area values downstream based on pre-computed
    receivers and a pointer to the current receiver for each node.

    Args:
        receivers (ti.field): 2D Taichi field of single-flow receivers.
        ptrrec (ti.field): 2D Taichi field of pointers to the current receiver for each node.
        DA (ti.field): 2D Taichi field of drainage area values.
    """
    for i, j in receivers:
        idx = i * GRID.nx + j

        # If the node is an outlet (receiver is itself), skip
        if receivers[i, j] == idx:
            continue

        # Get row/col of the current pointer
        ip = ptrrec[i, j] // GRID.nx
        jp = ptrrec[i, j] % GRID.nx

        # Get the flat index of the receiver to the current pointer
        newrec = receivers[ip, jp]

        # Get row/col of the receiver of the pointer
        ni = newrec // GRID.nx
        nj = newrec % GRID.nx

        # If the receiver of the pointer is an outlet or local minima, stop propagation
        if receivers[ni, nj] == newrec:
            continue

        # Atomically add the current node's drainage area to its receiver
        ti.atomic_add(DA[ni, nj], GRID.dx * GRID.dy)
        # Update the pointer to the next receiver in the flow path
        ptrrec[i, j] = newrec

@ti.kernel
def increment_QWD4(receivers: ti.template(), ptrrec: ti.template(), QW: ti.template(), Pf: ti.template()):
    """
    Increments the accumulated water discharge (QW) for each node in a D4 single-flow network.

    This function propagates water discharge values downstream, weighted by precipitation,
    based on pre-computed receivers and a pointer to the current receiver for each node.

    Args:
        receivers (ti.field): 2D Taichi field of single-flow receivers.
        ptrrec (ti.field): 2D Taichi field of pointers to the current receiver for each node.
        QW (ti.field): 2D Taichi field of accumulated water discharge.
        Pf (ti.field): 2D Taichi field of precipitation input values.
    """
    for i, j in receivers:
        idx = i * GRID.nx + j

        if receivers[i, j] == idx:
            continue

        ip = ptrrec[i, j] // GRID.nx
        jp = ptrrec[i, j] % GRID.nx

        newrec = receivers[ip, jp]

        ni = newrec // GRID.nx
        nj = newrec % GRID.nx

        if receivers[ni, nj] == newrec:
            continue

        ti.atomic_add(QW[ni, nj], Pf[i, j])
        ptrrec[i, j] = newrec

def compute_drainage_area_D4(rd, fill=True, N='auto', random_rec=False, Precipitations=None):
    """
    Computes drainage area (or accumulated precipitation) in D4 direction on the GPU.

    This function is designed for flexibility in numerical analysis rather than
    being an optimized building block within a larger model.

    Args:
        rd: The initialized Riverdale object.
        fill (bool, optional): If True, runs priority flood to remove local minima.
                               Defaults to True.
        N (int or str, optional): Number of iterations for drainage area increment.
                                 If 'auto', it runs until convergence or a high limit.
                                 Defaults to 'auto'.
        random_rec (bool, optional): If True, adds stochasticity to receiver selection.
                                     Defaults to False.
        Precipitations (numpy.ndarray, optional): If provided (2D array), accumulates
                                                  precipitation * drainage area.
                                                  Defaults to None.

    Returns:
        numpy.ndarray: A NumPy array of drainage area or accumulated precipitation.
    """
    # Optional filling operation to remove local minima
    if fill:
        lmfuncs.priority_flood(rd)

    # Determine if precipitation accumulation is needed
    prec = Precipitations is not None

    # Fetch temporary Taichi fields for receivers and pointers
    rec, ptrrec = rd.query_temporary_fields(2, dtype=ti.i32)

    # Fetch field for drainage area or accumulated precipitation
    if not prec:
        DA, = rd.query_temporary_fields(1, dtype=ti.f32)
        DA.fill(GRID.dx * GRID.dy)  # Initialize with cell area
    else:
        DA, Pfield = rd.query_temporary_fields(2, dtype=ti.f32)
        Pfield.from_numpy(Precipitations * GRID.dx * GRID.dy) # Convert precipitation to volume
        rut.A_equals_B(DA, Pfield) # Initialize DA with precipitation values

    # Precompute the single-flow receivers
    if random_rec:
        compute_Sreceivers_Zw_rand(rec, rd.Z, rd.hw, rd.BCs)
    else:
        compute_Sreceivers_Zw(rec, rd.Z, rd.hw, rd.BCs)

    # Initialize the pointer to receivers to the first receivers of each node
    rut.A_equals_B(ptrrec, rec)

    # Set number of iterations for propagation
    auton = False
    if isinstance(N, str) and N.lower() == 'auto':
        N = 1000000  # A large number for auto-convergence
        auton = True

    # Run N iterations of drainage area incrementation
    for i in range(N):
        if auton and i > 0 and i % 1000 == 0:
            # Check for convergence if in auto mode
            cop = DA.to_numpy()
            if not prec:
                increment_DAD4(rec, ptrrec, DA)
            else:
                increment_QWD4(rec, ptrrec, DA, Pfield)
            if np.sum(np.abs(cop - DA.to_numpy())) == 0.0:
                break # Converged
        else:
            if not prec:
                increment_DAD4(rec, ptrrec, DA)
            else:
                increment_QWD4(rec, ptrrec, DA, Pfield)

    # Return the result as a NumPy array
    return DA.to_numpy()

# The following kernels (`compute_D4_nolm`, `compute_D4`, `step_DA_D4`)
# are marked as experimental and not currently used in the main `compute_drainage_area_D4` function.
# They are left as placeholders for potential future development or testing.

@ti.kernel
def compute_D4_nolm(Z: ti.template(), D4dir: ti.template(), BCs: ti.template()):
    """
    Experimental: Computes D4 flow directions without local minima filling.
    Do not use at the moment.
    """
    for i, j in Z:
        D4dir[i, j] = ti.uint8(5) # Default to no flow

        if not gridfuncs.can_give(i, j, BCs) or not gridfuncs.is_active(i, j, BCs):
            continue

        SS = 0.0
        lowest_higher_Z = 0.0

        for k in range(4):
            ir, jr = gridfuncs.neighbours(i, j, k, BCs)

            if ir == -1:
                continue

            tS = Z[i, j] - Z[ir, jr]
            tS /= GRID.dx

            if tS <= 0:
                if Z[ir, jr] < lowest_higher_Z or lowest_higher_Z == 0.0:
                    lowest_higher_Z = Z[ir, jr]
                continue

            if tS > SS:
                D4dir[i, j] = ti.uint8(k)
                SS = tS

@ti.kernel
def compute_D4(Z: ti.template(), D4dir: ti.template(), BCs: ti.template(), checker: ti.template()):
    """
    Experimental: Computes D4 flow directions, attempting to resolve local minima.
    Do not use at the moment.
    """
    for i, j in Z:
        D4dir[i, j] = ti.uint8(5)

        if not gridfuncs.can_give(i, j, BCs) or not gridfuncs.is_active(i, j, BCs):
            continue

        SS = 0.0
        lowest_higher_Z = 0.0
        checked = True

        while SS == 0.0:
            for k in range(4):
                ir, jr = gridfuncs.neighbours(i, j, k, BCs)

                if ir == -1:
                    continue

                tS = Z[i, j] - Z[ir, jr]
                tS /= GRID.dx

                if tS <= 0:
                    if Z[ir, jr] < lowest_higher_Z or lowest_higher_Z == 0.0:
                        lowest_higher_Z = Z[ir, jr]
                    continue

                if tS > SS:
                    D4dir[i, j] = ti.uint8(k)
                    SS = tS

            if SS == 0.0:
                if checked:
                    checker[None] += 1
                    checked = False
                Z[i, j] = max(lowest_higher_Z, Z[i, j]) + 1e-4

@ti.kernel
def step_DA_D4(Z: ti.template(), DA: ti.template(), temp: ti.template(), D4dir: ti.template(), BCs: ti.template()):
    """
    Experimental: Computes and transfers discharge for drainage area calculation.
    Do not use at the moment.
    """
    for i, j in Z:
        temp[i, j] = GRID.dx * GRID.dy

    for i, j in Z:
        if gridfuncs.is_active(i, j, BCs):
            if D4dir[i, j] == 5:
                continue
            ir, jr = gridfuncs.neighbours(i, j, D4dir[i, j], BCs)
            if ir > -1:
                ti.atomic_add(temp[ir, jr], DA[i, j])

    for i, j in Z:
        DA[i, j] = temp[i, j]