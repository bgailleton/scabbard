# -*- coding: utf-8 -*-
"""
This module provides a high-level interface for running the Graphflood model.

It abstracts away the complexities of different backends (GPU, Dagger, TopoToolbox)
and provides a unified API for simulating hydrological processes.
"""

# __author__ = "B.G."

import numpy as np
import scabbard as scb
import dagger as dag
import taichi as ti

def _std_run_gpu_ndt(
    grid,
    P=None,  # precipitations, numpy array or scalar
    BCs=None,  # Boundary codes
    N_dt=5000,
    dt=1e-3,
    manning=0.033,
    init_hw=None,
):
    """
    Internal runner function for the standalone GPU (Taichi) backend.

    Args:
        grid (scb.raster.RegularRasterGrid): The input grid.
        P (numpy.ndarray or float, optional): Precipitation rate. Defaults to None.
        BCs (numpy.ndarray, optional): Boundary conditions. Defaults to None.
        N_dt (int, optional): Number of time steps. Defaults to 5000.
        dt (float, optional): Time step size. Defaults to 1e-3.
        manning (float, optional): Manning's roughness coefficient. Defaults to 0.033.
        init_hw (numpy.ndarray, optional): Initial water depth. Defaults to None.

    Returns:
        dict: A dictionary containing simulation results and model objects.
    """
    try:
        ti.init(ti.gpu)
    except Exception:
        raise RuntimeError("Could not initiate Taichi to GPU. Check Taichi installation.")

    # Create and configure the riverdale parameter object
    param = scb.rvd.param_from_grid(grid)
    param.manning = manning
    if init_hw is not None:
        param.initial_hw = init_hw
    if BCs is not None:
        param.BCs = BCs
    param.dt_hydro = dt
    param.precipitations = P

    # Compile parameters to a riverdale model object
    rd = scb.rvd.create_from_params(param)

    # Run the simulation
    rd.run_hydro(N_dt)

    # Return results
    return {'h': rd.hw.to_numpy(), 'Qi': rd.QwA.to_numpy(), 'Qo': rd.QwC.to_numpy(),
            'model': rd, 'param': param, 'backend_graphflood': 'riverdale'}

def _std_rerun_gpu_ndt(
    model_input,
    N_dt=1000,
    new_dt=None
):
    """
    Internal function to continue a simulation using the GPU (Taichi) backend.

    Args:
        model_input (dict): The output dictionary from a previous `std_run` call.
        N_dt (int, optional): Number of additional time steps. Defaults to 1000.
        new_dt (float, optional): New time step size. If None, uses existing. Defaults to None.

    Returns:
        dict: An updated dictionary containing simulation results and model objects.
    """
    if new_dt is not None:
        scb.rvd.change_dt_hydro(model_input['model'], model_input['param'], new_dt)

    rd = model_input['model']
    rd.run_hydro(N_dt)

    return {'h': rd.hw.to_numpy(), 'Qi': rd.QwA.to_numpy(), 'Qo': rd.QwC.to_numpy(),
            'model': rd, 'param': model_input['param'], 'backend_graphflood': 'riverdale'}

def _std_run_dagger_ndt(
    grid,
    P=None,  # precipitations, numpy array or scalar
    BCs=None,
    N_dt=5000,
    dt=1e-3,
    manning=0.033,
    init_hw=None,
    **kwargs
):
    """
    Internal function to run the simulation using the Dagger (C++) backend.

    Args:
        grid (scb.raster.RegularRasterGrid): The input grid.
        P (numpy.ndarray or float, optional): Precipitation rate. Defaults to None.
        BCs (numpy.ndarray, optional): Boundary conditions. Defaults to None.
        N_dt (int, optional): Number of time steps. Defaults to 5000.
        dt (float, optional): Time step size. Defaults to 1e-3.
        manning (float, optional): Manning's roughness coefficient. Defaults to 0.033.
        init_hw (numpy.ndarray, optional): Initial water depth. Defaults to None.
        **kwargs: Additional keyword arguments, e.g., 'SFD' for single flow direction.

    Returns:
        dict: A dictionary containing simulation results and model objects.
    """
    # Initialize Dagger objects
    con = dag.D8N(grid.geo.nx, grid.geo.ny, grid.geo.dx, grid.geo.dx, grid.geo.xmin, grid.geo.ymin)
    graph = dag.graph(con)

    if BCs is not None:
        con.set_custom_boundaries(BCs.ravel())

    flood = dag.graphflood(graph, con)
    flood.set_topo(grid.Z.ravel())

    # Set flow direction mode (SFD or MFD)
    setsfd = kwargs.get('SFD', False)
    if setsfd:
        flood.enable_SFD()
    else:
        flood.enable_MFD()

    # Configure simulation parameters
    flood.fill_minima()
    flood.disable_courant_dt_hydro()
    flood.set_dt_hydro(dt)
    flood.set_mannings(manning)

    if isinstance(P, np.ndarray):
        flood.set_water_input_by_variable_precipitation_rate(P.ravel())
    else:
        flood.set_water_input_by_constant_precipitation_rate(P)

    if init_hw is not None:
        flood.set_hw(init_hw.ravel())

    # Run the simulation
    for _ in range(N_dt):
        flood.run()

    return {'h': flood.get_hw().reshape(grid.rshp), 'Qi': flood.get_Qwin().reshape(grid.rshp),
            'Qo': flood.compute_tuqQ(3).reshape(grid.rshp), 'model': flood, 'param': None,
            'backend_graphflood': 'dagger'}

def _std_run_ttb_ndt(
    grid,
    P=None,  # precipitations, numpy array or scalar
    BCs=None,
    N_dt=5000,
    dt=1e-3,
    manning=0.033,
    init_hw=None,
    **kwargs
):
    """
    Internal function to run the simulation using the TopoToolbox backend.

    Args:
        grid (scb.raster.RegularRasterGrid): The input grid.
        P (numpy.ndarray or float, optional): Precipitation rate. Defaults to None.
        BCs (numpy.ndarray, optional): Boundary conditions. Defaults to None.
        N_dt (int, optional): Number of time steps. Defaults to 5000.
        dt (float, optional): Time step size. Defaults to 1e-3.
        manning (float, optional): Manning's roughness coefficient. Defaults to 0.033.
        init_hw (numpy.ndarray, optional): Initial water depth. Defaults to None.
        **kwargs: Additional keyword arguments, e.g., 'SFD' for single flow direction.

    Returns:
        dict: A dictionary containing simulation results.
    """
    import topotoolbox as ttb

    ttbgrid = grid.grid2ttb()

    sfd = kwargs.get('SFD', False)
    d8 = not kwargs.get('D4', False)

    res = ttb.run_graphflood(
        ttbgrid,
        initial_hw=init_hw,
        bcs=BCs,
        dt=dt,
        p=P,
        manning=manning,
        sfd=sfd,
        d8=d8,
        n_iterations=N_dt
    )

    return {'h': scb.raster.raster_from_ttb(res), 'Qi': None, 'Qo': None,
            'model': None, 'param': None, 'backend_graphflood': 'ttb'}

def std_run(
    model_input,  # String (path to raster) or grid object
    P=1e-5,  # precipitations, numpy array or scalar
    BCs=None,  # Boundary codes
    N_dt=5000,
    backend='gpu',
    dt=1e-3,
    manning=0.033,
    init_hw=None,
    **kwargs
):
    """
    Standardized function to run the Graphflood model with various backends.

    Args:
        model_input (str or scb.raster.RegularRasterGrid): Path to a raster file or a grid object.
        P (numpy.ndarray or float, optional): Precipitation rate. Defaults to 1e-5.
        BCs (numpy.ndarray, optional): Boundary conditions. Defaults to None.
        N_dt (int, optional): Number of time steps. Defaults to 5000.
        backend (str, optional): The simulation backend to use ('gpu', 'dagger', 'ttb').
                                Defaults to 'gpu'.
        dt (float, optional): Time step size. Defaults to 1e-3.
        manning (float, optional): Manning's roughness coefficient. Defaults to 0.033.
        init_hw (numpy.ndarray, optional): Initial water depth. Defaults to None.
        **kwargs: Additional keyword arguments passed to the backend-specific runner.

    Returns:
        dict: A dictionary containing simulation results and model objects.

    Raises:
        TypeError: If `model_input` is not a string or a RegularRasterGrid.
    """
    if isinstance(model_input, str):
        grid = scb.io.load_raster(model_input)
    elif isinstance(model_input, scb.raster.RegularRasterGrid):
        grid = model_input
    else:
        raise TypeError("model_input must be a string (path) or a RegularRasterGrid object.")

    if backend.lower() == 'gpu':
        return _std_run_gpu_ndt(grid, P, BCs, N_dt, dt, manning, init_hw=init_hw, **kwargs)
    elif backend.lower() in ['dagger', 'cpu']:
        return _std_run_dagger_ndt(grid, P, BCs, N_dt, dt, manning, init_hw, **kwargs)
    elif backend.lower() in ['ttb', 'topotoolbox']:
        return _std_run_ttb_ndt(grid, P, BCs, N_dt, dt, manning, init_hw, **kwargs)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

def std_rerun(model_input, N_dt=1000, new_dt=None):
    """
    Continues a Graphflood simulation from a previously returned model output.

    Args:
        model_input (dict): The output dictionary from a previous `std_run` call.
        N_dt (int, optional): Number of additional time steps. Defaults to 1000.
        new_dt (float, optional): New time step size. If None, uses existing. Defaults to None.

    Returns:
        dict: An updated dictionary containing simulation results and model objects.

    Raises:
        RuntimeError: If `model_input` is not a valid output from `std_run`.
    """
    if not isinstance(model_input, dict) or 'backend_graphflood' not in model_input:
        raise RuntimeError("model_input must be a dictionary returned by std_run.")

    backend = model_input['backend_graphflood']

    if backend == 'riverdale':
        return _std_rerun_gpu_ndt(model_input, N_dt, new_dt)
    elif backend == 'dagger':
        # Dagger backend does not have a separate rerun function, just continue running
        flood = model_input['model']
        if new_dt is not None:
            flood.set_dt_hydro(new_dt)
        for _ in range(N_dt):
            flood.run()
        return {'h': flood.get_hw().reshape(model_input['h'].shape), 'Qi': flood.get_Qwin().reshape(model_input['Qi'].shape),
                'Qo': flood.compute_tuqQ(3).reshape(model_input['Qo'].shape), 'model': flood, 'param': None,
                'backend_graphflood': 'dagger'}
    elif backend == 'ttb':
        # TopoToolbox backend does not have a separate rerun function, just continue running
        # This would require re-calling run_graphflood with updated parameters
        # For simplicity, this example assumes a new run for ttb, or a more complex state management
        raise NotImplementedError("Rerunning for TopoToolbox backend is not directly supported via this interface.")
    else:
        raise ValueError(f"Unsupported backend for rerun: {backend}")