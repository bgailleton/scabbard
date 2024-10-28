import scabbard as scb
import numpy as np
import numba as nb


def compute_flow_distance_from_outlet(
    grid: scb.raster.RegularRasterGrid,
    method: str = "mean",
    BCs=None,
    Stack=None,
    fill_LM=False,
    step_fill=1e-3,
    D8=True,
):

    if method.lower() == "mean":

        if isinstance(grid, scb.raster.RegularRasterGrid):
            tBCs = scb.flow.get_normal_BCs(grid) if BCs is None else BCs
        else:
            raise RuntimeError(
                "compute_flow_distance_from_outlet requires a RegularRasterGrid as grid"
            )

        if Stack is None:
            if fill_LM:
                Stack = np.zeros_like(grid.Z.ravel(), dtype=np.uint64)
                scb.ttb.graphflood.funcdict['priority_flood_TO'](
                    grid.Z.ravel(), Stack, BCs.ravel(), grid.dims, D8, step_fill
                )
            else:
                Stack = np.argsort(grid.Z.ravel()).astype(np.uint64)

        return scb.ste.mean_dist_to_outlet(
            Stack,
            grid.Z.ravel(),
            tBCs.ravel(),
            D8,
            grid.geo.nx,
            grid.geo.ny,
            grid.geo.dx,
        ).reshape(grid.rshp)
    elif method.lower() == "min":

        if isinstance(grid, scb.raster.RegularRasterGrid):
            tBCs = scb.flow.get_normal_BCs(grid) if BCs is None else BCs
        else:
            raise RuntimeError(
                "compute_flow_distance_from_outlet requires a RegularRasterGrid as grid"
            )

        if Stack is None:
            if fill_LM:
                Stack = np.zeros_like(grid.Z.ravel(), dtype=np.uint64)
                scb.ttb.graphflood.funcdict['priority_flood_TO'](
                    grid.Z.ravel(), Stack, BCs.ravel(), grid.dims, D8, step_fill
                )
            else:
                Stack = np.argsort(grid.Z.ravel()).astype(np.uint64)

        return scb.ste.min_dist_to_outlet(
            Stack,
            grid.Z.ravel(),
            tBCs.ravel(),
            D8,
            grid.geo.nx,
            grid.geo.ny,
            grid.geo.dx,
        ).reshape(grid.rshp)
    elif method.lower() == "max":

        if isinstance(grid, scb.raster.RegularRasterGrid):
            tBCs = scb.flow.get_normal_BCs(grid) if BCs is None else BCs
        else:
            raise RuntimeError(
                "compute_flow_distance_from_outlet requires a RegularRasterGrid as grid"
            )

        if Stack is None:
            if fill_LM:
                Stack = np.zeros_like(grid.Z.ravel(), dtype=np.uint64)
                scb.ttb.graphflood.funcdict['priority_flood_TO'](
                    grid.Z.ravel(), Stack, BCs.ravel(), grid.dims, D8, step_fill
                )
            else:
                Stack = np.argsort(grid.Z.ravel()).astype(np.uint64)

        return scb.ste.max_dist_to_outlet(
            Stack,
            grid.Z.ravel(),
            tBCs.ravel(),
            D8,
            grid.geo.nx,
            grid.geo.ny,
            grid.geo.dx,
        ).reshape(grid.rshp)
    else:
        raise RuntimeError("Methods can be: 'mean' or 'min' or 'max'. More will come.")
