import scabbard as scb
# from scabbard import local_file_picker
from nicegui import ui
import os
import numpy as np
from scabbard.riverdale.rd_params import param_from_dem
from scabbard.riverdale.rd_env import create_from_params, load_riverdale
import taichi as ti
import scabbard as scb
import matplotlib.pyplot as plt
import numpy as np
import time
import scabbard.riverdale.rd_grid as gridfuncs
import scabbard.riverdale.rd_hydrodynamics as hyd
import scabbard.riverdale.rd_hydrometrics as rdta
import scabbard.riverdale.rd_LM as lm
import scabbard.riverdale.rd_drainage_area as rda
from scabbard.riverdale.rd_hillshading import hillshading


dark = ui.dark_mode()

dark.enable()

ti.init(ti.gpu)

fig = None
rd = None
colormap = None
im = None
color_range = None
value = None
model = {"range": {"min": 0, "max": 100}}

current_path = os.getcwd()

def update_flow_depth():
    global im
    global fig
    global colormap
    global rd
    global value

    with fig:
        # colormap.remove()
        value = rd.hw.to_numpy()
        im.set_data(value)
        im.set_cmap('Blues')
        im.set_clim(0,1)


def update_Qw():
    global im
    global fig
    global colormap
    global rd
    global value

    with fig:
        # colormap.remove()
        value = rd.QwA.to_numpy()
        im.set_data(value)
        im.set_cmap('Purples')
        im.set_clim(value.min(),np.percentile(value,98))

def update_u():
    global im
    global fig
    global colormap
    global rd
    global value

    with fig:
        # colormap.remove()
        value = rdta.compute_flow_velocity(rd)
        im.set_data(value)
        im.set_cmap('viridis')
        im.set_clim(value.min(),np.percentile(value,98))

def update_effa():
    global im
    global fig
    global colormap
    global rd
    global value

    with fig:
        # colormap.remove()
        value = rdta.compute_effective_drainage_area(rd)
        im.set_data(value)
        im.set_cmap('cividis')
        im.set_clim(value.min(),np.percentile(value,90))

def update_shr():
    global im
    global fig
    global colormap
    global rd
    global value

    with fig:
        # colormap.remove()
        value = rdta.compute_shear_stress(rd)
        im.set_data(value)
        im.set_cmap('magma')
        im.set_clim(value.min(),np.percentile(value,95))

def update_clim():
    global im
    global fig
    global model

    with fig:
        im.set_clim(model['range']['min'], model['range']['max'])



async def pick_file() -> None:
    global load_button
    global rd
    global value
    global im
    global colormap
    global fig
    global color_range
    global model
    result = await scb.local_file_picker(current_path, multiple=True)
    # print(result)
    # quit()
    ui.notify(f'Loading {result[0]}')
    
    rd = load_riverdale(result[0])
    value = rd.Z.to_numpy()
    load_button.delete()

    ui.notify(f'Loaded !')


    with ui.row():
        with ui.column():

            fig = ui.matplotlib(figsize=(12, 12)).figure 

            with fig as tfig:
                ax = tfig.gca()
                im = ax.imshow(value, cmap = "gist_earth" )
                ax.imshow(hillshading(rd,), cmap = 'gray',alpha = 0.45)
                colormap = plt.colorbar(im)

        with ui.column():
            ui.button('Flow depth', on_click = update_flow_depth)
            ui.button('Qw', on_click = update_Qw)
            ui.button('Flow velocity', on_click = update_u)
            ui.button('Eff. area', on_click = update_effa)
            ui.button('Shear Stress', on_click = update_shr)

    # I'll need to make a factory for that 
    ui.label('Colormap range')
    color_range = ui.range(min=value.min(), max=value.max(), step = (value.max() - value.min())/250, value = {'min': 1000, 'max': 1400}) \
    .props('label-always snap label-color="secondary" right-label-text-color="black"', ).bind_value(model,'range').on('change', update_clim, throttle = 5)

    # ui.button('YOLO')



# @ui.page('/')
# def index():

ui.markdown('# Graphflood - Riverdale')
load_button = ui.button('Choose file', on_click=pick_file, icon='folder')

if __name__ in {"__main__", "__mp_main__"}:
    ui.run()
