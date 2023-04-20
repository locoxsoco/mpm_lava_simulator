import os
import argparse
import taichi as ti
import numpy as np
import time
from config_builder import SimConfig
from MOLASSES.driver import Driver as MOLASSESDriver
from MAGFLOW.driver import Driver as MAGFLOWDriver
from utils import *

ti.init(arch=ti.gpu)
particles_pos = ti.Vector.field(3, dtype=ti.f32, shape = 1)
vector_outside = ti.Vector([-9999, -9999, -9999])

######################### Debug Parameters #########################
normal_line_column = 0
debug_normals_checkbox = 0
debug_grid_checkbox = 1
debug_mesh_checkbox = 0
run_state = 0
pulse_state = 0
#######################################################################


def show_options(gui,pulseVolume,grid,simulation_time):
    global normal_line_column
    global debug_normals_checkbox
    global debug_grid_checkbox
    global debug_mesh_checkbox
    global run_state
    global pulse_state

    with gui.sub_window("Debug", 0.0, 0.875, 0.14, 0.125) as w:
        debug_normals_checkbox = w.checkbox("Show normals", debug_normals_checkbox)
        normal_line_column = w.slider_int("Column", normal_line_column, 0, 100)
        debug_grid_checkbox = w.checkbox("Show grid", debug_grid_checkbox)
        debug_mesh_checkbox = w.checkbox("Show mesh", debug_mesh_checkbox)
    
    run_state_text = 'Running' if run_state else 'Paused'
    with gui.sub_window(f'Simulation status: {run_state_text}', 0.0, 0.0, 0.25, 0.185) as w:
        # w.text(f'Volume Erupted: {round(volumeErupted,2)} km3')
        w.text(f'Simulation time: {simulation_time} s')
        if w.button("Run"):
            run_state = 1
        if w.button("Pause"):
            run_state = 0
        if w.button("Step"):
            run_state = 2
        pulseVolume = w.slider_float("Volume per pulse (km3)", pulseVolume, 0.0, 0.001)
        if w.button("Emit Pulse"):
            pulse_state = 1
        if w.button("Pause Pulse"):
            pulse_state = 0
    
    customPulseSize = 0.0
    customPulseVolume = 0.0
    with gui.sub_window(f'Pulse Brush', 0.0, 0.185, 0.145, 0.125) as w:
        customPulseSize = w.slider_float("Size (km)", customPulseSize, 0.0, 0.5)
        customPulseVolume = w.slider_float("Volume (m3)", customPulseVolume, 0.0, 1.0)
    
    # with gui.sub_window(f'Lava Properties', 0.0, 0.190, 0.145, 0.125) as w:
    #     grid.lava_density = w.slider_float("Lava density (kg/m3)", grid.lava_density, 2000.0, 4000.0)
    
    return pulseVolume/10.0

def render(camera,window,scene,canvas,heightmap,grid):
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)

    if(debug_mesh_checkbox):
        scene.mesh(vertices=heightmap.heightmap_positions, indices=heightmap.heightmap_indices,per_vertex_color=heightmap.heightmap_colors,normals=heightmap.heightmap_normals)

    if(debug_grid_checkbox):
        scene.mesh_instance(vertices=grid.cube_positions, indices=grid.cube_indices,per_vertex_color=grid.cube_colors_lvl0, transforms=grid.m_transforms_lvl0)
        scene.mesh_instance(vertices=grid.cube_positions2, indices=grid.cube_indices,per_vertex_color=grid.cube_colors_lvl1, transforms=grid.m_transforms_lvl1)
        scene.particles(particles_pos, color = (0.0, 1.0, 0.0), radius = 0.1)
        scene.mesh_instance(vertices=grid.cube_positions3, indices=grid.cube_indices,per_vertex_color=grid.cube_colors_lvl2, transforms=grid.m_transforms_lvl2)
    if(debug_normals_checkbox):
        for i in range(45):
            scene.lines(heightmap.verts, color = (0.28, 0.68, 0.99), width = 0.5, vertex_count = 2, vertex_offset = 4*(normal_line_column*(heightmap.hm_width_px)+i+75))

    scene.ambient_light((0.5, 0.5, 0.5))
    scene.point_light(pos=(heightmap.hm_width_px*heightmap.px_to_km/2.0, 3.0*heightmap.hm_elev_range_km/2.0, heightmap.hm_height_px*heightmap.px_to_km/2.0), color=(0.5, 0.5, 0.5))
    scene.point_light(pos=(heightmap.hm_width_px*heightmap.px_to_km/2.0, 3.0*heightmap.hm_elev_range_km/2.0, 3.0*heightmap.hm_height_px*heightmap.px_to_km/2.0), color=(0.5, 0.5, 0.5))

    canvas.scene(scene)

def main():
    global run_state
    global pulse_state
    
    parser = argparse.ArgumentParser(description='Lava Sim Taichi')
    parser.add_argument('--scene_file',
                        default='',
                        help='scene file')
    args = parser.parse_args()
    scene_path = args.scene_file

    config = SimConfig(scene_file_path=scene_path)
    heightmap_path = config.get_cfg("heightmapFile")
    simulation_method = config.get_cfg("simulationMethod")
    dim = config.get_cfg("dim")
    n_grid = config.get_cfg("nGrid")
    hm_elev_min_m = config.get_cfg("elevMinMeters")
    hm_elev_max_m = config.get_cfg("elevMaxMeters")

    if(simulation_method == 'MOLASSES'):
        solver = MOLASSESDriver(heightmap_path,dim,hm_elev_min_m,hm_elev_max_m,n_grid)
    elif(simulation_method == 'MAGFLOW'):
        solver = MAGFLOWDriver(heightmap_path,dim,hm_elev_min_m,hm_elev_max_m,n_grid)

    heightmap = solver.Heightmap
    grid = solver.Grid
    
    res = (1920, 1080)
    window = ti.ui.Window(f'Real {simulation_method} 3D', res, vsync=False)

    canvas = window.get_canvas()
    canvas.set_background_color((0.16796875,0.17578125,0.2578125))
    gui = window.GUI
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(0.0, heightmap.hm_elev_range_km, 0.0)
    camera.lookat(heightmap.hm_width_px*heightmap.px_to_km/2.0, 0.0, heightmap.hm_height_px*heightmap.px_to_km/2.0)
    camera.fov(55)
    substeps = 5
    simulation_time = 0.0
    while window.running:
        if(simulation_method == 'MAGFLOW'):
            mouse = window.get_cursor_pos()
            # print(dir(ti.ui))
            # print(f'window.is_pressed(ti.ui.LMB): {window.is_pressed(ti.ui.LMB)}')
            # print(f'window.is_pressed(ti.ui.MMB): {window.is_pressed(ti.ui.MMB)}')
            # print(f'window.is_pressed(ti.ui.RMB): {window.is_pressed(ti.ui.RMB)}')
            if window.is_pressed(ti.ui.CTRL):
                rayPoint, rayDirection = pixelToRay(camera, mouse[0], mouse[1], 1, 1, window.get_window_shape())
                # print(f'rayPoint: {rayPoint} rayDirection: {rayDirection}')
                validAnchor,ti_vector_pos = solver.Grid.Intersect(rayPoint,rayDirection)
                # if(validAnchor):
                #     print('Yes')
                # update_particle_pos(anchor_x,anchor_y)
                particles_pos[0] = ti_vector_pos
                # print(f'validAnchor: {validAnchor} , ti_vector_pos/grid.grid_size_to_km: {int(ti_vector_pos[0]/grid.grid_size_to_km)},{int(ti_vector_pos[2]/grid.grid_size_to_km)}')
                solver.Grid.calculate_m_transforms_lvl2(int(ti_vector_pos[0]/grid.grid_size_to_km),int(ti_vector_pos[2]/grid.grid_size_to_km))
            # ini_pulse_time = time.time()
            else:
                particles_pos[0] = vector_outside
                solver.Grid.calculate_m_transforms_lvl2(int(9999),int(9999))
        if(pulse_state == 1):
            if(simulation_method == 'MOLASSES'):
                solver.pulse()
            elif(simulation_method == 'MAGFLOW'):
                solver.set_active_pulses()
                # 1. Compute volumetrix lava flux for cell vents
                solver.Grid.pulse()
                # print(f'[PULSE] {time.time()-ini_pulse_time}')
            solver.Grid.calculate_m_transforms_lvl1()
        if(run_state == 1 or run_state == 2):
            if(simulation_method == 'MOLASSES'):
                solver.Grid.distribute()
            elif(simulation_method == 'MAGFLOW'):
                for i in range(substeps):
                    # 2. Compute flux transfer with neighbouring cells
                    # ini_flux_time = time.time()
                    solver.Grid.computeFluxTransfers()
                    # print(f'[FLUX] {time.time()-ini_flux_time}')
                    # 3. Computer the maximum allowed time-step
                    # ini_dt_time = time.time()
                    solver.Grid.computeTimeSteps()
                    # print(f'[TIMESTEP] {time.time()-ini_dt_time}')
                    # ini_global_time = time.time()
                    global_delta_time = solver.Grid.computeGlobalTimeStep()
                    solver.Grid.global_delta_time = global_delta_time
                    simulation_time += global_delta_time
                    # print(f'[GLOBAL] {time.time()-ini_global_time}')
                    # print(f'global_delta_time: {solver.Grid.global_delta_time}')
                    # 4. Update state of the cell
                    # 4.1 Compute the new lava thickness
                    # ini_lavah_time = time.time()
                    solver.Grid.computeNewLavaThickness()
                    # print(f'[NEWLAVAH] {time.time()-ini_lavah_time}')
                    # solver.Grid.updateLavaThickness()
                    # 4.2 Compute the heat radiation loss
                    solver.Grid.computeHeatRadiationLoss()
                    solver.Grid.updateTemperature()
                    # 4.3 Transfer an appropriate amount of lava thickness to the solid lava thickness if there is solidification
                    # solver.Grid.computeLavaSolidification()
            solver.Grid.calculate_m_transforms_lvl1()
            if(run_state == 2):
                run_state = 0
        render(camera,window,scene,canvas,heightmap,grid)
        solver.active_flow.pulsevolume = show_options(gui,solver.active_flow.pulsevolume,solver.Grid,simulation_time)
        # print(f'[RUNSIMULATION] solver.active_flow.pulsevolume: {solver.active_flow.pulsevolume}')
        window.show()

if __name__ == '__main__':
    main()