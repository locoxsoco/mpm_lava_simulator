import os
import argparse
import taichi as ti
import numpy as np
import time
from config_builder import SimConfig
from MOLASSES.driver import Driver as MOLASSESDriver
from MAGFLOW.driver import Driver as MAGFLOWDriver, PulseFileStatus
from utils import *
from enum import Enum

ti.init(arch=ti.gpu)
particles_pos = ti.Vector.field(3, dtype=ti.f32, shape = 1)
vector_outside = ti.Vector([-9999, -9999, -9999])
is_particles_outside = False
particle_radius = 0.5
particle_color_ti = ti.Vector.field(4, float, 1)
particle_color_green = ti.Vector([0.0/256.0, 256.0/256.0, 0.0/256.0, 0.2])
particle_color_red = ti.Vector([256.0/256.0, 0.0/256.0, 0.0/256.0, 0.2])
particle_color_ti[0] = particle_color_green

######################### Debug Parameters #########################
class Brush(Enum):
    DEM = 0
    LAVA = 1
    HEAT = 2
    COOL = 3

class GridLava(Enum):
    LAVA = 0
    HEATMAP = 1


normal_line_column = 0
debug_normals_checkbox = 0
debug_grid_dem_checkbox = 1
debug_grid_lava_checkbox = 1
debug_grid_lava_heatmap_checkbox = 0
debug_grid_lava_type = GridLava.HEATMAP
debug_mesh_checkbox = 0
dem_checkbox = 1
lava_checkbox = 0
heat_checkbox = 0
brush_strength = 5
brush_type = Brush.DEM
run_state = 0
global_delta_time = 10.0
init_sim_time = 0.0
#######################################################################

def show_options(gui,substeps,grid,simulation_time,simulation_method):
    global normal_line_column
    global debug_normals_checkbox
    global debug_grid_dem_checkbox
    global debug_grid_lava_checkbox
    global debug_grid_lava_heatmap_checkbox
    global debug_mesh_checkbox
    global dem_checkbox
    global lava_checkbox
    global heat_checkbox
    global cool_checkbox
    global run_state
    global particle_radius
    global brush_strength
    global brush_type
    global debug_grid_lava_type
    global global_delta_time

    with gui.sub_window("Visualization", 0.0, 0.80, 0.14, 0.13) as w:
        # debug_normals_checkbox = w.checkbox("Show normals", debug_normals_checkbox)
        # normal_line_column = w.slider_int("Column", normal_line_column, 0, 100)
        debug_grid_dem_checkbox = w.checkbox("Show grid dem", debug_grid_dem_checkbox)
        debug_grid_lava_checkbox = w.checkbox("Show grid lava", debug_grid_lava_checkbox)
        debug_grid_lava_heatmap_checkbox = w.checkbox("Show grid lava heatmap", debug_grid_lava_heatmap_checkbox)
        # debug_mesh_checkbox = w.checkbox("Show mesh", debug_mesh_checkbox)
        if(simulation_method=='MAGFLOW'):
            grid.rendering_lava_height_minimum_m[None] = w.slider_float("Rendering min lava height (m)", grid.rendering_lava_height_minimum_m[None], 0.0, 0.5)

        if(debug_grid_lava_checkbox and debug_grid_lava_type != GridLava.LAVA):
            debug_grid_lava_type = GridLava.LAVA
            debug_grid_lava_heatmap_checkbox = 0
        elif(debug_grid_lava_heatmap_checkbox and debug_grid_lava_type != GridLava.HEATMAP):
            debug_grid_lava_type = GridLava.HEATMAP
            debug_grid_lava_checkbox = 0
    
    run_state_text = 'Running' if run_state else 'Paused'
    with gui.sub_window(f'Simulation status: {run_state_text}', 0.0, 0.0, 0.27, 0.20) as w:
        w.text(f'Simulation time: {round(simulation_time,3)} s')
        if(simulation_method=='MAGFLOW'):
            substeps = w.slider_float("Simulation speed", substeps/5.0, 1.0, 4.0)
            grid.quality_tolerance[None] = w.slider_float("Slope Quality tolerance", grid.quality_tolerance[None], 1.0, 10.0)
            w.text(f'Volume Erupted: {round(grid.global_volume_lava_erupted_m3[None],2)} m3')
        elif(simulation_method=='MOLASSES'):
            grid.c_factor[None] = w.slider_float("Distribution factor", grid.c_factor[None], 0.0, 1.0)
            global_delta_time = w.slider_float("Time step (s)", global_delta_time, 1.0, 10.0)
        if w.button("Run"):
            run_state = 1
        if w.button("Pause"):
            run_state = 0
        if w.button("Step"):
            run_state = 2
    
    if(simulation_method == 'MAGFLOW'):
        with gui.sub_window(f'Lava Properties', 0.0, 0.20, 0.27, 0.185) as w:
            grid.lava_density[None] = w.slider_float("Lava density (kg/m3)", grid.lava_density[None], 20.0, 10000.0)
            grid.specific_heat_capacity[None] = w.slider_float("Heat Capacity (J/Kg K)", grid.specific_heat_capacity[None], 5.0, 1600.0)
            grid.H2O[None] = w.slider_float("Water content (wt%)", grid.H2O[None], 0.00, 50.0)
            grid.solidification_temperature[None] = w.slider_float("Solidification T. (K)", grid.solidification_temperature[None], 400.0, 4500.0)
            grid.extrusion_temperature[None] = w.slider_float("Extrusion T. (K)", grid.extrusion_temperature[None], 400.0, 2000.0)
            grid.emissivity[None] = w.slider_float("Lava emissivity factor", grid.emissivity[None], 0.0, 0.02)
            grid.cooling_accelerator_factor[None] = w.slider_float("Lava cooling factor", grid.cooling_accelerator_factor[None], 0.0, 10.0)
            

        # customPulseVolume = 0.0
        with gui.sub_window(f'Brush (Ctrl to add, Ctrl+Shift to remove)', 0.0, 0.385, 0.165, 0.145) as w:
            dem_checkbox = w.checkbox("DEM", dem_checkbox)
            lava_checkbox = w.checkbox("Lava", lava_checkbox)
            heat_checkbox = w.checkbox("Heat", heat_checkbox)
            brush_strength = w.slider_float("Strength", brush_strength, 1.0, 10.0)
            particle_radius = w.slider_float("Size (km)", particle_radius, grid.grid_size_km_to_scaled_grid_size_km*2, grid.grid_size_km_to_scaled_grid_size_km*20)

            if(dem_checkbox and brush_type != Brush.DEM):
                brush_type = Brush.DEM
                lava_checkbox = 0
                heat_checkbox = 0
            elif(lava_checkbox and brush_type != Brush.LAVA):
                brush_type = Brush.LAVA
                dem_checkbox = 0
                heat_checkbox = 0
            elif(heat_checkbox and brush_type != Brush.HEAT):
                brush_type = Brush.HEAT
                dem_checkbox = 0
                lava_checkbox = 0
            # customPulseVolume = w.slider_float("Volume (m3)", customPulseVolume, 0.0, 1.0)
    
    if(simulation_method == 'MAGFLOW'):
        return int(substeps*5.0)


def render(camera,window,scene,canvas,heightmap,grid,simulation_method):
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)

    if(debug_mesh_checkbox):
        scene.mesh(vertices=heightmap.heightmap_positions, indices=heightmap.heightmap_indices,per_vertex_color=heightmap.heightmap_colors)
    if(debug_grid_dem_checkbox):
        scene.mesh_instance(vertices=grid.cube_positions_dem, indices=grid.cube_indices,color=(100.0/256.0, 80.0/256.0, 80.0/256.0), transforms=grid.m_transforms_dem)
    if(debug_grid_lava_checkbox):
        scene.mesh_instance(vertices=grid.cube_positions_lava1, indices=grid.cube_indices,color=(256.0/256.0, 0.0/256.0, 0.0/256.0), transforms=grid.m_transforms_lava)
    elif(debug_grid_lava_heatmap_checkbox and simulation_method == 'MAGFLOW'):
        scene.mesh_instance(vertices=grid.cube_positions_lava, indices=grid.cube_indices_lava,per_vertex_color=grid.cube_colors_lava)
    if(simulation_method == 'MAGFLOW'):
        scene.particles(particles_pos, per_vertex_color = particle_color_ti, radius = particle_radius*grid.grid_size_km_to_scaled_grid_size_km/2.0)
    if(debug_normals_checkbox):
        for i in range(45):
            scene.lines(heightmap.verts, color = (0.28, 0.68, 0.99), width = 0.5, vertex_count = 2, vertex_offset = 4*(normal_line_column*(heightmap.hm_width_px)+i+75))

    scene.ambient_light((0.5, 0.5, 0.5))
    if(simulation_method == 'MAGFLOW'):
        scene.point_light(pos=(heightmap.hm_width_px*heightmap.px_to_km/2.0, 3.0*heightmap.hm_elev_range_km/2.0, heightmap.hm_height_px*heightmap.px_to_km/2.0), color=(0.5, 0.5, 0.5))
        scene.point_light(pos=(heightmap.hm_width_px*heightmap.px_to_km/2.0, 3.0*heightmap.hm_elev_range_km/2.0, 3.0*heightmap.hm_height_px*heightmap.px_to_km/2.0), color=(0.5, 0.5, 0.5))
        # scene.point_light(pos=(
        #     0.0,
        #     (heightmap.hm_elev_range_km+grid.underground_m[None]/1000.0)*grid.grid_size_km_to_scaled_grid_size_km*3.0,
        #     0.0
        # ), color=(0.7, 0.7, 0.7))
        # scene.point_light(pos=(
        #     grid.scaled_grid_size_km*grid.n_grid,
        #     (heightmap.hm_elev_range_km+grid.underground_m[None]/1000.0)*grid.grid_size_km_to_scaled_grid_size_km*3.0,
        #     0.0
        # ), color=(0.7, 0.7, 0.7))
        # scene.point_light(pos=(
        #     0.0,
        #     (heightmap.hm_elev_range_km+grid.underground_m[None]/1000.0)*grid.grid_size_km_to_scaled_grid_size_km*3.0,
        #     grid.scaled_grid_size_km*grid.n_grid
        # ), color=(0.7, 0.7, 0.7))
        # scene.point_light(pos=(
        #     grid.scaled_grid_size_km*grid.n_grid,
        #     (heightmap.hm_elev_range_km+grid.underground_m[None]/1000.0)*grid.grid_size_km_to_scaled_grid_size_km*3.0,
        #     grid.scaled_grid_size_km*grid.n_grid
        # ), color=(0.7, 0.7, 0.7))
    elif(simulation_method=='MOLASSES'):
        scene.point_light(pos=(heightmap.hm_width_px*heightmap.px_to_km/2.0, 3.0*heightmap.hm_elev_range_km/2.0, heightmap.hm_height_px*heightmap.px_to_km/2.0), color=(0.5, 0.5, 0.5))
        scene.point_light(pos=(heightmap.hm_width_px*heightmap.px_to_km/2.0, 3.0*heightmap.hm_elev_range_km/2.0, 3.0*heightmap.hm_height_px*heightmap.px_to_km/2.0), color=(0.5, 0.5, 0.5))

    canvas.scene(scene)

def main():
    global run_state
    global is_particles_outside
    global particle_color_ti
    global global_delta_time
    global brush_strength
    global debug_grid_dem_checkbox
    global debug_grid_lava_checkbox
    global debug_grid_lava_heatmap_checkbox
    
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
    if(simulation_method == 'MAGFLOW'):
        global_delta_time = grid.global_delta_time
    
    res = (1920, 1080)
    window = ti.ui.Window(f'Real {simulation_method} 3D', res, vsync=False)

    canvas = window.get_canvas()
    canvas.set_background_color((0.16796875,0.17578125,0.2578125))
    gui = window.GUI
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    if(simulation_method == 'MOLASSES'):
        # camera.position(0.0, heightmap.hm_elev_range_km, 0.0)
        # camera.lookat(grid.n_grid/2.0, heightmap.hm_elev_range_km, grid.n_grid/2.0)
        # camera.position(1.38376481, 0.89988685, 1.80057741)
        # camera.lookat(2.12759371, 0.24434504, 1.93089818)
        # # Top view
        # camera.position(2.0, 2.5, 2.0)
        # camera.lookat(2.0001, 0.00001, 2.0)
        # Side view
        camera.position(0.8, 1.35, 2.5)
        camera.lookat(2.0, 0.0, 2.0)
    elif(simulation_method == 'MAGFLOW'):
        # camera.position(0.0, heightmap.hm_elev_range_km*grid.grid_size_km_to_scaled_grid_size_km, 0.0)
        # camera.lookat(grid.scaled_grid_size_km*grid.n_grid/2.0, heightmap.hm_elev_range_km*grid.grid_size_km_to_scaled_grid_size_km, grid.scaled_grid_size_km*grid.n_grid/2.0)
        # camera.position(1.38376481, 0.89988685, 1.80057741)
        # camera.lookat(2.12759371, 0.24434504, 1.93089818)
        # # Top view
        # camera.position(2.0, 2.5, 2.0)
        # camera.lookat(2.0001, 0.00001, 2.0)
        # Side view
        camera.position(grid.scaled_grid_size_km*200, heightmap.hm_elev_range_km*grid.grid_size_km_to_scaled_grid_size_km*7.5, grid.scaled_grid_size_km*200)
        camera.lookat(grid.scaled_grid_size_km*200 +0.0001, 0.00001, grid.scaled_grid_size_km*200)
    camera.fov(55)
    substeps = 20
    simulation_time = 0.0
    prev_solver_index_file = 0
    while window.running:
        if(simulation_method == 'MAGFLOW'):
            mouse = window.get_cursor_pos()
            # print(dir(ti.ui))
            # print(f'window.is_pressed(ti.ui.LMB): {window.is_pressed(ti.ui.LMB)}')
            # print(f'window.is_pressed(ti.ui.MMB): {window.is_pressed(ti.ui.MMB)}')
            # print(f'window.is_pressed(ti.ui.RMB): {window.is_pressed(ti.ui.RMB)}')
            if window.is_pressed(ti.ui.CTRL):
                if window.is_pressed(ti.ui.SPACE) and run_state != 1:
                    run_state = 1
                    init_sim_time = time.time()
                rayPoint, rayDirection = pixelToRay(camera, mouse[0], mouse[1], 1, 1, window.get_window_shape())
                # print(f'rayPoint: {rayPoint} rayDirection: {rayDirection}')
                validAnchor,ti_vector_pos = solver.Grid.Intersect(rayPoint,rayDirection)
                if(window.is_pressed(ti.ui.LMB) and validAnchor):
                    ti_vector_pos_grid = ti_vector_pos/grid.scaled_grid_size_km
                    # print(f'i,k: {int(ti_vector_pos_grid[0])},{int(ti_vector_pos_grid[2])} lava thickness: {solver.Grid.lava_thickness[int(ti_vector_pos_grid[0]),int(ti_vector_pos_grid[2])]}')
                    # print(f'i,k: {200},{200} lava thickness: {solver.Grid.lava_thickness[200,200]}')
                    # solver.Grid.calculate_m_transforms_lvl2(int(ti_vector_pos_grid[0]),int(ti_vector_pos_grid[2]))
                    if (brush_type == Brush.DEM):
                        if window.is_pressed(ti.ui.SHIFT):
                            solver.remove_dem(int(ti_vector_pos_grid[0]),int(ti_vector_pos_grid[2]),particle_radius,brush_strength)
                        else:
                            solver.add_dem(int(ti_vector_pos_grid[0]),int(ti_vector_pos_grid[2]),particle_radius,brush_strength)
                        if(debug_grid_dem_checkbox):
                            solver.Grid.calculate_m_transforms_dem()
                    elif (brush_type == Brush.LAVA):
                        if window.is_pressed(ti.ui.SHIFT):
                            solver.set_active_pulses(int(ti_vector_pos_grid[0]),int(ti_vector_pos_grid[2]),particle_radius,-substeps,brush_strength)
                        else:
                            solver.set_active_pulses(int(ti_vector_pos_grid[0]),int(ti_vector_pos_grid[2]),particle_radius,substeps,brush_strength)
                    elif (brush_type == Brush.HEAT):
                        if window.is_pressed(ti.ui.SHIFT):
                            solver.remove_heat(int(ti_vector_pos_grid[0]),int(ti_vector_pos_grid[2]),particle_radius,brush_strength)
                        else:
                            solver.add_heat(int(ti_vector_pos_grid[0]),int(ti_vector_pos_grid[2]),particle_radius,brush_strength)
                # if(validAnchor):
                # update_particle_pos(anchor_x,anchor_y)
                particles_pos[0] = ti_vector_pos
                is_particles_outside = False
                if window.is_pressed(ti.ui.SHIFT):
                    particle_color_ti[0] = particle_color_red
                else:
                    particle_color_ti[0] = particle_color_green
                # print(f'validAnchor: {validAnchor} , ti_vector_pos/grid.grid_size_to_km: {int(ti_vector_pos[0]/grid.grid_size_to_km)},{int(ti_vector_pos[2]/grid.grid_size_to_km)}')
            # ini_pulse_time = time.time()
            else:
                if not is_particles_outside:
                    particles_pos[0] = vector_outside
                    # solver.Grid.calculate_m_transforms_lvl2(int(9999),int(9999))
                    is_particles_outside = True
        elif(simulation_method=='MOLASSES'):
            if window.is_pressed(ti.ui.CTRL):
                if window.is_pressed(ti.ui.SPACE) and run_state != 1:
                    run_state = 1
                    init_sim_time = time.time()
        # solver.Grid.updateTemperature()
        # print(f'[PULSE] {time.time()-ini_pulse_time}')
        if(debug_grid_lava_checkbox):
            solver.Grid.calculate_m_transforms_lava()
        elif(debug_grid_lava_heatmap_checkbox):
            if(simulation_method == 'MAGFLOW'):
                solver.Grid.calculate_lava_height_and_color()
        if(simulation_method == 'MOLASSES'):
            if(run_state == 1 or run_state == 2):
                solver.pulse(global_delta_time)
                solver.Grid.distribute()
                solver.Grid.updateEffElev()
                simulation_time += global_delta_time
                if(run_state==2):
                    run_state = 0
        elif(simulation_method == 'MAGFLOW'):
            if(run_state == 1 or run_state == 2):
                solver.set_active_pulses_file(simulation_time,substeps)
                for _ in range(substeps):
                    # 1. Compute volumetrix lava flux for cell vents
                    solver.Grid.pulse()
                    simulation_time += solver.Grid.global_delta_time[None]
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
                    # global_delta_time = solver.Grid.computeGlobalTimeStep()
                    # print(f'[Driver] global_delta_time: {global_delta_time} index_global: {index_global_x},{index_global_y}')
                    solver.Grid.global_delta_time[None] = global_delta_time                    
                    # print(f'[GLOBAL] {time.time()-ini_global_time}')
                    # print(f'[Driver] global_delta_time: {solver.Grid.global_delta_time} volumeErupted: {grid.global_volume_lava_erupted_m3}')
                    # 4. Update state of the cell
                    # 4.1 Compute the new lava thickness
                    # ini_lavah_time = time.time()
                    solver.Grid.computeNewLavaThickness()
                    # print(f'[NEWLAVAH] {time.time()-ini_lavah_time}')
                    # solver.Grid.updateLavaThickness()
                    # 4.2 Compute the heat radiation loss
                    # heat_loss_time = time.time()
                    solver.Grid.computeHeatRadiationLoss()
                    solver.Grid.updateTemperature()
                    # print(f'[HEATLOSS] {time.time()-heat_loss_time}')
                    # 4.3 Transfer an appropriate amount of lava thickness to the solid lava thickness if there is solidification
                    # solver.Grid.computeLavaSolidification(global_delta_time)
                    # solver.Grid.updateTemperature()
                    # solid_lava_time = time.time()
                    solver.Grid.computeLavaSolidification()
                    # print(f'[SOLIDLAVA] {time.time()-solid_lava_time}')
                    # solver.Grid.updateTemperature()
            if(run_state == 1 or run_state == 2):
                if(debug_grid_lava_checkbox):
                    solver.Grid.calculate_m_transforms_lava()
                    solver.Grid.calculate_lava_height_and_color()
                elif(debug_grid_lava_heatmap_checkbox):
                    solver.Grid.calculate_lava_height_and_color()
                if(debug_grid_dem_checkbox):
                    solver.Grid.calculate_m_transforms_dem()
            if(run_state == 2):
                run_state = 0
        render(camera,window,scene,canvas,heightmap,grid,simulation_method)
        if(simulation_method == 'MAGFLOW'):
            substeps = show_options(gui,substeps,solver.Grid,simulation_time,simulation_method)
        elif(simulation_method == 'MOLASSES'):
            show_options(gui,substeps,solver.Grid,simulation_time,simulation_method)
        # print(f'solver.Grid: {solver.Grid.lava_density[None]}')
        # print(f'[RUNSIMULATION] solver.active_flow.pulsevolume: {solver.active_flow.pulsevolume}')
        # if(simulation_method == 'MAGFLOW'):
        #     if(solver.pulse_file_status == PulseFileStatus.END and run_state != 0):
        #         run_state = 0
        #         print(f'[SIMULATION] Simulation time: {round(time.time()-init_sim_time,2)} s')
        #     if(prev_solver_index_file != solver.pulse_file_index):
        #         solver.Grid.calculate_m_transforms_dem()
        #         solver.Grid.calculate_m_transforms_lava()
        #         solver.Grid.calculate_lava_height_and_color()
        #         # Top view
        #         camera.position(grid.scaled_grid_size_km*200, heightmap.hm_elev_range_km*grid.grid_size_km_to_scaled_grid_size_km*7.5, grid.scaled_grid_size_km*200)
        #         camera.lookat(grid.scaled_grid_size_km*200 +0.0001, 0.00001, grid.scaled_grid_size_km*200)
        #         debug_grid_dem_checkbox = 1
        #         debug_grid_lava_checkbox = 1
        #         debug_grid_lava_heatmap_checkbox = 0
        #         render(camera,window,scene,canvas,heightmap,grid,simulation_method)
        #         window.save_image(f'./results/v2/cooling_factor/8/frame_top_view_{solver.pulse_file_index}.png')
        #         window.show()
        #         debug_grid_lava_checkbox = 0
        #         debug_grid_lava_heatmap_checkbox = 1
        #         render(camera,window,scene,canvas,heightmap,grid,simulation_method)
        #         # substeps = show_options(gui,substeps,solver.Grid,simulation_time,simulation_method)
        #         window.save_image(f'./results/v2/cooling_factor/8/frame_top_view_heatmap_{solver.pulse_file_index}.png')
        #         window.show()
        #         # Side view
        #         camera.position(0.8, 1.35, 2.5)
        #         camera.lookat(2.0, 0.0, 2.0)
        #         debug_grid_dem_checkbox = 1
        #         debug_grid_lava_checkbox = 1
        #         debug_grid_lava_heatmap_checkbox = 0
        #         render(camera,window,scene,canvas,heightmap,grid,simulation_method)
        #         window.save_image(f'./results/v2/cooling_factor/8/frame_side_view_{solver.pulse_file_index}.png')
        #         window.show()
        #         debug_grid_lava_checkbox = 0
        #         debug_grid_lava_heatmap_checkbox = 1
        #         render(camera,window,scene,canvas,heightmap,grid,simulation_method)
        #         # substeps = show_options(gui,substeps,solver.Grid,simulation_time,simulation_method)
        #         window.save_image(f'./results/v2/cooling_factor/8/frame_side_view_heatmap_{solver.pulse_file_index}.png')
        #         window.show()
        #         debug_grid_dem_checkbox = 0
        #         debug_grid_lava_checkbox = 0
        #         debug_grid_lava_heatmap_checkbox = 0
        #         prev_solver_index_file = solver.pulse_file_index
        # elif(simulation_method == 'MOLASSES'):
        #     if(simulation_time==600.0 or simulation_time==1200.0 or simulation_time==1800.0 or simulation_time==2400.0 or
        #        simulation_time==3000.0 or simulation_time==3600.0 or simulation_time==4200.0 or simulation_time==4800.0 or
        #        simulation_time==5400.0 or simulation_time==6000.0 or simulation_time==6600.0 or simulation_time==7200.0) and run_state!=0:
        #         solver.Grid.calculate_m_transforms_dem()
        #         solver.Grid.calculate_m_transforms_lava()
        #         # Top view
        #         camera.position(2.0, 2.5, 2.0)
        #         camera.lookat(2.0001, 0.00001, 2.0)
        #         debug_grid_dem_checkbox = 1
        #         debug_grid_lava_checkbox = 1
        #         debug_grid_lava_heatmap_checkbox = 0
        #         render(camera,window,scene,canvas,heightmap,grid,simulation_method)
        #         window.save_image(f'./results/modelcomparison/aa/molasses/frame_top_view_{simulation_time}.png')
        #         window.show()
        #         debug_grid_lava_checkbox = 0
        #         debug_grid_lava_heatmap_checkbox = 1
        #         render(camera,window,scene,canvas,heightmap,grid,simulation_method)
        #         # substeps = show_options(gui,substeps,solver.Grid,simulation_time,simulation_method)
        #         window.save_image(f'./results/modelcomparison/aa/molasses/frame_top_view_heatmap_{simulation_time}.png')
        #         window.show()
        #         # Side view
        #         camera.position(0.8, 1.35, 2.5)
        #         camera.lookat(2.0, 0.0, 2.0)
        #         debug_grid_dem_checkbox = 1
        #         debug_grid_lava_checkbox = 1
        #         debug_grid_lava_heatmap_checkbox = 0
        #         render(camera,window,scene,canvas,heightmap,grid,simulation_method)
        #         window.save_image(f'./results/modelcomparison/aa/molasses/frame_side_view_{simulation_time}.png')
        #         window.show()
        #         debug_grid_lava_checkbox = 0
        #         debug_grid_lava_heatmap_checkbox = 1
        #         render(camera,window,scene,canvas,heightmap,grid,simulation_method)
        #         # substeps = show_options(gui,substeps,solver.Grid,simulation_time,simulation_method)
        #         window.save_image(f'./results/modelcomparison/aa/molasses/frame_side_view_heatmap_{simulation_time}.png')
        #         window.show()
        #         debug_grid_dem_checkbox = 0
        #         debug_grid_lava_checkbox = 0
        #         debug_grid_lava_heatmap_checkbox = 0
        #     if(simulation_time >= 7200.0 and run_state != 0):
        #         run_state = 0
        #         print(f'[SIMULATION] Simulation time: {round(time.time()-init_sim_time,2)} s')
        window.show()

if __name__ == '__main__':
    main()