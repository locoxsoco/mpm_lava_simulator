import os
import argparse
import taichi as ti
import numpy as np
from config_builder import SimConfig
from MOLASSES.driver import Driver

ti.init(arch=ti.gpu)

######################### Debug Parameters #########################
normal_line_column = 0
debug_normals_checkbox = 0
debug_grid_checkbox = 1
debug_mesh_checkbox = 0
run_state = 1
#######################################################################


def show_options(gui):
    global normal_line_column
    global debug_normals_checkbox
    global debug_grid_checkbox
    global debug_mesh_checkbox
    global run_state

    with gui.sub_window("Debug", 0.05, 0.1, 0.2, 0.15) as w:
        debug_normals_checkbox = w.checkbox("Show normals", debug_normals_checkbox)
        normal_line_column = w.slider_int("Column", normal_line_column, 0, 100)
        debug_grid_checkbox = w.checkbox("Show grid", debug_grid_checkbox)
        debug_mesh_checkbox = w.checkbox("Show mesh", debug_mesh_checkbox)
        if w.button("Run"):
            run_state = 1
        if w.button("Pause"):
            run_state = 0
        if w.button("Step"):
            run_state = 2

def render(camera,window,scene,canvas,heightmap,grid):
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)

    if(debug_mesh_checkbox):
        scene.mesh(vertices=heightmap.heightmap_positions, indices=heightmap.heightmap_indices,per_vertex_color=heightmap.heightmap_colors,normals=heightmap.heightmap_normals)

    if(debug_grid_checkbox):
        scene.mesh_instance(vertices=grid.cube_positions, indices=grid.cube_indices,per_vertex_color=grid.cube_colors_lvl0, transforms=grid.m_transforms_lvl0)
        scene.mesh_instance(vertices=grid.cube_positions2, indices=grid.cube_indices,per_vertex_color=grid.cube_colors_lvl1, transforms=grid.m_transforms_lvl1)
    if(debug_normals_checkbox):
        for i in range(45):
            scene.lines(heightmap.verts, color = (0.28, 0.68, 0.99), width = 0.5, vertex_count = 2, vertex_offset = 4*(normal_line_column*(heightmap.hm_width_px)+i+75))

    scene.ambient_light((0.3, 0.3, 0.3))
    scene.point_light(pos=(heightmap.hm_width_px*heightmap.px_to_km/2.0, 3.0*heightmap.hm_elev_range_km/2.0, heightmap.hm_height_px*heightmap.px_to_km/2.0), color=(0.5, 0.5, 0.5))
    scene.point_light(pos=(heightmap.hm_width_px*heightmap.px_to_km/2.0, 3.0*heightmap.hm_elev_range_km/2.0, 3.0*heightmap.hm_height_px*heightmap.px_to_km/2.0), color=(0.5, 0.5, 0.5))

    canvas.scene(scene)

def main():
    global run_state
    parser = argparse.ArgumentParser(description='MOLASSES Taichi')
    parser.add_argument('--scene_file',
                        default='',
                        help='scene file')
    args = parser.parse_args()
    scene_path = args.scene_file

    config = SimConfig(scene_file_path=scene_path)
    heightmap_path = config.get_cfg("heightmapFile")
    dim = config.get_cfg("dim")
    n_grid = config.get_cfg("nGrid")
    hm_elev_min_m = config.get_cfg("elevMinMeters")
    hm_elev_max_m = config.get_cfg("elevMaxMeters")

    solver = Driver(heightmap_path,dim,hm_elev_min_m,hm_elev_max_m,n_grid)

    heightmap = solver.Heightmap
    grid = solver.Grid
    
    res = (1080, 720)
    window = ti.ui.Window("Real MOLASSES 3D", res, vsync=False)

    canvas = window.get_canvas()
    canvas.set_background_color((0.16796875,0.17578125,0.2578125))
    gui = window.GUI
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(0.0, heightmap.hm_elev_range_km, 0.0)
    camera.lookat(heightmap.hm_width_px*heightmap.px_to_km/2.0, 0.0, heightmap.hm_height_px*heightmap.px_to_km/2.0)
    camera.fov(55)
    while window.running:
        if(run_state == 1 or run_state == 2):
            solver.step(render,camera,window,scene,canvas,show_options,gui)
            # solver.Grid.calculate_m_transforms_lvl1()
            if(run_state == 2):
                run_state = 0
        # render(camera,window,scene,canvas,heightmap,grid)
        # show_options(gui)
        # window.show()

if __name__ == '__main__':
    main()