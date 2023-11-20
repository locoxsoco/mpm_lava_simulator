class GGUI:
    
    def __init__(self):
        self.normal_line_column = 0
        self.debug_normals_checkbox = 0
        self.debug_grid_dem_checkbox = 1
        self.debug_grid_lava_checkbox = 0
        self.debug_grid_lava_heatmap_checkbox = 1
        self.debug_grid_lava_type = GridLava.HEATMAP
        self.debug_mesh_checkbox = 0
        self.dem_checkbox = 1
        self.lava_checkbox = 0
        self.heat_checkbox = 0
        self.brush_strength = 5
        self.brush_type = Brush.DEM
        self.run_state = 0
        self.global_delta_time = 10.0
        self.init_sim_time = 0.0
    

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
        global mouse_selection_radius
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
            substeps = w.slider_float("Simulation speed", substeps/5.0, 1.0, 4.0)
            grid.quality_tolerance[None] = w.slider_float("Slope Quality tolerance", grid.quality_tolerance[None], 1.0, 10.0)
            w.text(f'Volume Erupted: {round(grid.global_volume_lava_erupted_m3[None],2)} m3')
            if w.button("Run"):
                run_state = 1
            if w.button("Pause"):
                run_state = 0
            if w.button("Step"):
                run_state = 2
        
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
            mouse_selection_radius = w.slider_float("Size (km)", mouse_selection_radius, grid.grid_size_km_to_scaled_grid_size_km*2, grid.grid_size_km_to_scaled_grid_size_km*20)

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
        
        return int(substeps*5.0)
    
    def render(camera,window,scene,canvas,heightmap,grid,simulation_method):
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)

        if(debug_mesh_checkbox):
            scene.mesh(vertices=heightmap.heightmap_positions, indices=heightmap.heightmap_indices,per_vertex_color=heightmap.heightmap_colors)
        if(debug_grid_dem_checkbox):
            scene.mesh_instance(vertices=grid.cube_positions_dem, indices=grid.cube_indices,color=(100.0/256.0, 80.0/256.0, 80.0/256.0), transforms=grid.m_transforms_dem)
        if(debug_grid_lava_checkbox):
            scene.mesh_instance(vertices=grid.cube_positions_lava, indices=grid.cube_indices,color=(256.0/256.0, 0.0/256.0, 0.0/256.0), transforms=grid.m_transforms_lava)
        elif(debug_grid_lava_heatmap_checkbox):
            scene.mesh_instance(vertices=grid.cube_positions_lava_heatmap, indices=grid.cube_indices_lava_heatmap,per_vertex_color=grid.cube_colors_lava_heatmap)
        scene.particles(mouse_selection_pos, per_vertex_color = mouse_selection_color_ti, radius = mouse_selection_radius*grid.grid_size_km_to_scaled_grid_size_km/2.0)
        if(debug_normals_checkbox):
            for i in range(45):
                scene.lines(heightmap.verts, color = (0.28, 0.68, 0.99), width = 0.5, vertex_count = 2, vertex_offset = 4*(normal_line_column*(heightmap.hm_width_px)+i+75))

        scene.ambient_light((0.5, 0.5, 0.5))
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

        canvas.scene(scene)