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

        if(debug_grid_lava_checkbox and debug_grid_lava_type != GridLava.LAVA):
            debug_grid_lava_type = GridLava.LAVA
            debug_grid_lava_heatmap_checkbox = 0
        elif(debug_grid_lava_heatmap_checkbox and debug_grid_lava_type != GridLava.HEATMAP):
            debug_grid_lava_type = GridLava.HEATMAP
            debug_grid_lava_checkbox = 0
    
    run_state_text = 'Running' if run_state else 'Paused'
    with gui.sub_window(f'Simulation status: {run_state_text}', 0.0, 0.0, 0.27, 0.20) as w:
        w.text(f'Simulation time: {round(simulation_time,3)} s')
        grid.c_factor[None] = w.slider_float("Distribution factor", grid.c_factor[None], 0.0, 1.0)
        global_delta_time = w.slider_float("Time step (s)", global_delta_time, 1.0, 10.0)
        if w.button("Run"):
            run_state = 1
        if w.button("Pause"):
            run_state = 0
        if w.button("Step"):
            run_state = 2