def pulse(actList, active_flow, grid, volumeRemaining, gridinfo, volumeErupted,global_delta_time,grid_size_m_to_scaled_grid_size_m):
    
    # pulseThickness = 0.0 # Pulse Volume divided by data grid resolution
    
    if volumeRemaining > 0.0:
        # print(f'active_flow.currentvolume: {active_flow.currentvolume}')
        if active_flow.pulsevolume*global_delta_time > active_flow.currentvolume:
            pulsevolume = active_flow.currentvolume
        else:
            pulsevolume = active_flow.pulsevolume*global_delta_time
        pulseThickness = pulsevolume*grid_size_m_to_scaled_grid_size_m**3 / grid.cell_area_m
        active_flow.currentvolume -= pulsevolume # Subtract pulse volume from flow's total magma budget
        volumeErupted += pulsevolume
        volumeRemaining = active_flow.currentvolume
        # If the flow has a thickness of lava greater than it's residual, then it has lava to give so put it on the active list
        grid.eff_elev[actList.row,actList.col] += pulseThickness
        grid.new_eff_elev[actList.row,actList.col] += pulseThickness

    return volumeRemaining,volumeErupted
