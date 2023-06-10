def pulse(actList, active_flow, grid, volumeRemaining, gridinfo, volumeErupted):
    
    # pulseThickness = 0.0 # Pulse Volume divided by data grid resolution
    
    if volumeRemaining > 0.0:
        if active_flow.pulsevolume > active_flow.currentvolume:
            pulsevolume = active_flow.currentvolume
        else:
            pulsevolume = active_flow.pulsevolume
        pulseThickness = pulsevolume / (gridinfo[1] * gridinfo[5])
        active_flow.currentvolume -= pulsevolume # Subtract pulse volume from flow's total magma budget
        volumeErupted += pulsevolume
        volumeRemaining = active_flow.currentvolume
        # If the flow has a thickness of lava greater than it's residual, then it has lava to give so put it on the active list
        grid.eff_elev[actList.row,actList.col] += pulseThickness

    return volumeRemaining,volumeErupted
