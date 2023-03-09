import numpy as np
from MOLASSES.neighbor_8 import neighbor_id

np.random.seed(42)

def distribute(grid, activeList, CAListSize, activeCount, activeNeighbor, gridinfo, render,camera,window,scene,canvas,heightmap,show_options,gui):
    # print('INIT')
    ct = 0
    neighborCount = 0
    myResidual, thickness = 0.0, 0.0
    n = 0
    lavaOut, lavaIn = 0.0, 0.0
    parentCode = 0
    # more = None
    active_neighbor = 0
    total_wt, my_wt = 0, 0
    i, j, max_, temp, r, nc = 0, 0, 0, 0, 0, 0
    shuffle = [0] * 8
    excess = 0

    activeCount = 1
    # for all active cells
    while ct < activeCount:
        myResidual = grid.residual[activeList[ct].row,activeList[ct].col]
        thickness = grid.eff_elev[activeList[ct].row,activeList[ct].col] - grid.dem_elev[activeList[ct].row,activeList[ct].col]
        lavaOut = thickness - myResidual

        # Find neighbor cells which are not parents and have lower elevation than active cell
        neighborCount = neighbor_id(activeList[ct], # Automata Center Cell (parent))
                                    grid,           # DataCell Global Data Grid
                                    gridinfo,       # double grid data
                                    activeNeighbor  # list for active neighbors
                                    )
        # print(f'ct: {ct} neighborCount: {neighborCount} activeList[ct].row: {activeList[ct].row} activeList[ct].col: {activeList[ct].col}')

        # If neighbors are found
        if neighborCount > 0:
            max_ = neighborCount - 1
            total_wt = 0.0
            lavaIn = 0.0
            for i in range(neighborCount):
                total_wt += activeNeighbor[i].elev_diff
                shuffle[i] = i

            if neighborCount > 1:
                for i in range(neighborCount - 1):
                    r = np.random.randint(0, max_)
                    temp = shuffle[r]
                    shuffle[r] = shuffle[max_]
                    shuffle[max_] = temp
                    max_ -= 1
            
            # for i in range(neighborCount):
            #     print(shuffle[i])

            # For each neighbor
            for nc in range(neighborCount):
                n = shuffle[nc]

                if activeNeighbor[n].row > activeList[ct].row and activeNeighbor[n].col < activeList[ct].col:
                    # (0011) this neighbor's parent is SE
                    parentCode = 3
                elif activeNeighbor[n].row > activeList[ct].row and activeNeighbor[n].col > activeList[ct].col:
                    # (1001) this neighbor's parent is SW
                    parentCode = 9
                elif activeNeighbor[n].row < activeList[ct].row and activeNeighbor[n].col < activeList[ct].col:
                    # (0110) this neighbor's parent is NE
                    parentCode = 6
                elif activeNeighbor[n].row < activeList[ct].row and activeNeighbor[n].col > activeList[ct].col:
                    # (1100) this neighbor's parent is NW
                    parentCode = 12
                elif activeNeighbor[n].row > activeList[ct].row:
                    # (0001) this neighbor's parent is SOUTH
                    parentCode = 1
                elif activeNeighbor[n].col < activeList[ct].col:
                    # (0010) this neighbor's parent is EAST
                    parentCode = 2
                elif activeNeighbor[n].row < activeList[ct].row:
                    # (0100) this neighbor's parent is NORTH
                    parentCode = 4
                elif activeNeighbor[n].col > activeList[ct].col:
                    # (1000) this neighbor's parent is WEST
                    parentCode = 8

                # Assign parentCode to neighbor grid cell
                grid.parentcode[activeNeighbor[n].row,activeNeighbor[n].col] = parentCode

                # Now Calculate the amount of lava this neighbor gets		
				# This neighbor gets lava proportional to the elevation difference with its parent;
				# lower neighbors get more lava, higher neighbors get less lava.
                if total_wt > 0.0:
                    my_wt = activeNeighbor[n].elev_diff
                    lavaIn = lavaOut * (my_wt / total_wt)
                else:
                    print("PROBLEM: Cannot divide by zero or difference is less than 0: total_wt =", total_wt)
                    return -1, activeCount
                
                # Distribute lava to neighbor
                grid.eff_elev[activeNeighbor[n].row,activeNeighbor[n].col] += lavaIn
                myResidual = grid.residual[activeNeighbor[n].row,activeNeighbor[n].col]
                thickness = grid.eff_elev[activeNeighbor[n].row,activeNeighbor[n].col] - grid.dem_elev[activeNeighbor[n].row,activeNeighbor[n].col]

                # NOW, IF neighbor has excess lava
                # print(f'n: {n} activeNeighbor[n].row {activeNeighbor[n].row} activeNeighbor[n].col {activeNeighbor[n].col} thickness:  {thickness} myResidual: {myResidual}')
                if(thickness > myResidual):
                    # check if this neighbor is active
                    active_neighbor = grid.active[activeNeighbor[n].row,activeNeighbor[n].col]
                    # print(f'n: {n} active_neighbor: {active_neighbor}')
                    if (active_neighbor < 0):
                        # If neighbor cell is not on active list (first time with excess lava) or 
                        # neighbor was on active list of previous pulse

                        # Add neighbor to end of current active list
                        activeList[activeCount].row = activeNeighbor[n].row
                        activeList[activeCount].col = activeNeighbor[n].col
                        grid.active[activeNeighbor[n].row,activeNeighbor[n].col] = activeCount
                        activeCount += 1
                        # print(f'activeCount: {activeCount}')

                        # if(activeCount == CAListSize): # resize active list if more space is needed
                        #     CAListSize *= 2
                        #     more = activeList
                        #     if(more != None):
                        #         activeList = more
                        #     else:
                        #         return 1, activeCount
        
            # REMOVE LAVA FROM Parent CELL
            # Subtract lavaOut  from activeCell's  effective elevation
            grid.eff_elev[activeList[ct].row,activeList[ct].col] -= lavaOut
            activeList[ct].excess = 0
            
            grid.calculate_m_transforms_lvl1()
            render(camera,window,scene,canvas,heightmap,grid)
            show_options(gui)
            window.show()
        elif (neighborCount < 0): # might be off the grid
            return neighborCount, activeCount
        ct+=1
        # print(f'ct: {ct} activeCount: {activeCount}')
        if(ct == activeCount and excess < 3):
            ct = 0
            excess += 1
    
    for j in range(activeCount):
        grid.active[activeList[j].row,activeList[j].col] = -1
    return 0, activeCount