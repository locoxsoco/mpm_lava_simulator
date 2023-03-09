import math

def neighbor_id(active, grid, gridMetadata, neighborList):
    # Parent bitcode
    code = None
    # Neighbor Row and Col relative to active (active cell)
    neighborCount = 0
    aRow, aCol = active.row, active.col
    Nrow, Srow, Ecol, Wcol = int(aRow + 1), int(aRow - 1), int(aCol + 1), int(aCol - 1)
    # print(f'aRow: {aRow} aCol: {aCol} Nrow: {Nrow} Srow: {Srow} Ecol: {Ecol} Wcol: {Wcol}')
    
    # Calculate row and column locations for active cell and its neighbor cells
    if Srow < 0:
        print("\nFLOW IS OFF THE MAP! (South) [NEIGHBOR_ID]\n")
        return -1
    elif Nrow >= gridMetadata[4]:
        print("\nFLOW IS OFF THE MAP! (North) [NEIGHBOR_ID]\n")
        print(f'Nrow: {Nrow} >= gridMetadata[4] {gridMetadata[4]}')
        return -2
    elif Wcol < 0:
        print("\nFLOW IS OFF THE MAP! (West) [NEIGHBOR_ID]\n")
        return -3
    elif Ecol >= gridMetadata[2]:
        print("\nFLOW IS OFF THE MAP! (East) [NEIGHBOR_ID]\n")
        return -4
    
    # NORTH neighbor
    # code = grid.parentcode[aRow,aCol] & 4
    # if not(grid.parentcode[aRow,aCol] == 4): # NORTH cell is not the parent of active cell
    if grid.eff_elev[aRow,aCol] > grid.eff_elev[Nrow,aCol]: # active cell is higher than North neighbor
        # Calculate elevation difference between active cell and its North neighbor
        neighborList[neighborCount].elev_diff = grid.eff_elev[aRow,aCol] - grid.eff_elev[Nrow,aCol] # 1.0 is the weight for a cardinal direction cell
        neighborList[neighborCount].row = Nrow
        neighborList[neighborCount].col = aCol
        # print(f'neighborCount: {neighborCount} Nrow:{Nrow} aCol: {aCol} neighborList[{neighborCount}].row: {neighborList[neighborCount].row} neighborList[{neighborCount}].col: {neighborList[neighborCount].col}')
        neighborCount += 1
    
    # EAST
    # code = grid.parentcode[aRow,aCol] & 2
    # if not(grid.parentcode[aRow,aCol] == 2): # EAST cell is not the parent of active cell
    if grid.eff_elev[aRow,aCol] > grid.eff_elev[aRow,Ecol]: # active cell is higher than EAST neighbor
        # Calculate elevation difference between active and neighbor
        neighborList[neighborCount].elev_diff = grid.eff_elev[aRow,aCol] - grid.eff_elev[aRow,Ecol] # 1.0 is the weight for a cardinal direction cell
        neighborList[neighborCount].row = aRow
        neighborList[neighborCount].col = Ecol
        # print(f'neighborCount: {neighborCount-1} Nrow:{Nrow} aCol: {aCol} neighborList[{neighborCount-1}].row: {neighborList[neighborCount-1].row} neighborList[{neighborCount-1}].col: {neighborList[neighborCount-1].col}')
            # print(f'neighborCount: {neighborCount} aRow:{aRow} Ecol: {Ecol} neighborList[{neighborCount}].row: {neighborList[neighborCount].row} neighborList[{neighborCount}].col: {neighborList[neighborCount].col}')
        neighborCount += 1
            
    # SOUTH
    # code = grid.parentcode[aRow,aCol] & 1
    # if not(grid.parentcode[aRow,aCol] == 1): # SOUTH cell is not the parent of active cell
    if grid.eff_elev[aRow,aCol] > grid.eff_elev[Srow,aCol]: # active cell is higher than SOUTH neighbor
        # Calculate elevation difference between active and neighbor
        neighborList[neighborCount].elev_diff = grid.eff_elev[aRow,aCol] - grid.eff_elev[Srow,aCol] # 1.0 is the weight for a cardinal direction cell
        neighborList[neighborCount].row = Srow
        neighborList[neighborCount].col = aCol
        # print(f'neighborCount: {neighborCount} Srow:{Srow} aCol: {aCol} neighborList[{neighborCount}].row: {neighborList[neighborCount].row} neighborList[{neighborCount}].col: {neighborList[neighborCount].col}')
        neighborCount += 1
            
    # WEST
    # code = grid.parentcode[aRow,aCol] & 8
    # if not(grid.parentcode[aRow,aCol] == 8): # WEST cell is not the parent of active cell
    if grid.eff_elev[aRow,aCol] > grid.eff_elev[aRow,Wcol]: # active cell is higher than WEST neighbor
        # Calculate elevation difference between active and neighbor
        neighborList[neighborCount].elev_diff = grid.eff_elev[aRow,aCol] - grid.eff_elev[aRow,Wcol] # 1.0 is the weight for a cardinal direction cell
        neighborList[neighborCount].row = aRow
        neighborList[neighborCount].col = Wcol
        neighborCount += 1
    
    # DIAGONAL CELLS
    # SOUTHWEST
    # code = grid.parentcode[aRow,aCol] & 9
    # if not(grid.parentcode[aRow,aCol] == 9): # SW cell is not the parent of active cell
    if grid.eff_elev[aRow,aCol] > grid.eff_elev[Srow,Wcol]: # active cell is higher than SW neighbor
        # Calculate elevation difference between active and neighbor
        neighborList[neighborCount].elev_diff = (grid.eff_elev[aRow,aCol] - grid.eff_elev[Srow,Wcol])/math.sqrt(2) # SQRT2 is the weight for a diagonal cell
        neighborList[neighborCount].row = Srow
        neighborList[neighborCount].col = Wcol
        neighborCount += 1
    
    # SOUTHEAST
    # code = grid.parentcode[aRow,aCol] & 3
    # if not(grid.parentcode[aRow,aCol] == 3): # SE cell is not the parent of active cell
    if grid.eff_elev[aRow,aCol] > grid.eff_elev[Srow,Ecol]: # active cell is higher than SE neighbor
        # Calculate elevation difference between active and neighbor
        neighborList[neighborCount].elev_diff = (grid.eff_elev[aRow,aCol] - grid.eff_elev[Srow,Ecol])/math.sqrt(2) # SQRT2 is the weight for a diagonal cell
        neighborList[neighborCount].row = Srow
        neighborList[neighborCount].col = Ecol
        neighborCount += 1
    
    # NORTHEAST
    # code = grid.parentcode[aRow,aCol] & 6
    # if not(grid.parentcode[aRow,aCol] == 6): # NE cell is not the parent of active cell
    if grid.eff_elev[aRow,aCol] > grid.eff_elev[Nrow,Ecol]: # active cell is higher than NE neighbor
        # Calculate elevation difference between active and neighbor
        neighborList[neighborCount].elev_diff = (grid.eff_elev[aRow,aCol] - grid.eff_elev[Nrow,Ecol])/math.sqrt(2) # SQRT2 is the weight for a diagonal cell
        neighborList[neighborCount].row = Nrow
        neighborList[neighborCount].col = Ecol
        neighborCount += 1
    
    # NORTHWEST
    # code = grid.parentcode[aRow,aCol] & 12
    # if not(grid.parentcode[aRow,aCol] == 12): # NW cell is not the parent of active cell
    if grid.eff_elev[aRow,aCol] > grid.eff_elev[Nrow,Wcol]: # active cell is higher than NW neighbor
        # Calculate elevation difference between active and neighbor
        neighborList[neighborCount].elev_diff = (grid.eff_elev[aRow,aCol] - grid.eff_elev[Nrow,Wcol])/math.sqrt(2) # SQRT2 is the weight for a diagonal cell
        neighborList[neighborCount].row = Nrow
        neighborList[neighborCount].col = Wcol
        neighborCount += 1
    
    # for i in range(neighborCount):
    #     print(f'i: {i} neighborList[i].row: {neighborList[i].row} neighborList[i].col: {neighborList[i].col}')
    return neighborCount
    