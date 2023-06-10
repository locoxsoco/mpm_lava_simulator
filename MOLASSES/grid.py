import taichi as ti
import numpy as np
from heightmap import Heightmap

############################## Cube ################################
cube_verts_list = np.array([
    [-0.5, 0.0,-0.5],
    [-0.5, 0.0, 0.5],
    [-0.5, 1.0,-0.5],
    [-0.5, 1.0, 0.5],
    [ 0.5, 0.0,-0.5],
    [ 0.5, 0.0, 0.5],
    [ 0.5, 1.0,-0.5],
    [ 0.5, 1.0, 0.5]
], dtype=np.float32)
cube_faces_list = np.array([
    0, 1, 2, 2, 1, 3,
    5, 4, 7, 7, 4, 6,
    0, 4, 1, 1, 4, 5,
    3, 7, 2, 2, 7, 6,
    4, 0, 6, 6, 0, 2,
    1, 5, 3, 3, 5, 7
], dtype=np.int32)
cube_face_normals_list = np.array([
    [-1.0, 0.0,  0.0],
    [ 1.0, 0.0,  0.0],
    [ 0.0,-1.0,  0.0],
    [ 0.0, 1.0,  0.0],
    [ 0.0, 0.0, -1.0],
    [ 0.0, 0.0,  1.0]
], dtype=np.float32)
# Heightmap color
cube_colors_list_lvl0 = np.array([
    [54.0/256.0, 47.0/256.0, 54.0/256.0, 1.0],
    [54.0/256.0, 47.0/256.0, 54.0/256.0, 1.0],
    [54.0/256.0, 47.0/256.0, 54.0/256.0, 1.0],
    [54.0/256.0, 47.0/256.0, 54.0/256.0, 1.0],
    [54.0/256.0, 47.0/256.0, 54.0/256.0, 1.0],
    [54.0/256.0, 47.0/256.0, 54.0/256.0, 1.0],
    [54.0/256.0, 47.0/256.0, 54.0/256.0, 1.0],
    [54.0/256.0, 47.0/256.0, 54.0/256.0, 1.0]
], dtype=np.float32)
# Lava color
# cube_colors_list_lvl1 = np.array([
#     [169.0/256.0, 24.0/256.0, 35.0/256.0, 1.0],
#     [169.0/256.0, 24.0/256.0, 35.0/256.0, 1.0],
#     [169.0/256.0, 24.0/256.0, 35.0/256.0, 1.0],
#     [169.0/256.0, 24.0/256.0, 35.0/256.0, 1.0],
#     [169.0/256.0, 24.0/256.0, 35.0/256.0, 1.0],
#     [169.0/256.0, 24.0/256.0, 35.0/256.0, 1.0],
#     [169.0/256.0, 24.0/256.0, 35.0/256.0, 1.0],
#     [169.0/256.0, 24.0/256.0, 35.0/256.0, 1.0]
# ], dtype=np.float32)
cube_colors_list_lvl1 = np.array([
    [207.0/256.0, 16.0/256.0, 32.0/256.0, 1.0],
    [207.0/256.0, 16.0/256.0, 32.0/256.0, 1.0],
    [207.0/256.0, 16.0/256.0, 32.0/256.0, 1.0],
    [207.0/256.0, 16.0/256.0, 32.0/256.0, 1.0],
    [207.0/256.0, 16.0/256.0, 32.0/256.0, 1.0],
    [207.0/256.0, 16.0/256.0, 32.0/256.0, 1.0],
    [207.0/256.0, 16.0/256.0, 32.0/256.0, 1.0],
    [207.0/256.0, 16.0/256.0, 32.0/256.0, 1.0]
], dtype=np.float32)
# Crust color
cube_colors_list_lvl2 = np.array([
    [93.0/256.0, 40.0/256.0, 39.0/256.0, 1.0],
    [93.0/256.0, 40.0/256.0, 39.0/256.0, 1.0],
    [93.0/256.0, 40.0/256.0, 39.0/256.0, 1.0],
    [93.0/256.0, 40.0/256.0, 39.0/256.0, 1.0],
    [93.0/256.0, 40.0/256.0, 39.0/256.0, 1.0],
    [93.0/256.0, 40.0/256.0, 39.0/256.0, 1.0],
    [93.0/256.0, 40.0/256.0, 39.0/256.0, 1.0],
    [93.0/256.0, 40.0/256.0, 39.0/256.0, 1.0]
], dtype=np.float32)

# @ti.func
# class Neighbor:
#     def __init__(self,row,col,elev_diff):
#         self.row = row
#         self.col = col
#         self.elev_diff = elev_diff

@ti.data_oriented
class Grid:
    def __init__(self,n_grid,dim,heightmap):
        self.grid_size_to_km = heightmap.hm_height_px*heightmap.px_to_km/n_grid

        self.cube_positions_dem = ti.Vector.field(dim, ti.f32, 8)
        self.cube_positions_lava1 = ti.Vector.field(dim, ti.f32, 8)
        self.cube_positions_dem.from_numpy(cube_verts_list)
        self.cube_positions_lava1.from_numpy(cube_verts_list)
        self.cube_indices = ti.field(ti.i32, shape=len(cube_faces_list))
        self.cube_indices.from_numpy(cube_faces_list)
        self.cube_normals = ti.Vector.field(dim, ti.f32, 8)
        self.cube_normals.from_numpy(cube_face_normals_list)
        self.cube_colors_lvl0 = ti.Vector.field(4, ti.f32, 8)
        self.cube_colors_lvl0.from_numpy(cube_colors_list_lvl0)
        self.cube_colors_lvl1 = ti.Vector.field(4, ti.f32, 8)
        self.cube_colors_lvl1.from_numpy(cube_colors_list_lvl1)
        self.curr_cube_positions = ti.Vector.field(dim, ti.f32, 8)

        self.m_transforms_lvl0 = ti.Matrix.field(4,4,dtype=ti.f32,shape=n_grid*n_grid)
        self.m_transforms_lvl1 = ti.Matrix.field(4,4,dtype=ti.f32,shape=n_grid*n_grid)

        self.n_grid = n_grid

        self.residual = ti.field(ti.f32, (n_grid, ) * (dim-1))
        self.eff_elev = ti.field(ti.f32, (n_grid, ) * (dim-1))
        self.out_eff_elev = ti.field(ti.f32, (n_grid, ) * (dim-1))
        self.dem_elev = ti.field(ti.f32, (n_grid, ) * (dim-1))
        self.neighborListElevDiff = ti.field(ti.f32, shape=(n_grid,n_grid,8))
        self.neighborListRow = ti.field(ti.f32, shape=(n_grid,n_grid,8))
        self.neighborListCol = ti.field(ti.f32, shape=(n_grid,n_grid,8))
        self.neighborListCounter = ti.field(ti.i32, shape=(n_grid,n_grid))

        self.parentcodes = ti.field(ti.i32, (n_grid, ) * (dim-1))
        opRows = np.array([ 1,-1, 0, 0, 1, 1,-1,-1], dtype=np.int32)
        opCols = np.array([ 0, 0,+1,-1,+1,-1,+1,-1], dtype=np.int32)
        neighCodes = np.array([1,2,4,8,16,32,64,128], dtype=np.int32)
        sqrt2 = ti.math.sqrt(2)
        neighDistances = np.array([1.0,1.0,1.0,1.0,sqrt2,sqrt2,sqrt2,sqrt2], dtype=np.float32)

        self.opRows = ti.field(ti.i32, 8)
        self.opCols = ti.field(ti.i32, 8)
        self.neighCodes = ti.field(ti.i32, 8)
        self.neighDistances = ti.field(ti.f32, 8)
        self.opRows.from_numpy(opRows)
        self.opCols.from_numpy(opCols)
        self.neighCodes.from_numpy(neighCodes)
        self.neighDistances.from_numpy(neighDistances)


        self.active = ti.field(ti.i32, (n_grid, ) * (dim-1))
        self.init_values(heightmap)
        self.calculate_m_transforms_lvl0()
        self.calculate_m_transforms_lvl1()

        # DEMGeoTransform[0] lower left x
        # DEMGeoTransform[1] w-e pixel resolution (positive value)
        # DEMGeoTransform[2] number of cols, assigned manually in this module
        # DEMGeoTransform[3] lower left y
        # DEMGeoTransform[4] number of rows, assigned manually in this module
        # DEMGeoTransform[5] n-s pixel resolution (negative value)
        self.info = [None]*6
        self.info[0] = 0.0
        self.info[1] = self.grid_size_to_km
        self.info[2] = n_grid
        self.info[3] = 0.0
        self.info[4] = n_grid
        self.info[5] = self.grid_size_to_km
        print(f'self.info[5]: {self.info[5]}')

    @ti.kernel
    def init_values(self,heightmap: ti.template()):
        for i,j in self.residual:
            self.residual[i,j] = 0.0
            self.dem_elev[i,j] = heightmap.heightmap_positions[int((i/self.n_grid+1.0/(2.0*self.n_grid))*heightmap.hm_height_px)*heightmap.hm_width_px+int((j/self.n_grid+1.0/(2.0*self.n_grid))*heightmap.hm_width_px)][1]
            self.eff_elev[i,j] = self.dem_elev[i,j]
            self.out_eff_elev[i,j] = self.dem_elev[i,j]
            self.parentcodes[i,j] = 0
            self.active[i,j] = -1

    @ti.kernel
    def calculate_m_transforms_lvl0(self):
        for idx in self.m_transforms_lvl0:
            i = idx//self.n_grid
            k = idx%self.n_grid
            self.m_transforms_lvl0[idx] = ti.Matrix.identity(float,4)
            self.m_transforms_lvl0[idx] *= self.grid_size_to_km
            self.m_transforms_lvl0[idx][1,1] = 1.0
            self.m_transforms_lvl0[idx][1,1] *= self.dem_elev[i,k]
            self.m_transforms_lvl0[idx][0,3] = i*self.grid_size_to_km + self.grid_size_to_km
            # self.m_transforms_lvl0[idx][1,3] = self.dem_elev[i,k] + self.grid_size_to_km
            self.m_transforms_lvl0[idx][2,3] = k*self.grid_size_to_km + self.grid_size_to_km
            self.m_transforms_lvl0[idx][3,3] = 1

    @ti.kernel
    def calculate_m_transforms_lvl1(self):
        for idx in self.m_transforms_lvl1:
            i = idx//self.n_grid
            k = idx%self.n_grid
            thickness = self.eff_elev[i,k]-self.dem_elev[i,k]
            if thickness > 0.00001:
                self.m_transforms_lvl1[idx] = ti.Matrix.identity(float,4)
                self.m_transforms_lvl1[idx] *= self.grid_size_to_km
                self.m_transforms_lvl1[idx][1,1] = 1.0
                self.m_transforms_lvl1[idx][1,1] *= thickness
                self.m_transforms_lvl1[idx][0,3] = i*self.grid_size_to_km + self.grid_size_to_km
                self.m_transforms_lvl1[idx][1,3] = self.dem_elev[i,k]
                self.m_transforms_lvl1[idx][2,3] = k*self.grid_size_to_km + self.grid_size_to_km
                self.m_transforms_lvl1[idx][3,3] = 1

    @ti.kernel
    def fill_residual(self,residual_value: float):
        for i in range(self.info[4]):
            for j in range(self.info[2]):
                self.residual[i,j] = residual_value

    @ti.kernel
    def distribute(self):
        for i,k in self.residual:
            myResidual = self.residual[i,k]
            thickness = self.eff_elev[i,k] - self.dem_elev[i,k]
            lavaOut = thickness - myResidual

            if(lavaOut>0):
                # Find neighbor cells which are not parents and have lower elevation than active cell
                self.neighbor_id(i,k) # Automata Center Cell (parent))
                total_wt = 0.0
                for n in range(self.neighborListCounter[i,k]):
                    total_wt += self.neighborListElevDiff[i,k,n]
                for n in range(self.neighborListCounter[i,k]):
                    my_wt = self.neighborListElevDiff[i,k,n]
                    lavaIn = lavaOut * (my_wt / total_wt)
                    self.eff_elev[int(self.neighborListRow[i,k,n]),int(self.neighborListCol[i,k,n])] += lavaIn
                self.eff_elev[i,k] -= lavaOut

    @ti.func
    def neighbor_id(self,i,k):
        neighborCount = 0
        aRow,aCol = i,k
        # Nrow, Srow, Ecol, Wcol = int(aRow + 1), int(aRow - 1), int(aCol + 1), int(aCol - 1)

        for n in range(8):
            neighborCount = self.addNeighbor(aRow,aCol,self.opRows[n],self.opCols[n],self.neighCodes[n],self.neighDistances[n],neighborCount)

        # # NORTH neighbor
        # code = self.parentcodes[aRow,aCol] & 1
        # if not(code) and Nrow < 400: # NORTH cell is not the parent of active cell
        #     if self.eff_elev[aRow,aCol] > self.eff_elev[Nrow,aCol]: # active cell is higher than North neighbor
        #         # Calculate elevation difference between active cell and its North neighbor
        #         self.neighborListElevDiff[i,k,neighborCount] = self.eff_elev[aRow,aCol] - self.eff_elev[Nrow,aCol] # 1.0 is the weight for a cardinal direction cell
        #         self.neighborListRow[i,k,neighborCount] = Nrow
        #         self.neighborListCol[i,k,neighborCount] = aCol
        #         # print(f'neighborCount: {neighborCount} Nrow:{Nrow} aCol: {aCol} neighborList[{neighborCount}].row: {neighborList[neighborCount].row} neighborList[{neighborCount}].col: {neighborList[neighborCount].col}')
        #         neighborCount += 1
        
        # # SOUTH
        # code = self.parentcodes[aRow,aCol] & 2
        # if not(code) and Srow >= 0: # SOUTH cell is not the parent of active cell
        #     if self.eff_elev[aRow,aCol] > self.eff_elev[Srow,aCol]: # active cell is higher than SOUTH neighbor
        #         # Calculate elevation difference between active and neighbor
        #         self.neighborListElevDiff[i,k,neighborCount] = self.eff_elev[aRow,aCol] - self.eff_elev[Srow,aCol] # 1.0 is the weight for a cardinal direction cell
        #         self.neighborListRow[i,k,neighborCount] = Srow
        #         self.neighborListCol[i,k,neighborCount] = aCol
        #         # # print(f'neighborCount: {neighborCount} Srow:{Srow} aCol: {aCol} neighborList[{neighborCount}].row: {neighborList[neighborCount].row} neighborList[{neighborCount}].col: {neighborList[neighborCount].col}')
        #         neighborCount += 1

        # # EAST
        # code = self.parentcodes[aRow,aCol] & 4
        # if not(code) and Ecol < 400: # EAST cell is not the parent of active cell
        #     if self.eff_elev[aRow,aCol] > self.eff_elev[aRow,Ecol]: # active cell is higher than EAST neighbor
        #         # Calculate elevation difference between active and neighbor
        #         self.neighborListElevDiff[i,k,neighborCount] = self.eff_elev[aRow,aCol] - self.eff_elev[aRow,Ecol] # 1.0 is the weight for a cardinal direction cell
        #         self.neighborListRow[i,k,neighborCount] = aRow
        #         self.neighborListCol[i,k,neighborCount] = Ecol
        #         # # print(f'neighborCount: {neighborCount-1} Nrow:{Nrow} aCol: {aCol} neighborList[{neighborCount-1}].row: {neighborList[neighborCount-1].row} neighborList[{neighborCount-1}].col: {neighborList[neighborCount-1].col}')
        #         # # print(f'neighborCount: {neighborCount} aRow:{aRow} Ecol: {Ecol} neighborList[{neighborCount}].row: {neighborList[neighborCount].row} neighborList[{neighborCount}].col: {neighborList[neighborCount].col}')
        #         neighborCount += 1

        # # WEST
        # code = self.parentcodes[aRow,aCol] & 8
        # if not(code) and Wcol >= 0: # WEST cell is not the parent of active cell
        #     if self.eff_elev[aRow,aCol] > self.eff_elev[aRow,Wcol]: # active cell is higher than WEST neighbor
        #         # Calculate elevation difference between active and neighbor
        #         self.neighborListElevDiff[i,k,neighborCount] = self.eff_elev[aRow,aCol] - self.eff_elev[aRow,Wcol] # 1.0 is the weight for a cardinal direction cell
        #         self.neighborListRow[i,k,neighborCount] = aRow
        #         self.neighborListCol[i,k,neighborCount] = Wcol
        #         neighborCount += 1

        # # DIAGONAL CELLS
        # # NORTHEAST
        # code = self.parentcodes[aRow,aCol] & 16
        # if not(code) and Nrow < 400 and Ecol < 400: # NE cell is not the parent of active cell
        #     if self.eff_elev[aRow,aCol] > self.eff_elev[Nrow,Ecol]: # active cell is higher than NE neighbor
        #         # Calculate elevation difference between active and neighbor
        #         self.neighborListElevDiff[i,k,neighborCount] = (self.eff_elev[aRow,aCol] - self.eff_elev[Nrow,Ecol])/ti.math.sqrt(2) # SQRT2 is the weight for a diagonal cell
        #         self.neighborListRow[i,k,neighborCount] = Nrow
        #         self.neighborListCol[i,k,neighborCount] = Ecol
        #         neighborCount += 1

        # # NORTHWEST
        # code = self.parentcodes[aRow,aCol] & 32
        # if not(code) and Nrow < 400 and Wcol >= 0: # NW cell is not the parent of active cell
        #     if self.eff_elev[aRow,aCol] > self.eff_elev[Nrow,Wcol]: # active cell is higher than NW neighbor
        #         # Calculate elevation difference between active and neighbor
        #         self.neighborListElevDiff[i,k,neighborCount] = (self.eff_elev[aRow,aCol] - self.eff_elev[Nrow,Wcol])/ti.math.sqrt(2) # SQRT2 is the weight for a diagonal cell
        #         self.neighborListRow[i,k,neighborCount] = Nrow
        #         self.neighborListCol[i,k,neighborCount] = Wcol
        #         neighborCount += 1
        
        # # SOUTHEAST
        # code = self.parentcodes[aRow,aCol] & 64
        # if not(code) and Srow >= 0 and Ecol < 400: # SE cell is not the parent of active cell
        #     if self.eff_elev[aRow,aCol] > self.eff_elev[Srow,Ecol]: # active cell is higher than SE neighbor
        #         # Calculate elevation difference between active and neighbor
        #         self.neighborListElevDiff[i,k,neighborCount] = (self.eff_elev[aRow,aCol] - self.eff_elev[Srow,Ecol])/ti.math.sqrt(2) # SQRT2 is the weight for a diagonal cell
        #         self.neighborListRow[i,k,neighborCount] = Srow
        #         self.neighborListCol[i,k,neighborCount] = Ecol
        #         neighborCount += 1
        
        # # SOUTHWEST
        # code = self.parentcodes[aRow,aCol] & 128
        # if not(code) and Srow >= 0 and Wcol >= 0: # SW cell is not the parent of active cell
        #     if self.eff_elev[aRow,aCol] > self.eff_elev[Srow,Wcol]: # active cell is higher than SW neighbor
        #         # Calculate elevation difference between active and neighbor
        #         self.neighborListElevDiff[i,k,neighborCount] = (self.eff_elev[aRow,aCol] - self.eff_elev[Srow,Wcol])/ti.math.sqrt(2) # SQRT2 is the weight for a diagonal cell
        #         self.neighborListRow[i,k,neighborCount] = Srow
        #         self.neighborListCol[i,k,neighborCount] = Wcol
        #         neighborCount += 1    

        self.neighborListCounter[i,k] = neighborCount
        # # for i in range(neighborCount):
        # #     print(f'i: {i} neighborList[i].row: {neighborList[i].row} neighborList[i].col: {neighborList[i].col}')
        # # return neighborListRow,neighborListCol,neighborListElevDiff
    
    @ti.func
    def addNeighbor(self, aRow: int, aCol: int, opRow: int, opCol: int, neighCode: int, distance: float, neighborCount: int):
        nRow, nCol = aRow + opRow, aCol + opCol
        code = self.parentcodes[aRow,aCol] & neighCode
        if not(code) and nRow >= 0 and nRow < 400 and nCol >=0 and nRow < 400: # neigh cell is not the parent of active cell
            if self.eff_elev[aRow,aCol] > self.eff_elev[nRow,nCol]: # active cell is higher than SW neighbor
                # Calculate elevation difference between active and neighbor
                self.neighborListElevDiff[aRow,aCol,neighborCount] = (self.eff_elev[aRow,aCol] - self.eff_elev[nRow,nCol])/distance
                self.neighborListRow[aRow,aCol,neighborCount] = nRow
                self.neighborListCol[aRow,aCol,neighborCount] = nCol
                neighborCount += 1
        return neighborCount

    # def Intersect(self,rayPosition,rayDirection):
    #     # Compute bounding box
    #     Box2 box = getDomain()

    #     # Check the intersection with the bounding box
    #     Vector3 r0 = ray(0);
    #     Vector3 r1 = ray(1);
    #     if (!box.Intersect(Vector2(r0[0], r0[1]), Vector2(r1[0], r1[1]), ta, tb))
    #         return false;

    #     t = ta + 0.0001;
    #     if (ta < 0.0)
    #     {
    #         t = 0.0;
    #     }

    #     # Ray marching
    #     while (t < tb)
    #     {
    #         // Point along the ray
    #         Vector3 p = ray(t);
    #         double h = Height(Vector2(p[0], p[1]));
    #         if (h > p[2]) {
    #             q = Vector3(p[0], p[1], h);
    #             return true;
    #         }
    #         else {
    #             t += 1.0;
    #         }
    #     }
    #     return false;
