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
    def __init__(self,n_grid,dim,heightmap,scaled_grid_size_m):
        self.km_to_m = 1000.0
        self.grid_size_to_km = heightmap.hm_height_px*heightmap.px_to_km/n_grid
        self.grid_size_to_m = self.grid_size_to_km * self.km_to_m
        self.scaled_grid_size_m = scaled_grid_size_m
        self.scaled_grid_size_km = self.scaled_grid_size_m/self.km_to_m
        self.grid_size_m_to_scaled_grid_size_m = self.scaled_grid_size_m/(self.grid_size_to_km*self.km_to_m)
        self.grid_size_km_to_scaled_grid_size_km = self.scaled_grid_size_km/self.grid_size_to_km

        self.cell_area_m = (self.scaled_grid_size_m)**2

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

        self.m_transforms_dem = ti.Matrix.field(4,4,dtype=ti.f32,shape=n_grid*n_grid)
        self.m_transforms_lava = ti.Matrix.field(4,4,dtype=ti.f32,shape=n_grid*n_grid)

        self.n_grid = n_grid

        self.residual = ti.field(ti.f32, (n_grid, ) * (dim-1))
        self.eff_elev = ti.field(ti.f32, (n_grid, ) * (dim-1))
        self.new_eff_elev = ti.field(ti.f32, (n_grid, ) * (dim-1))
        self.dem_elev = ti.field(ti.f32, (n_grid, ) * (dim-1))
        self.neighborListElevDiff = ti.field(ti.f32, shape=(n_grid,n_grid,8))
        self.neighborListRow = ti.field(ti.f32, shape=(n_grid,n_grid,8))
        self.neighborListCol = ti.field(ti.f32, shape=(n_grid,n_grid,8))
        self.neighborListCounter = ti.field(ti.i32, shape=(n_grid,n_grid))

        self.underground_m = ti.field(ti.f32, shape=())
        self.underground_m[None] = 1000.0

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

        self.c_factor = ti.field(ti.f32, shape=())
        self.c_factor[None] = 1.0


        self.active = ti.field(ti.i32, (n_grid, ) * (dim-1))
        self.init_values(heightmap)
        self.initialize_m_transforms_dem()
        self.initialize_m_transforms_lava()

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
            self.dem_elev[i,j] = (heightmap.heightmap_positions[i*heightmap.hm_width_px+j][1]*self.km_to_m+self.underground_m[None])*self.grid_size_m_to_scaled_grid_size_m
            self.eff_elev[i,j] = self.dem_elev[i,j]
            self.new_eff_elev[i,j] = self.eff_elev[i,j]
            self.parentcodes[i,j] = 0
            self.active[i,j] = -1
            if(i==200 and j==200):
                print(f'self.dem_elev[{i},{j}]: {self.dem_elev[i,j]}')

    @ti.kernel
    def initialize_m_transforms_dem(self):
        for idx in self.m_transforms_dem:
            i = idx//self.n_grid
            k = idx%self.n_grid
            self.m_transforms_dem[idx] = ti.Matrix.identity(float,4)
            self.m_transforms_dem[idx] *= self.scaled_grid_size_km
            self.m_transforms_dem[idx][1,1] = 1.0
            self.m_transforms_dem[idx][1,1] *= self.dem_elev[i,k]/self.km_to_m
            self.m_transforms_dem[idx][0,3] = i*self.scaled_grid_size_km
            # self.m_transforms_dem[idx][1,3] = self.dem_elev[i,k]
            self.m_transforms_dem[idx][2,3] = k*self.scaled_grid_size_km
            self.m_transforms_dem[idx][3,3] = 1
    
    @ti.kernel
    def initialize_m_transforms_lava(self):
        for idx in self.m_transforms_lava:
            i = idx//self.n_grid
            k = idx%self.n_grid
            self.m_transforms_lava[idx] = ti.Matrix.identity(float,4)
            self.m_transforms_lava[idx] *= self.scaled_grid_size_km
            self.m_transforms_lava[idx][0,3] = i*self.scaled_grid_size_km
            self.m_transforms_lava[idx][1,3] = 654654654
            self.m_transforms_lava[idx][2,3] = k*self.scaled_grid_size_km
            self.m_transforms_lava[idx][3,3] = 1
    
    @ti.kernel
    def calculate_m_transforms_dem(self):
        for idx in self.m_transforms_dem:
            i = idx//self.n_grid
            k = idx%self.n_grid
            self.m_transforms_dem[idx] = ti.Matrix.identity(float,4)
            self.m_transforms_dem[idx] *= self.scaled_grid_size_km
            self.m_transforms_dem[idx][1,1] = 1.0
            self.m_transforms_dem[idx][1,1] *= self.dem_elev[i,k]/self.km_to_m
            self.m_transforms_dem[idx][0,3] = i*self.scaled_grid_size_km
            # self.m_transforms_dem[idx][1,3] = self.dem_elev[i,k]
            self.m_transforms_dem[idx][2,3] = k*self.scaled_grid_size_km
            self.m_transforms_dem[idx][3,3] = 1

    @ti.kernel
    def calculate_m_transforms_lava(self):
        for idx in self.m_transforms_lava:
            i = idx//self.n_grid
            k = idx%self.n_grid
            thickness = self.eff_elev[i,k]-self.dem_elev[i,k]
            if thickness > 0.0:
                self.m_transforms_lava[idx][1,1] = 1.0
                self.m_transforms_lava[idx][1,1] *= (thickness/self.km_to_m)
                self.m_transforms_lava[idx][1,3] = (self.dem_elev[i,k]/self.km_to_m)
            else:
                self.m_transforms_lava[idx][1,3] = 654654654

                # self.m_transforms_lava[idx] = ti.Matrix.identity(float,4)
                # self.m_transforms_lava[idx] *= self.scaled_grid_size_km
                # self.m_transforms_lava[idx][1,1] = 1.0
                # self.m_transforms_lava[idx][1,1] *= thickness/self.km_to_m
                # self.m_transforms_lava[idx][0,3] = i*self.scaled_grid_size_km
                # self.m_transforms_lava[idx][1,3] = self.dem_elev[i,k]
                # self.m_transforms_lava[idx][2,3] = k*self.scaled_grid_size_km
                # self.m_transforms_lava[idx][3,3] = 1

    @ti.kernel
    def fill_residual(self,residual_value: float):
        for i,k in self.residual:
            self.residual[i,k] = residual_value

    @ti.kernel
    def distribute(self):
        for i,k in self.residual:
            myResidual = self.residual[i,k]
            thickness = self.eff_elev[i,k] - self.dem_elev[i,k]
            lavaOut = (thickness - myResidual) * self.c_factor[None]
            # if(i==200 and k==200):
            #     print(f'lavaOut: {lavaOut} thickness: {thickness}')

            if(lavaOut>0):
                # Find neighbor cells which are not parents and have lower elevation than active cell
                self.neighbor_id(i,k) # Automata Center Cell (parent))
                total_wt = 0.0
                for n in range(self.neighborListCounter[i,k]):
                    total_wt += self.neighborListElevDiff[i,k,n]
                for n in range(self.neighborListCounter[i,k]):
                    my_wt = self.neighborListElevDiff[i,k,n]
                    lavaIn = lavaOut * (my_wt / total_wt)
                    self.new_eff_elev[int(self.neighborListRow[i,k,n]),int(self.neighborListCol[i,k,n])] += lavaIn
                self.new_eff_elev[i,k] -= lavaOut
    
    @ti.kernel
    def updateEffElev(self):
        for i,k in self.residual:
            # if(i==200 and k==200):
            #     print(f'self.eff_elev[i,k]: {self.eff_elev[i,k]} self.new_eff_elev[i,k]: {self.new_eff_elev[i,k]}')
            self.eff_elev[i,k] = self.new_eff_elev[i,k]

    @ti.func
    def neighbor_id(self,i,k):
        neighborCount = 0
        aRow,aCol = i,k

        for n in range(8):
            neighborCount = self.addNeighbor(aRow,aCol,self.opRows[n],self.opCols[n],self.neighCodes[n],self.neighDistances[n],neighborCount)

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