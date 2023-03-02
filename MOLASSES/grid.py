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
cube_colors_list_lvl0 = np.array([
    [0, 0.5, 0, 1.0],
    [0, 0.5, 0, 1.0],
    [0, 0.5, 0, 1.0],
    [0, 0.5, 0, 1.0],
    [0, 0.5, 0, 1.0],
    [0, 0.5, 0, 1.0],
    [0, 0.5, 0, 1.0],
    [0, 0.5, 0, 1.0]
], dtype=np.float32)
cube_colors_list_lvl1 = np.array([
    [0.5, 0, 0, 0.5],
    [0.5, 0, 0, 0.5],
    [0.5, 0, 0, 0.5],
    [0.5, 0, 0, 0.5],
    [0.5, 0, 0, 0.5],
    [0.5, 0, 0, 0.5],
    [0.5, 0, 0, 0.5],
    [0.5, 0, 0, 0.5]
], dtype=np.float32)

@ti.data_oriented
class Grid:
    def __init__(self,n_grid,dim,heightmap):
        self.grid_size_to_km = heightmap.hm_height_px*heightmap.px_to_km/n_grid

        self.cube_positions = ti.Vector.field(dim, ti.f32, 8)
        self.cube_positions2 = ti.Vector.field(dim, ti.f32, 8)
        self.cube_positions.from_numpy(cube_verts_list)
        self.cube_positions2.from_numpy(cube_verts_list)
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
        self.dem_elev = ti.field(ti.f32, (n_grid, ) * (dim-1))
        self.parentcode = ti.field(ti.i32, (n_grid, ) * (dim-1))
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
            self.parentcode[i,j] = -1
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