import taichi as ti
import numpy as np
from heightmap import Heightmap

############################## Cube ################################
cube_verts_list = np.array([
    [-1,-1,-1],
    [-1,-1, 1],
    [-1, 1,-1],
    [-1, 1, 1],
    [1,-1,-1],
    [1,-1, 1],
    [1, 1,-1],
    [1, 1, 1]
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
    [-1, 0, 0],
    [1, 0, 0],
    [0,-1, 0],
    [0, 1, 0],
    [0, 0,-1],
    [0, 0, 1]
], dtype=np.float32)
cube_colors_list_lvl0 = np.array([
    [0, 0.5, 0, 0.1],
    [0, 0.5, 0, 0.1],
    [0, 0.5, 0, 0.1],
    [0, 0.5, 0, 0.1],
    [0, 0.5, 0, 0.1],
    [0, 0.5, 0, 0.1],
    [0, 0.5, 0, 0.1],
    [0, 0.5, 0, 0.1]
], dtype=np.float32)
cube_colors_list_lvl1 = np.array([
    [0.5, 0, 0, 0.1],
    [0.5, 0, 0, 0.1],
    [0.5, 0, 0, 0.1],
    [0.5, 0, 0, 0.1],
    [0.5, 0, 0, 0.1],
    [0.5, 0, 0, 0.1],
    [0.5, 0, 0, 0.1],
    [0.5, 0, 0, 0.1]
], dtype=np.float32)

@ti.data_oriented
class Grid:
    def __init__(self,n_grid,dim,heightmap):
        self.F_grid_level = ti.field(int, (n_grid, ) * dim)
        self.grid_size_to_km = heightmap.hm_height_px*heightmap.px_to_km/n_grid
        self.cube_positions = ti.Vector.field(dim, ti.f32, 8)
        self.cube_positions.from_numpy(cube_verts_list)

        self.cube_indices = ti.field(ti.i32, shape=len(cube_faces_list))
        self.cube_indices.from_numpy(cube_faces_list)

        self.cube_normals = ti.Vector.field(dim, ti.f32, 8)
        self.cube_normals.from_numpy(cube_face_normals_list)

        self.cube_colors_lvl0 = ti.Vector.field(4, ti.f32, 8)
        self.cube_colors_lvl0.from_numpy(cube_colors_list_lvl0)
        self.cube_colors_lvl1 = ti.Vector.field(4, ti.f32, 8)
        self.cube_colors_lvl1.from_numpy(cube_colors_list_lvl1)

        self.curr_cube_positions = ti.Vector.field(dim, ti.f32, 8)
        self.m_transforms_lvl0 = ti.Matrix.field(4,4,dtype=ti.f32,shape=n_grid*n_grid*n_grid)
        self.n_grid = n_grid
        self.set_levels(heightmap)
        self.calculate_m_transforms_lvl0()
        self.info = [None]*6
        self.info[0] = 0.0
        self.info[1] = self.grid_size_to_km
        self.info[2] = n_grid
        self.info[3] = 0.0
        self.info[4] = n_grid
        self.info[5] = self.grid_size_to_km
    
    @ti.kernel
    def calculate_m_transforms_lvl0(self):
        for idx in self.m_transforms_lvl0:
            i = idx//(self.n_grid*self.n_grid)
            j = (idx-self.n_grid*self.n_grid*i)//self.n_grid
            k = idx%self.n_grid
            if (self.F_grid_level[i,j,k]==0):
                self.m_transforms_lvl0[idx] = ti.Matrix.identity(float,4)
                self.m_transforms_lvl0[idx] *= self.grid_size_to_km/2.0
                self.m_transforms_lvl0[idx][0,3] = i*self.grid_size_to_km + self.grid_size_to_km/2.0
                self.m_transforms_lvl0[idx][1,3] = j*self.grid_size_to_km + self.grid_size_to_km/2.0
                self.m_transforms_lvl0[idx][2,3] = k*self.grid_size_to_km + self.grid_size_to_km/2.0
                self.m_transforms_lvl0[idx][3,3] = 1
            else:
                self.m_transforms_lvl0[idx] = ti.Matrix.identity(float,4)
                self.m_transforms_lvl0[idx] *= self.grid_size_to_km/2.0
                self.m_transforms_lvl0[idx][0,3] = 324534654
                self.m_transforms_lvl0[idx][1,3] = 324534654
                self.m_transforms_lvl0[idx][2,3] = 324534654
                self.m_transforms_lvl0[idx][3,3] = 1


    @ti.kernel
    def set_levels(self,heightmap: ti.template()):
        # Fill the image
        for i,j,k in self.F_grid_level:
            # if grid height is inside terrain
            if(j*self.grid_size_to_km <= heightmap.heightmap_positions[int((i/self.n_grid+1.0/(2.0*self.n_grid))*heightmap.hm_height_px)*heightmap.hm_width_px+int((k/self.n_grid+1.0/(2.0*self.n_grid))*heightmap.hm_width_px)][1]):
                self.F_grid_level[i,j,k] = 0
            else:
                self.F_grid_level[i,j,k] = 1