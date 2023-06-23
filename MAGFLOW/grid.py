import taichi as ti
import numpy as np

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
# Heightmap and solid lava color
cube_colors_list_dem = np.array([
    [54.0/256.0, 47.0/256.0, 54.0/256.0, 1.0],
    [54.0/256.0, 47.0/256.0, 54.0/256.0, 1.0],
    [54.0/256.0, 47.0/256.0, 54.0/256.0, 1.0],
    [54.0/256.0, 47.0/256.0, 54.0/256.0, 1.0],
    [54.0/256.0, 47.0/256.0, 54.0/256.0, 1.0],
    [54.0/256.0, 47.0/256.0, 54.0/256.0, 1.0],
    [54.0/256.0, 47.0/256.0, 54.0/256.0, 1.0],
    [54.0/256.0, 47.0/256.0, 54.0/256.0, 1.0],
], dtype=np.float32)
# Lava colors
color_lava_high = ti.Vector([169.0/256.0, 24.0/256.0, 35.0/256.0, 1.0])
color_lava_low = ti.Vector([93.0/256.0, 40.0/256.0, 39.0/256.0, 1.0])
colors_lava = np.array([[0.5, 0.0, 0.0, 1.0],
                   [1.0, 0.0, 0.0, 1.0],
                   [1.0, 0.5, 0.0, 1.0],
                   [1.0, 1.0, 0.0, 1.0],
                   [1.0, 1.0, 1.0, 1.0]])
# Select color
cube_colors_list_lvl2 = np.array([
    [0.0/256.0, 256.0/256.0, 0.0/256.0, 1.0],
    [0.0/256.0, 256.0/256.0, 0.0/256.0, 1.0],
    [0.0/256.0, 256.0/256.0, 0.0/256.0, 1.0],
    [0.0/256.0, 256.0/256.0, 0.0/256.0, 1.0],
    [0.0/256.0, 256.0/256.0, 0.0/256.0, 1.0],
    [0.0/256.0, 256.0/256.0, 0.0/256.0, 1.0],
    [0.0/256.0, 256.0/256.0, 0.0/256.0, 1.0],
    [0.0/256.0, 256.0/256.0, 0.0/256.0, 1.0]
], dtype=np.float32)

@ti.data_oriented
class Grid:
    def __init__(self,n_grid,dim,heightmap,scaled_grid_size_m):
        self.km_to_m = 1000.0
        self.grid_size_to_km = heightmap.hm_height_px*heightmap.px_to_km/n_grid
        self.scaled_grid_size_m = scaled_grid_size_m        
        self.scaled_grid_size_km = self.scaled_grid_size_m/self.km_to_m
        self.grid_size_m_to_scaled_grid_size_m = self.scaled_grid_size_m/(self.grid_size_to_km*self.km_to_m)
        self.grid_size_km_to_scaled_grid_size_km = self.scaled_grid_size_km/self.grid_size_to_km
        print(f'self.grid_size_to_km: {self.grid_size_to_km} self.km_to_m: {self.km_to_m}')

        self.cube_positions_dem = ti.Vector.field(dim, ti.f32, 8)
        self.cube_positions_dem.from_numpy(cube_verts_list)
        self.cube_positions_lava1 = ti.Vector.field(dim, ti.f32, 8)
        self.cube_positions_lava1.from_numpy(cube_verts_list)
        self.cube_indices = ti.field(ti.i32, shape=len(cube_faces_list))
        self.cube_indices.from_numpy(cube_faces_list)
        self.cube_colors_dem = ti.Vector.field(4, ti.f32, 8)
        self.cube_colors_dem.from_numpy(cube_colors_list_dem)
        self.cube_positions_lava = ti.Vector.field(dim, ti.f32, 8*(n_grid*n_grid))
        self.cube_indices_lava = ti.field(ti.i32, shape=3*2*6*(n_grid*n_grid))
        self.cube_colors_lava = ti.Vector.field(4, ti.f32, 8*n_grid*n_grid)

        self.colors_lava = ti.Vector.field(4, ti.f32, 5)
        self.colors_lava.from_numpy(colors_lava)

        self.m_transforms_dem = ti.Matrix.field(4,4,dtype=ti.f32,shape=n_grid*n_grid)
        self.m_transforms_lava = ti.Matrix.field(4,4,dtype=ti.f32,shape=n_grid*n_grid)

        self.n_grid = n_grid

        self.neighborListElevDiff = ti.field(ti.f32, shape=(n_grid,n_grid,8))
        self.neighborListRow = ti.field(ti.f32, shape=(n_grid,n_grid,8))
        self.neighborListCol = ti.field(ti.f32, shape=(n_grid,n_grid,8))
        self.neighborListCounter = ti.field(ti.i32, shape=(n_grid,n_grid))

        self.parentcodes = ti.field(ti.i32, (n_grid, ) * (dim-1))
        # north, south, east, west, north-east, north-west, south-east, south-west
        nRows = np.array([+1,-1,+0,+0,+1,+1,-1,-1], dtype=np.int32)
        nCols = np.array([+0,+0,+1,-1,+1,-1,+1,-1], dtype=np.int32)
        neighCodes = np.array([1,2,4,8,16,32,64,128], dtype=np.int32)
        sqrt2 = ti.math.sqrt(2)
        neighDistances = np.array([1.0*self.scaled_grid_size_m,
                                   1.0*self.scaled_grid_size_m,
                                   1.0*self.scaled_grid_size_m,
                                   1.0*self.scaled_grid_size_m,
                                   sqrt2*self.scaled_grid_size_m,
                                   sqrt2*self.scaled_grid_size_m,
                                   sqrt2*self.scaled_grid_size_m,
                                   sqrt2*self.scaled_grid_size_m], dtype=np.float32)

        self.dem_elev = ti.field(ti.f32, (n_grid, ) * (dim-1))
        self.lava_thickness = ti.field(ti.f32, (n_grid, ) * (dim-1))
        self.solid_lava_thickness = ti.field(ti.f32, (n_grid, ) * (dim-1))
        self.heat_quantity = ti.field(ti.f32, (n_grid, ) * (dim-1))
        self.temperature = ti.field(ti.f32, (n_grid, ) * (dim-1))
        self.new_temperature = ti.field(ti.f32, (n_grid, ) * (dim-1))
        self.delta_time = ti.field(ti.f32, (n_grid, ) * (dim-1))
        self.lava_flux = ti.field(ti.f32, shape=(n_grid,n_grid,8))
        self.global_delta_time = ti.field(ti.f32, shape=())
        self.global_volume_lava_erupted_m3 = ti.field(ti.f32, shape=())
        
        self.is_active = ti.field(ti.i32, shape=(n_grid,n_grid))
        self.is_active_ui = ti.field(ti.i32, shape=(n_grid,n_grid))
        self.pulse_volume = ti.field(ti.f32, shape=(n_grid,n_grid))

        self.underground_m = ti.field(ti.f32, shape=())
        self.underground_m[None] = 1000.0
        # Etna's lava parameters
        self.lava_density = ti.field(ti.f32, shape=())
        self.specific_heat_capacity = ti.field(ti.f32, shape=())
        self.emissivity = ti.field(ti.f32, shape=())
        self.cooling_accelerator_factor = ti.field(ti.f32, shape=())
        self.ambient_temperature = ti.field(ti.f32, shape=())
        self.solidification_temperature = ti.field(ti.f32, shape=())
        self.extrusion_temperature = ti.field(ti.f32, shape=())
        self.max_temperature = ti.field(ti.f32, shape=())
        self.H2O = ti.field(ti.f32, shape=())
        self.gravity = ti.field(ti.f32, shape=())
        self.delta_time_c = ti.field(ti.f32, shape=())
        self.cell_area_m = ti.field(ti.f32, shape=())
        self.max_lava_thickness_m = ti.field(ti.f32, shape=())
        self.stefan_boltzmann_constant = ti.field(ti.f32, shape=())

        self.rendering_lava_height_minimum_m = ti.field(ti.f32, shape=())
        self.flux_height_minimum_m = ti.field(ti.f32, shape=())
        self.update_temperature_lava_height_minimum_m = ti.field(ti.f32, shape=())
        self.update_heat_quantity_lava_height_minimum_m = ti.field(ti.f32, shape=())
        # self.delta_total_height_minimum_m = ti.field(ti.f32, shape=())
        self.quality_tolerance = ti.field(ti.f32, shape=())
        self.global_delta_time_maximum_s = ti.field(ti.f32, shape=())

        self.lava_density[None] = 2600.0
        self.specific_heat_capacity[None] = 1150.0
        self.emissivity[None] = 0.9
        self.cooling_accelerator_factor[None] = 0.0
        # self.ambient_temperature = 298.15
        self.ambient_temperature[None] = 400.0
        self.solidification_temperature[None] = 950.0
        self.extrusion_temperature[None] = 1360.0
        self.max_temperature[None] = 2000.0
        self.H2O[None] = 0.06325
        self.gravity[None] = 9.81
        self.delta_time_c[None] = 0.2
        self.cell_area_m[None] = (self.scaled_grid_size_m)**2
        print(f'self.grid_size_to_km: {self.grid_size_to_km} self.cell_area_m: {self.cell_area_m[None]}')
        self.global_volume_lava_erupted_m3[None] = 0.0
        self.c_v = self.specific_heat_capacity
        self.max_lava_thickness_m[None] = 250000.0
        self.stefan_boltzmann_constant[None] = 5.68 * 10**(-8)

        self.rendering_lava_height_minimum_m[None] = 0.0
        self.flux_height_minimum_m[None] = self.scaled_grid_size_m/100.0
        self.update_temperature_lava_height_minimum_m[None] = 0.00
        self.update_heat_quantity_lava_height_minimum_m[None] = 0.00
        self.delta_total_height_min = self.scaled_grid_size_m/7.208175
        self.delta_total_height_max = self.scaled_grid_size_m/5.0
        self.quality_tolerance[None] = 1.0
        self.global_delta_time_maximum_s[None] = 10.0
        self.global_delta_time[None] = self.global_delta_time_maximum_s[None]

        self.temperature_lava_high = self.solidification_temperature[None] + (self.extrusion_temperature[None]-self.solidification_temperature[None])
        self.temperature_lava_high_medium = self.solidification_temperature[None] + (self.extrusion_temperature[None]-self.solidification_temperature[None])*3.0/4.0
        self.temperature_lava_medium = self.solidification_temperature[None] + (self.extrusion_temperature[None]-self.solidification_temperature[None])/2.0 
        self.temperature_lava_low_medium = self.solidification_temperature[None] + (self.extrusion_temperature[None]-self.solidification_temperature[None])/4.0
        self.temperature_lava_low = self.solidification_temperature[None]

        self.min_heatmap_temperature = 950.0
        self.max_heatmap_temperature = 1405.0

        self.nRows = ti.field(ti.i32, 8)
        self.nCols = ti.field(ti.i32, 8)
        self.neighCodes = ti.field(ti.i32, 8)
        self.neighDistances = ti.field(ti.f32, 8)
        self.nRows.from_numpy(nRows)
        self.nCols.from_numpy(nCols)
        self.neighCodes.from_numpy(neighCodes)
        self.neighDistances.from_numpy(neighDistances)


        self.init_values(heightmap)
        self.initialize_m_transforms_dem()
        self.initialize_m_transforms_lava()
        self.initialize_lava_cubes()

    # @ti.kernel
    # def updateDensity(self,value:float):
    #     self.lava_density[None] = value

    @ti.kernel
    def init_values(self,heightmap: ti.template()):
        for i,j in self.dem_elev:
            # self.dem_elev[i,j] = heightmap.heightmap_positions[int((i/self.n_grid+1.0/(2.0*self.n_grid))*heightmap.hm_height_px)*heightmap.hm_width_px+int((j/self.n_grid+1.0/(2.0*self.n_grid))*heightmap.hm_width_px)][1]*self.km_to_m
            self.dem_elev[i,j] = (heightmap.heightmap_positions[i*heightmap.hm_width_px+j][1]*self.km_to_m+self.underground_m[None])*self.grid_size_m_to_scaled_grid_size_m
            self.parentcodes[i,j] = 0
            self.lava_thickness[i,j] = 0.0
            self.solid_lava_thickness[i,j] = 0.0
            self.heat_quantity[i,j] = 0.0
            self.is_active[i,j] = 0
            self.is_active_ui[i,j] = 0
            self.pulse_volume[i,j] = 0.0

            self.temperature[i,j] = self.ambient_temperature[None]
            self.delta_time[i,j] = 0.0

            # if(i==200 and j==200):
            #     self.lava_thickness[i,j] = 50.0
    
    @ti.kernel
    def initialize_m_transforms_dem(self):
        for idx in self.m_transforms_dem:
            i = idx//self.n_grid
            k = idx%self.n_grid
            self.m_transforms_dem[idx] = ti.Matrix.identity(float,4)
            self.m_transforms_dem[idx] *= self.scaled_grid_size_km
            self.m_transforms_dem[idx][1,1] = 1.0
            self.m_transforms_dem[idx][1,1] *= ((self.dem_elev[i,k]+self.solid_lava_thickness[i,k])/self.km_to_m)
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
            self.m_transforms_dem[idx][1,1] = 1.0
            self.m_transforms_dem[idx][1,1] *= ((self.dem_elev[i,k]+self.solid_lava_thickness[i,k])/self.km_to_m)
    
    @ti.kernel
    def calculate_m_transforms_lava(self):
        for idx in self.m_transforms_lava:
            i = idx//self.n_grid
            k = idx%self.n_grid
            lava_thickness_m = self.lava_thickness[i,k]
            if lava_thickness_m > self.rendering_lava_height_minimum_m[None]:
                self.m_transforms_lava[idx][1,1] = 1.0
                self.m_transforms_lava[idx][1,1] *= (self.lava_thickness[i,k]/self.km_to_m)
                self.m_transforms_lava[idx][1,3] = ((self.dem_elev[i,k]+self.solid_lava_thickness[i,k])/self.km_to_m)
            else:
                self.m_transforms_lava[idx][1,3] = 654654654
    
    @ti.kernel
    def initialize_lava_cubes(self):
        for i,k in self.lava_thickness:
            # i = idx//self.n_grid
            # k = idx%self.n_grid
            idx = self.n_grid*i + k
            # cube pos
            pos_idx = 8*idx
            self.cube_positions_lava[pos_idx+0] = ti.Vector([i*self.scaled_grid_size_km-self.scaled_grid_size_km/2.0,0.0,k*self.scaled_grid_size_km-self.scaled_grid_size_km/2.0])
            self.cube_positions_lava[pos_idx+1] = ti.Vector([i*self.scaled_grid_size_km-self.scaled_grid_size_km/2.0,0.0,k*self.scaled_grid_size_km+self.scaled_grid_size_km/2.0])
            self.cube_positions_lava[pos_idx+2] = ti.Vector([i*self.scaled_grid_size_km+self.scaled_grid_size_km/2.0,0.0,k*self.scaled_grid_size_km-self.scaled_grid_size_km/2.0])
            self.cube_positions_lava[pos_idx+3] = ti.Vector([i*self.scaled_grid_size_km+self.scaled_grid_size_km/2.0,0.0,k*self.scaled_grid_size_km+self.scaled_grid_size_km/2.0])
            self.cube_positions_lava[pos_idx+4] = ti.Vector([i*self.scaled_grid_size_km-self.scaled_grid_size_km/2.0,0.0,k*self.scaled_grid_size_km-self.scaled_grid_size_km/2.0])
            self.cube_positions_lava[pos_idx+5] = ti.Vector([i*self.scaled_grid_size_km-self.scaled_grid_size_km/2.0,0.0,k*self.scaled_grid_size_km+self.scaled_grid_size_km/2.0])
            self.cube_positions_lava[pos_idx+6] = ti.Vector([i*self.scaled_grid_size_km+self.scaled_grid_size_km/2.0,0.0,k*self.scaled_grid_size_km-self.scaled_grid_size_km/2.0])
            self.cube_positions_lava[pos_idx+7] = ti.Vector([i*self.scaled_grid_size_km+self.scaled_grid_size_km/2.0,0.0,k*self.scaled_grid_size_km+self.scaled_grid_size_km/2.0])
            # cube indices
            index_idx = 36*idx
            self.cube_indices_lava[index_idx+0],self.cube_indices_lava[index_idx+1],self.cube_indices_lava[index_idx+2],self.cube_indices_lava[index_idx+3],self.cube_indices_lava[index_idx+4],self.cube_indices_lava[index_idx+5] = pos_idx+0,pos_idx+1,pos_idx+2,pos_idx+1,pos_idx+3,pos_idx+2
            self.cube_indices_lava[index_idx+6],self.cube_indices_lava[index_idx+7],self.cube_indices_lava[index_idx+8],self.cube_indices_lava[index_idx+9],self.cube_indices_lava[index_idx+10],self.cube_indices_lava[index_idx+11] = pos_idx+0,pos_idx+4,pos_idx+5,pos_idx+0,pos_idx+5,pos_idx+1
            self.cube_indices_lava[index_idx+12],self.cube_indices_lava[index_idx+13],self.cube_indices_lava[index_idx+14],self.cube_indices_lava[index_idx+15],self.cube_indices_lava[index_idx+16],self.cube_indices_lava[index_idx+17] = pos_idx+0,pos_idx+2,pos_idx+4,pos_idx+2,pos_idx+6,pos_idx+4
            self.cube_indices_lava[index_idx+18],self.cube_indices_lava[index_idx+19],self.cube_indices_lava[index_idx+20],self.cube_indices_lava[index_idx+21],self.cube_indices_lava[index_idx+22],self.cube_indices_lava[index_idx+23] = pos_idx+2,pos_idx+7,pos_idx+6,pos_idx+2,pos_idx+3,pos_idx+7
            self.cube_indices_lava[index_idx+24],self.cube_indices_lava[index_idx+25],self.cube_indices_lava[index_idx+26],self.cube_indices_lava[index_idx+27],self.cube_indices_lava[index_idx+28],self.cube_indices_lava[index_idx+29] = pos_idx+3,pos_idx+5,pos_idx+7,pos_idx+1,pos_idx+5,pos_idx+3
            self.cube_indices_lava[index_idx+30],self.cube_indices_lava[index_idx+31],self.cube_indices_lava[index_idx+32],self.cube_indices_lava[index_idx+33],self.cube_indices_lava[index_idx+34],self.cube_indices_lava[index_idx+35] = pos_idx+4,pos_idx+6,pos_idx+5,pos_idx+5,pos_idx+6,pos_idx+7
            # Color
            self.cube_colors_lava[pos_idx+0] = ti.Vector([1.0,0.0,0.0,1.0])
            self.cube_colors_lava[pos_idx+1] = ti.Vector([1.0,0.0,0.0,1.0])
            self.cube_colors_lava[pos_idx+2] = ti.Vector([1.0,0.0,0.0,1.0])
            self.cube_colors_lava[pos_idx+3] = ti.Vector([1.0,0.0,0.0,1.0])
            self.cube_colors_lava[pos_idx+4] = ti.Vector([1.0,0.0,0.0,1.0])
            self.cube_colors_lava[pos_idx+5] = ti.Vector([1.0,0.0,0.0,1.0])
            self.cube_colors_lava[pos_idx+6] = ti.Vector([1.0,0.0,0.0,1.0])
            self.cube_colors_lava[pos_idx+7] = ti.Vector([1.0,0.0,0.0,1.0])


    @ti.kernel
    def calculate_lava_height_and_color(self):
        for i,k in self.lava_thickness:
            # i = idx//self.n_grid
            # k = idx%self.n_grid
            idx = self.n_grid*i + k
            pos_idx = 8*idx
            lava_thickness_m = self.lava_thickness[i,k]
            if lava_thickness_m > self.rendering_lava_height_minimum_m[None]:
                base_height = (self.dem_elev[i,k]+self.solid_lava_thickness[i,k])/self.km_to_m
                top_height = base_height + lava_thickness_m/self.km_to_m
                # Height pos
                self.cube_positions_lava[pos_idx+0].y = base_height
                self.cube_positions_lava[pos_idx+1].y = base_height
                self.cube_positions_lava[pos_idx+2].y = base_height
                self.cube_positions_lava[pos_idx+3].y = base_height
                self.cube_positions_lava[pos_idx+4].y = top_height
                self.cube_positions_lava[pos_idx+5].y = top_height
                self.cube_positions_lava[pos_idx+6].y = top_height
                self.cube_positions_lava[pos_idx+7].y = top_height
                # Color
                color_heatmap = ti.min(ti.max(self.temperature[i,k],self.min_heatmap_temperature),self.max_heatmap_temperature)
                heatNorm = (4) * (color_heatmap-self.min_heatmap_temperature)/(self.max_heatmap_temperature-self.min_heatmap_temperature)
                # heatNormIdx = len(colors_lava-1) * heatNorm
                heatColorLow = self.colors_lava[int(ti.floor(heatNorm))]
                heatColorHigh = self.colors_lava[int(ti.ceil(heatNorm))]
                
                # heatVal = (heatNorm >= i/(len(colors_lava)-1)) & (heatNorm < (i+1)/(len(colors_lava)-1))
                
                color_temperature_ratio = (color_heatmap-self.min_heatmap_temperature)/(self.max_heatmap_temperature-self.min_heatmap_temperature)
                color_temperature_ratio_2 = ti.math.sqrt(color_temperature_ratio)
                color_temperature = color_temperature_ratio_2*color_lava_high + (1-color_temperature_ratio_2)*color_lava_low
                color_temperature = ti.math.fract(heatNorm)*heatColorHigh + (1-ti.math.fract(heatNorm))*heatColorLow
                self.cube_colors_lava[pos_idx+0] = color_temperature
                self.cube_colors_lava[pos_idx+1] = color_temperature
                self.cube_colors_lava[pos_idx+2] = color_temperature
                self.cube_colors_lava[pos_idx+3] = color_temperature
                self.cube_colors_lava[pos_idx+4] = color_temperature
                self.cube_colors_lava[pos_idx+5] = color_temperature
                self.cube_colors_lava[pos_idx+6] = color_temperature
                self.cube_colors_lava[pos_idx+7] = color_temperature
            else:
                # Height pos
                self.cube_positions_lava[pos_idx+0].y = 654654654
                self.cube_positions_lava[pos_idx+1].y = 654654654
                self.cube_positions_lava[pos_idx+2].y = 654654654
                self.cube_positions_lava[pos_idx+3].y = 654654654
                self.cube_positions_lava[pos_idx+4].y = 654654654
                self.cube_positions_lava[pos_idx+5].y = 654654654
                self.cube_positions_lava[pos_idx+6].y = 654654654
                self.cube_positions_lava[pos_idx+7].y = 654654654

    @ti.kernel
    def computeFluxTransfers(self):
        for i,k,n in self.lava_flux:
            cur_cell_total_height = self.dem_elev[i,k] + self.solid_lava_thickness[i,k]
            i_n,k_n = int(i+self.nRows[n]), int(k+self.nCols[n])
            if(i_n < 0 or k_n < 0 or i_n > self.n_grid or k_n > self.n_grid):
                self.lava_flux[i,k,n] = 0.0
            else:
                neigh_cell_total_height = self.dem_elev[i_n,k_n] + self.solid_lava_thickness[i_n,k_n]

                delta_z = neigh_cell_total_height - cur_cell_total_height
                delta_h = self.lava_thickness[i_n,k_n] - self.lava_thickness[i,k]
                h = self.lava_thickness[i,k]
                T = self.temperature[i,k]
                delta_x_sign = -1
                if((delta_z+delta_h) > 0):
                    h = self.lava_thickness[i_n,k_n]
                    T = self.temperature[i_n,k_n]
                    delta_x_sign = 1
                
                # if (h<=0):
                delta_total_height_minimum_m = (self.quality_tolerance[None]-1)/9.0*(self.delta_total_height_max-self.delta_total_height_min)+self.delta_total_height_min
                if (h<=self.flux_height_minimum_m[None] or ti.abs(delta_z+delta_h) < delta_total_height_minimum_m):
                    self.lava_flux[i,k,n] = 0.0
                else:
                    rho = self.lava_density[None]
                    g = self.gravity[None]
                    delta_x = delta_x_sign * self.neighDistances[n]
                    S_y = 10.0**(13.00997 - 0.0089*T)
                    eta = 10.0**(-4.643 + (5812.44 - 427.04*self.H2O[None])/(T - 499.31 + 28.74*ti.log(self.H2O[None])))
                    h_cr = S_y * (ti.math.sqrt(delta_z**2 + delta_x**2)) / (rho*g*delta_x_sign*(delta_z+delta_h))
                    a = h/h_cr
                    if(h>h_cr):
                        q = (S_y * h_cr**2 * delta_x)/(3.0*eta) * (a**3 - 3.0/2.0*a**2 + 1.0/2.0)
                        # print(f'q: {q}')
                        if(ti.math.isinf(q) or ti.math.isnan(q)):
                            print(f'i,k: {i},{k} i_n,k_n: {i_n},{k_n} S_y: {S_y} T: {T} h: {h} h_cr: {h_cr} eta: {eta} a: {a}')
                        self.lava_flux[i,k,n] = q
                        # print(f'h_cr: {h_cr} h: {h} delta_x: {delta_x} delta_h: {delta_h} a: {a} q: {q} i: {i} k: {k} i_n: {i_n} k_n: {k_n}')
                    else:
                        self.lava_flux[i,k,n] = 0.0
        
            # if(i==200 and k==200 and n==0):
            #     print(f'self.lava_density: {self.lava_density[None]}')
    
    @ti.kernel
    def computeTimeSteps(self):
        for i,k in self.dem_elev:
            c = self.delta_time_c[None]
            h = self.lava_thickness[i,k]
            A = self.cell_area_m[None]
            q_tot = 0.0
            for n in ti.static(range(8)):
                q_tot += self.lava_flux[i,k,n]
            if q_tot<0.0 and h>self.flux_height_minimum_m[None]:
                self.delta_time[i,k] = c*h*A/ti.abs(q_tot)
            else:
                self.delta_time[i,k] = self.global_delta_time_maximum_s[None]
            # if((i < 50 or k < 50) and self.delta_time[i,k] != self.global_delta_time_maximum_s[None]):
            #     print(f'i: {i} k: {k} self.delta_time[i,k]: {self.delta_time[i,k]}')
            # if(self.delta_time[i,k] < 1e-5):
            #     print(f'i: {i} k: {k} self.delta_time[i,k]: {self.delta_time[i,k]}')
    
    @ti.kernel
    def computeGlobalTimeStep(self) -> ti.f32:
        global_delta_time = self.global_delta_time_maximum_s[None]
        for i,k in self.dem_elev:
            ti.atomic_min(global_delta_time, self.delta_time[i,k])
        return global_delta_time
    
    @ti.kernel
    def computeNewLavaThickness(self):
        for i,k in self.dem_elev:
            q_tot = 0.0
            for n in ti.static(range(8)):
                q_tot += self.lava_flux[i,k,n]
            delta_lava_thickness = q_tot*self.global_delta_time[None]/self.cell_area_m[None]
            self.lava_thickness[i,k] += delta_lava_thickness
            if(self.lava_thickness[i,k]<=0):
                self.lava_thickness[i,k] = 0.0
                # self.temperature[i,k] = self.ambient_temperature[None]
            # else:
            #     rho = self.lava_density[None]
            #     c_v = self.c_v[None]
            #     h_t_dt = self.lava_thickness[i,k]
            #     A = self.cell_area_m[None]
            #     curr_temperature = self.heat_quantity[i,k] / (rho * c_v * h_t_dt * A)
            #     self.temperature[i,k] = ti.max(self.ambient_temperature[None],ti.min(curr_temperature,self.max_temperature[None]))
            if(ti.math.isnan(delta_lava_thickness)):
                print(f'i: {i} k: {k} q_tot:{q_tot} delta_lava_thickness: {delta_lava_thickness} self.delta_time[i,k]: {self.delta_time[i,k]} self.lava_thickness[i,k]: {self.lava_thickness[i,k]}')
                print(f'i: {i} k: {k} self.lava_flux[i,k,0]: {self.lava_flux[i,k,0]} self.lava_flux[i,k,1]: {self.lava_flux[i,k,1]} self.lava_flux[i,k,2]: {self.lava_flux[i,k,2]} self.lava_flux[i,k,3]: {self.lava_flux[i,k,3]} self.lava_flux[i,k,4]: {self.lava_flux[i,k,4]} self.lava_flux[i,k,5]: {self.lava_flux[i,k,5]} self.lava_flux[i,k,6]: {self.lava_flux[i,k,6]} self.lava_flux[i,k,7]: {self.lava_flux[i,k,7]}')
    
    @ti.kernel
    def computeHeatRadiationLoss(self):
        for i,k in self.dem_elev:
            delta_Q_t_m = 0.0
            for n in ti.static(range(8)):
                i_n,k_n = int(i+self.nRows[n]), int(k+self.nCols[n])
                q_i = self.lava_flux[i,k,n]
                if(q_i>0):
                    delta_Q_t_m += q_i*self.temperature[i_n,k_n]
                else:
                    delta_Q_t_m += q_i*self.temperature[i,k]
            rho = self.lava_density[None]
            c_v = self.c_v[None]
            delta_Q_t_m *= rho * c_v * self.global_delta_time[None]
            
            epsilon = self.emissivity[None]
            cooling_factor = self.cooling_accelerator_factor[None]
            A = self.cell_area_m[None]
            # Stefan–Boltzmann
            sigma = self.stefan_boltzmann_constant[None]
            delta_Q_t_r = 0.0
            if self.lava_thickness[i,k] > self.update_heat_quantity_lava_height_minimum_m[None]:
                delta_Q_t_r = epsilon * A * sigma * self.temperature[i,k]**4 * self.global_delta_time[None] * 1.8**cooling_factor

            self.heat_quantity[i,k] += delta_Q_t_m - delta_Q_t_r
            
            if(self.heat_quantity[i,k]<0):
                self.heat_quantity[i,k] = 0.0
                self.new_temperature[i,k] = self.ambient_temperature[None]
            else:
                 # rho = self.lava_density[None]
                # c_v = self.c_v[None]
                h_t_dt = self.lava_thickness[i,k]
                # A = self.cell_area_m[None]
                curr_temperature = self.heat_quantity[i,k] / (rho * c_v * h_t_dt * A)
                self.new_temperature[i,k] = ti.max(self.ambient_temperature[None],ti.min(curr_temperature,self.max_temperature[None]))
            # self.heat_quantity[i,k] += delta_Q_t_m
            # if(i==201 and k==201):
            #     print(f'self.heat_quantity[i,k]: {self.heat_quantity[i,k]} delta_Q_t_m: {delta_Q_t_m} delta_Q_t_r: {delta_Q_t_r} global_delta_time: {global_delta_time}')
    
    @ti.kernel
    def updateTemperature(self):
        for i,k in self.dem_elev:
            self.temperature[i,k] = self.new_temperature[i,k]
    
    @ti.kernel
    def computeLavaSolidification(self):
        for i,k in self.temperature:
            if (self.temperature[i,k] < self.solidification_temperature[None]):
                epsilon = self.emissivity[None]
                sigma = self.stefan_boltzmann_constant[None]
                rho = self.lava_density[None]
                c_v = self.c_v[None]
                new_solid_lava = (epsilon * sigma * self.solidification_temperature[None]**3 * self.global_delta_time[None]) / (rho * c_v)
                if(self.lava_thickness[i,k] > new_solid_lava):
                    self.solid_lava_thickness[i,k] += new_solid_lava
                    self.lava_thickness[i,k] -= new_solid_lava
                    # rho = self.lava_density[None]
                    # c_v = self.c_v[None]
                    h_t_dt = self.lava_thickness[i,k]
                    A = self.cell_area_m[None]
                    curr_temperature = self.heat_quantity[i,k] / (rho * c_v * h_t_dt * A)
                    self.temperature[i,k] = ti.max(self.ambient_temperature[None],ti.min(curr_temperature,self.max_temperature[None]))
                else:
                    self.solid_lava_thickness[i,k] += self.lava_thickness[i,k]
                    self.lava_thickness[i,k] = 0.0
                    self.temperature[i,k] = self.ambient_temperature[None]
    
    # @ti.kernel
    # def updateTemperature(self):
    #     for i,k in self.temperature:
    #         if(self.heat_quantity[i,k]>0 and self.lava_thickness[i,k] > self.update_temperature_lava_height_minimum_m[None]):
    #             rho = self.lava_density[None]
    #             c_v = self.c_v[None]
    #             h_t_dt = self.lava_thickness[i,k]
    #             A = self.cell_area_m[None]
    #             curr_temperature = self.heat_quantity[i,k] / (rho * c_v * h_t_dt * A)
    #             if(curr_temperature > self.extrusion_temperature[None]):
    #                 self.heat_quantity[i,k] = self.extrusion_temperature[None] * (rho * c_v * h_t_dt * A)
    #             elif(curr_temperature < self.ambient_temperature[None]):
    #                 self.heat_quantity[i,k] = self.ambient_temperature[None] * (rho * c_v * h_t_dt * A)
    #             self.temperature[i,k] = ti.max(self.ambient_temperature[None],ti.min(curr_temperature,self.max_temperature[None]))
                # self.temperature[i,k] = curr_temperature
            # else:
            #     self.temperature[i,k] = 0.0
            #     self.heat_quantity[i,k] = 0.0
            # if(i==201 and k==201):
            #     print(f'self.temperature[i,k]: {self.temperature[i,k]}')

        
    
    @ti.kernel
    def pulse(self):
        for i,k in self.dem_elev:
            if(self.is_active[i,k]>0 and self.lava_thickness[i,k] < self.max_lava_thickness_m[None]):
                # if(i==200 and k==200):
                #     print(f'self.is_active[i,k]: {self.is_active[i,k]}')
                pulse_volume_m3 = self.pulse_volume[i,k]
                if(not self.is_active_ui[i,k]):
                    pulse_volume_m3 *= self.global_delta_time[None]
                pulseThickness = pulse_volume_m3 / self.cell_area_m[None]
                new_lava_thickness = self.lava_thickness[i,k] + pulseThickness
                if (new_lava_thickness > self.max_lava_thickness_m[None]):
                    pulseThickness = self.max_lava_thickness_m[None] - self.lava_thickness[i,k]
                    pulse_volume_m3 = pulseThickness * self.cell_area_m[None]
                self.lava_thickness[i,k] += pulseThickness
                ti.atomic_add(self.global_volume_lava_erupted_m3[None], pulse_volume_m3/(self.grid_size_m_to_scaled_grid_size_m**3))
                self.is_active[i,k] -= 1
                # self.pulse_volume[i,k] = 0.0
                if(pulseThickness>0):
                    # print(f'i,k: {i,k} self.lava_thickness[i,k]: {self.lava_thickness[i,k]}')
                    rho = self.lava_density[None]
                    c_v = self.c_v[None]
                    A = self.cell_area_m[None]
                    h_t_dt = self.lava_thickness[i,k]
                    self.heat_quantity[i,k] += pulseThickness*A*self.extrusion_temperature[None]*rho*c_v
                    curr_temperature = self.heat_quantity[i,k]/(rho*c_v*h_t_dt*A)
                    self.temperature[i,k] = ti.max(self.ambient_temperature[None],ti.min(curr_temperature,self.max_temperature[None]))
            elif(self.is_active[i,k]<0 and self.lava_thickness[i,k] > 0.0):
                pulse_volume_m3 = self.pulse_volume[i,k]
                pulseThickness = pulse_volume_m3 / self.cell_area_m[None]
                new_lava_thickness = self.lava_thickness[i,k] - pulseThickness
                if (new_lava_thickness < 0):
                    pulseThickness = self.lava_thickness[i,k]
                    pulse_volume_m3 = pulseThickness * self.cell_area_m[None]
                self.lava_thickness[i,k] -= pulseThickness
                ti.atomic_sub(self.global_volume_lava_erupted_m3[None], pulse_volume_m3/(self.grid_size_m_to_scaled_grid_size_m**3))
                self.is_active[i,k] += 1
                # self.pulse_volume[i,k] = 0.0
                if(pulseThickness>0):
                    rho = self.lava_density[None]
                    c_v = self.c_v[None]
                    A = self.cell_area_m[None]
                    h_t_dt = self.lava_thickness[i,k]
                    self.heat_quantity[i,k] -= pulseThickness*A*self.extrusion_temperature[None]*rho*c_v
                    curr_temperature = self.heat_quantity[i,k]/(rho*c_v*h_t_dt*A)
                    self.temperature[i,k] = ti.max(self.ambient_temperature[None],ti.min(curr_temperature,self.max_temperature[None]))


    def bboxIntersect(self,rayPosition,rayDirection):
        epsilon = 1.0e-5
        tmin = -1e-16
        tmax = 1e16

        a = [0,0]
        b = [self.n_grid, self.n_grid]
        p = rayPosition
        d = rayDirection

        t = 0.0
        # Ox
        if (d[0] < -epsilon):
            t = (a[0] - p[0]) / d[0]
            if (t < tmin):
                return False,0.0,0.0
            if (t <= tmax):
                tmax = t
            t = (b[0] - p[0]) / d[0]
            if (t >= tmin):
                if(t > tmax):
                    return False,0.0,0.0
                tmin = t
        elif (d[0] > epsilon):
            t = (b[0] - p[0]) / d[0]
            if (t < tmin):
                return False,0.0,0.0
            if (t <= tmax):
                tmax = t
            t = (a[0] - p[0]) / d[0]
            if (t >= tmin):
                if(t > tmax):
                    return False,0.0,0.0
                tmin = t
        elif (p[0]<a[0] or p[0]>b[0]):
            return False,0.0,0.0
        
        # Oy
        if (d[1] < -epsilon):
            t = (a[1] - p[1]) / d[1]
            if (t < tmin):
                return False,0.0,0.0
            if (t <= tmax):
                tmax = t
            t = (b[1] - p[1]) / d[1]
            if (t >= tmin):
                if(t > tmax):
                    return False,0.0,0.0
                tmin = t
        elif (d[1] > epsilon):
            t = (b[1] - p[1]) / d[1]
            if (t < tmin):
                return False,0.0,0.0
            if (t <= tmax):
                tmax = t
            t = (a[1] - p[1]) / d[1]
            if (t >= tmin):
                if(t > tmax):
                    return False,0.0,0.0
                tmin = t
        elif (p[1]<a[1] or p[1]>b[1]):
            return False,0.0,0.0
        
        return True,tmin,tmax

    def Intersect(self,rayPosition,rayDirection):
        rayPosList = [rayPosition[0],rayPosition[2]]
        rayPositionGrid = np.array(rayPosList)/self.scaled_grid_size_km
        rayDirGridList = [rayDirection[0],rayDirection[2]]
        rayDirectionGrid = np.array(rayDirGridList)
        # print(f'rayPositionGrid: {rayPositionGrid} rayDirectionGrid: {rayDirectionGrid}')
        # Check the intersection with the bounding box
        isBboxIntersected,ta,tb = self.bboxIntersect(rayPositionGrid,rayDirectionGrid)
        # print(f'isBboxIntersected: {isBboxIntersected} ta: {ta} tb: {tb}')
        if(isBboxIntersected):
            # Ray marching
            t = ta + 0.0001
            if (ta < 0.0):
                t = 0.0
            
            while (t < tb):
                #  Point along the ray
                p = rayPosition/self.scaled_grid_size_km + t*rayDirection
                # print(f'p_curr: {p} p[1]*self.km_to_m: {p[1]*self.km_to_m}')
                h = self.dem_elev[int(p[0]),int(p[2])] + self.lava_thickness[int(p[0]),int(p[2])] + self.solid_lava_thickness[int(p[0]),int(p[2])]
                # print(f'curr_p: {p} h: {h} p[1]*self.grid_size_to_km*self.km_to_m: {p[1]*self.grid_size_to_km*self.km_to_m}')
                pos_height_meters = p[1]*self.scaled_grid_size_km*self.km_to_m
                if (h > pos_height_meters):
                    p[1] = h/(self.scaled_grid_size_km*self.km_to_m)
                    # # print(f'p[0]: {p[0]} p[2]: {p[2]} h: {h} p[1]*self.km_to_m: {p[1]*self.km_to_m} p: {p}')
                    # if(pos_height_meters>0.0):
                    #     return True,p*self.grid_size_to_km
                    # else:
                    #     p[1] = h/(self.grid_size_to_km*self.km_to_m)
                    return True,p*self.scaled_grid_size_km
                else:
                    t += 1.0
            
        return False,rayPosition*9999