import taichi as ti
import numpy as np
import Common.Constants as Constants

@ti.data_oriented
class Grid:
    def __init__(self,n_grid,dim,heightmap,scaled_grid_size_m):
        self.km_to_m = 1000.0
        self.grid_size_to_km = heightmap.hm_height_px*heightmap.px_to_km/n_grid
        self.scaled_grid_size_m = scaled_grid_size_m        
        self.scaled_grid_size_km = self.scaled_grid_size_m/self.km_to_m
        self.grid_size_m_to_scaled_grid_size_m = self.scaled_grid_size_m/(self.grid_size_to_km*self.km_to_m)
        self.grid_size_km_to_scaled_grid_size_km = self.scaled_grid_size_km/self.grid_size_to_km

        self.cube_positions_dem = ti.Vector.field(dim, ti.f32, 8)
        self.cube_positions_dem.from_numpy(Constants.cube_verts_list)
        self.cube_positions_lava = ti.Vector.field(dim, ti.f32, 8)
        self.cube_positions_lava.from_numpy(Constants.cube_verts_list)
        self.cube_indices = ti.field(ti.i32, shape=len(Constants.cube_faces_list))
        self.cube_indices.from_numpy(Constants.cube_faces_list)
        self.cube_colors_dem = ti.Vector.field(4, ti.f32, 8)
        self.cube_colors_dem.from_numpy(Constants.cube_colors_list_dem)

        self.m_transforms_dem = ti.Matrix.field(4,4,dtype=ti.f32,shape=n_grid*n_grid)
        self.m_transforms_lava = ti.Matrix.field(4,4,dtype=ti.f32,shape=n_grid*n_grid)

        self.n_grid = n_grid
        
        self.is_active = ti.field(ti.i32, shape=(n_grid,n_grid))
        self.is_active_ui = ti.field(ti.i32, shape=(n_grid,n_grid)) # Param for UI
        self.pulse_volume = ti.field(ti.f32, shape=(n_grid,n_grid))

        self.underground_m = ti.field(ti.f32, shape=())
        self.underground_m[None] = 1000.0

        # MAGFLOW Params
        # Etna's lava parameters
        # self.lava_density = ti.field(ti.f32, shape=())
        # self.specific_heat_capacity = ti.field(ti.f32, shape=())
        # self.emissivity = ti.field(ti.f32, shape=())
        # self.cooling_accelerator_factor = ti.field(ti.f32, shape=())
        # self.ambient_temperature = ti.field(ti.f32, shape=())
        # self.solidification_temperature = ti.field(ti.f32, shape=())
        # self.extrusion_temperature = ti.field(ti.f32, shape=())
        # self.max_temperature = ti.field(ti.f32, shape=())
        # self.H2O = ti.field(ti.f32, shape=())
        # self.gravity = ti.field(ti.f32, shape=())
        # self.delta_time_c = ti.field(ti.f32, shape=())
        # self.cell_area_m = ti.field(ti.f32, shape=())
        # self.max_lava_thickness_m = ti.field(ti.f32, shape=())
        # self.stefan_boltzmann_constant = ti.field(ti.f32, shape=())

        # self.rendering_lava_height_minimum_m = ti.field(ti.f32, shape=())
        # self.flux_height_minimum_m = ti.field(ti.f32, shape=())
        # self.update_temperature_lava_height_minimum_m = ti.field(ti.f32, shape=())
        # self.update_heat_quantity_lava_height_minimum_m = ti.field(ti.f32, shape=())
        # # self.delta_total_height_minimum_m = ti.field(ti.f32, shape=())
        # self.quality_tolerance = ti.field(ti.f32, shape=())
        # self.global_delta_time_maximum_s = ti.field(ti.f32, shape=())

        # self.lava_density[None] = 2600.0
        # self.specific_heat_capacity[None] = 1150.0
        # self.emissivity[None] = 0.9
        # self.cooling_accelerator_factor[None] = 3.0
        # # self.ambient_temperature = 298.15
        # self.ambient_temperature[None] = 400.0
        # self.solidification_temperature[None] = 1173.0
        # self.extrusion_temperature[None] = 1360.0
        # self.max_temperature[None] = 2000.0
        # self.H2O[None] = 0.06325
        # self.gravity[None] = 9.81
        # self.delta_time_c[None] = 0.2
        # self.cell_area_m[None] = (self.scaled_grid_size_m)**2
        # print(f'self.grid_size_to_km: {self.grid_size_to_km} self.cell_area_m: {self.cell_area_m[None]}')
        # self.global_volume_lava_erupted_m3[None] = 0.0
        # self.c_v = self.specific_heat_capacity
        # self.max_lava_thickness_m[None] = 250000.0
        # self.stefan_boltzmann_constant[None] = 5.68 * 10**(-8)

        # self.rendering_lava_height_minimum_m[None] = 0.0
        # self.flux_height_minimum_m[None] = self.scaled_grid_size_m/100.0
        # self.update_temperature_lava_height_minimum_m[None] = 0.00
        # self.update_heat_quantity_lava_height_minimum_m[None] = 0.00
        # self.delta_total_height_min = self.scaled_grid_size_m/7.208175
        # self.delta_total_height_max = self.scaled_grid_size_m/5.0
        # self.quality_tolerance[None] = 1.0
        # self.global_delta_time_maximum_s[None] = 10.0
        # self.global_delta_time[None] = self.global_delta_time_maximum_s[None]

        # self.temperature_lava_high = self.solidification_temperature[None] + (self.extrusion_temperature[None]-self.solidification_temperature[None])
        # self.temperature_lava_high_medium = self.solidification_temperature[None] + (self.extrusion_temperature[None]-self.solidification_temperature[None])*3.0/4.0
        # self.temperature_lava_medium = self.solidification_temperature[None] + (self.extrusion_temperature[None]-self.solidification_temperature[None])/2.0 
        # self.temperature_lava_low_medium = self.solidification_temperature[None] + (self.extrusion_temperature[None]-self.solidification_temperature[None])/4.0
        # self.temperature_lava_low = self.solidification_temperature[None]

        # self.min_heatmap_temperature = 950.0
        # self.max_heatmap_temperature = 1405.0

        # self.neighDistances.from_numpy(neighDistances)


        self.init_values(heightmap)
        self.initialize_m_transforms_dem()
        self.initialize_m_transforms_lava()
        self.initialize_lava_cubes()