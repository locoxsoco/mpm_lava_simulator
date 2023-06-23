import math
import numpy as np
import scipy.stats as st
from MAGFLOW.initialize import initialize, Vent
from MAGFLOW.magflow_input_file import configureParams
from MAGFLOW.heightmap import Heightmap
from MAGFLOW.grid import Grid
from enum import Enum

np.random.seed(42)
max_brush_strength_factor = 5.0

class PulseFileStatus(Enum):
    INACTIVE = 0
    ACTIVE = 1
    END = 2

def gennor(av,sd):
    return sd*np.random.normal() + av

def genunf(low,high):
    return low + (high-low)*np.random.random()

def cubicSmooth(x,r):
    return (1.0-x/r)*(1.0-x/r)*(1.0-x/r)

def gkern(kernlen=5, nsig=2.5):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()

class ActiveList:
    def __init__(self):
        self.row = 0
        self.col = 0
        self.excess = 0

class Driver:
    def __init__(self,heightmap_path,dim,hm_elev_min_m,hm_elev_max_m,n_grid):
        # Initialize variables with values from the config file
        self.active_flow, self.inParams, self.outParams = initialize()
        configureParams(self.inParams, self.outParams)
        self.load_vent_data()
        self.km_to_m = 1000.0
        # Read in the DEM using the gdal library
        self.Heightmap = Heightmap(heightmap_path,dim,hm_elev_min_m,hm_elev_max_m)
        self.grid_size_to_km = self.Heightmap.hm_height_px*self.Heightmap.px_to_km/n_grid
        self.scaled_grid_size_m = self.grid_size_to_km*self.km_to_m
        self.scaled_grid_size_m = 5.0
        self.grid_size_m_to_scaled_grid_size_m = self.scaled_grid_size_m/(self.grid_size_to_km*self.km_to_m)
        self.Grid = Grid(n_grid,dim,self.Heightmap,self.scaled_grid_size_m)
        self.set_flow_params()
        # Initialize the lava flow data structures and initialize vent cell
        self.CAList = self.init_flow()
        self.CAListSize = 0
        self.current_vent = -1
        self.pulseCount = 0
        self.ActiveCounter = 0

        # Read pulse txt file
        self.pulse_file_init_time = [0.0,600.0,1200.0,1800.0,2400.0,3000.0,3600.0,4200.0,4800.0,5400.0,6000.0,6600.0]
        self.pulse_file_end_time = [600.0,1200.0,1800.0,2400.0,3000.0,3600.0,4200.0,4800.0,5400.0,6000.0,6600.0,7200.0]
        # Mount Etna, Italy (2001)
        # # Case IV
        # self.pulse_file_volume_m3_per_s = [
        #     56.25,
        #     168.75,
        #     281.25,
        #     393.75,
        #     421.875,
        #     365.625,
        #     309.375,
        #     253.125,
        #     196.875,
        #     140.625,
        #     84.375,
        #     28.125
        # ]
        # # Case VII
        # self.pulse_file_volume_m3_per_s = [
        #     112.5,
        #     337.5,
        #     562.5,
        #     787.5,
        #     843.75,
        #     731.25,
        #     618.75,
        #     506.25,
        #     393.75,
        #     281.25,
        #     168.75,
        #     56.25
        # ]
        # # Case IV * 9*9*9
        # self.pulse_file_volume_m3_per_s = [
        #     41006.25,
        #     123018.75,
        #     205031.25,
        #     287043.75,
        #     307546.875,
        #     266540.625,
        #     225534.375,
        #     184528.125,
        #     143521.875,
        #     102515.625,
        #     61509.375,
        #     20503.125
        # ]
        # Custom case
        # self.pulse_file_volume_m3_per_s = [20503.125, 61509.375, 102515.625, 143521.875, 153773.4375, 133270.3125, 112767.1875, 92264.0625, 71760.9375, 51257.8125, 30754.6875, 10251.5625]
        self.pulse_file_volume_m3_per_s = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        self.pulse_file_radius = [5,5,6,6,6,7,7,7,7,7,8,8]
        self.pulse_file_vent_x = [200,200,200,200,200,200,200,200,200,200,200,200]
        self.pulse_file_vent_y = [200,200,200,200,200,200,200,200,200,200,200,200]
        self.pulse_file_len = 12
        self.pulse_file_index = 0
        self.pulse_file_status = PulseFileStatus.INACTIVE
        # Generate Gaussian filters
        self.pulse_file_gaussian_filters = [
            gkern(self.pulse_file_radius[0]*2+1),
            gkern(self.pulse_file_radius[1]*2+1),
            gkern(self.pulse_file_radius[2]*2+1),
            gkern(self.pulse_file_radius[3]*2+1),
            gkern(self.pulse_file_radius[4]*2+1),
            gkern(self.pulse_file_radius[5]*2+1),
            gkern(self.pulse_file_radius[6]*2+1),
            gkern(self.pulse_file_radius[7]*2+1),
            gkern(self.pulse_file_radius[8]*2+1),
            gkern(self.pulse_file_radius[9]*2+1),
            gkern(self.pulse_file_radius[10]*2+1),
            gkern(self.pulse_file_radius[11]*2+1)
        ]

        self.n_steps = 0
        self.time = 0.0
    
    def load_vent_data(self):
        self.active_flow.num_vents = 1
        self.active_flow.source = []
        for _ in range(self.active_flow.num_vents):
            self.active_flow.source.append(Vent())
        self.active_flow.source[0].easting = 20.0
        self.active_flow.source[0].northing = 20.0
        self.active_flow.source[0].easting = 18.52
        self.active_flow.source[0].northing = 18.52
        print(f'self.active_flow.source[0].easting: {self.active_flow.source[0].easting} self.active_flow.source[0].now: {self.active_flow.source[0].row}')
    
    def set_flow_params(self):        
        print(f'self.inParams.min_total_volume: {self.inParams.min_total_volume} self.inParams.max_total_volume: {self.inParams.max_total_volume}')
        if(self.inParams.min_total_volume > 0 and self.inParams.max_total_volume > 0 and self.inParams.max_total_volume >= self.inParams.min_total_volume):
            if (self.inParams.log_mean_volume > 0 and self.inParams.log_std_volume > 0):
                log_min = math.log(self.inParams.min_total_volume,10)
                log_max = math.log(self.inParams.max_total_volume,10)
                self.active_flow.volumeToErupt = gennor(self.inParams.log_mean_volume, self.inParams.log_std_volume)
                print(f'self.active_flow.volumeToErupt: {self.active_flow.volumeToErupt}')
                while(self.active_flow.volumeToErupt > log_max or self.active_flow.volumeToErupt < log_min):
                    self.active_flow.volumeToErupt = gennor(self.inParams.log_mean_volume, self.inParams.log_std_volume)
                self.active_flow.volumeToErupt = math.pow(self.active_flow.volumeToErupt,10)
            else:
                self.active_flow.volumeToErupt = genunf(self.inParams.min_total_volume, self.inParams.max_total_volume)
            self.active_flow.currentvolume = self.active_flow.volumeToErupt
        
        print(f'self.inParams.min_pulse_volume: {self.inParams.min_pulse_volume} self.inParams.max_pulse_volume: {self.inParams.max_pulse_volume}')
        if(self.inParams.min_pulse_volume > 0 and self.inParams.max_pulse_volume > 0 and self.inParams.max_pulse_volume >= self.inParams.min_pulse_volume):
            self.active_flow.pulsevolume = genunf(self.inParams.min_pulse_volume, self.inParams.max_pulse_volume)
    
    def init_flow(self):
        maxCellsPossible = self.Grid.n_grid * self.Grid.n_grid
        local_CAList = None
        if(maxCellsPossible > math.pow(10,6)):
            maxCellsPossible = math.pow(10,6)
        self.CAListSize = maxCellsPossible
        local_CAList = []
        for _ in range(self.CAListSize): 
            local_CAList.append(ActiveList())
        if (local_CAList == None):
            return None
        print("Allocating Memory for Active Cell List, size = %u ", self.CAListSize)

        #  Do not put vent(s) on active list 
        # Initialize vents, assign vent its grid location
        for i in range(self.active_flow.num_vents):
            self.active_flow.source[i].row = int(self.active_flow.source[i].northing/self.Grid.grid_size_to_km)   # Row (Y) of vent cell
            self.active_flow.source[i].col = int(self.active_flow.source[i].easting/self.Grid.grid_size_to_km)   # Col (X) of vent cell

        return local_CAList

    def set_active_pulses(self,center_x,center_y,radius,active_value,brush_strength):
        pulse_m3_per_s = 10.0*brush_strength/max_brush_strength_factor
        radius_grid = math.floor(radius/self.Grid.grid_size_to_km)
        bbox_min_x = center_x - radius_grid
        bbox_min_y = center_y - radius_grid
        bbox_max_x = center_x + radius_grid
        bbox_max_y = center_y + radius_grid

        for y in range(bbox_min_y,bbox_max_y):
            for x in range(bbox_min_x,bbox_max_x):
                u = (center_x-x)**2 + (center_y-y)**2
                if(u<radius_grid*radius_grid):
                    self.Grid.is_active[x,y] = active_value
                    self.Grid.is_active_ui[x,y] = 1
                    self.Grid.pulse_volume[x,y] += pulse_m3_per_s * cubicSmooth(u,radius_grid*radius_grid)
    
    def set_active_pulses_gaussian_kernel(self,center_x,center_y,radius,active_value: int):
        pulse_m3_per_s = self.pulse_file_volume_m3_per_s[self.pulse_file_index]
        bbox_min_x = center_x - radius
        bbox_min_y = center_y - radius
        bbox_max_x = center_x + radius
        bbox_max_y = center_y + radius

        for index_y, y in enumerate(range(bbox_min_y,bbox_max_y),start=0):
            for index_x, x in enumerate(range(bbox_min_x,bbox_max_x),start=0):
                self.Grid.is_active[x,y] = active_value
                self.Grid.is_active_ui[x,y] = 0
                # print(f'pulse_file_index: {self.pulse_file_index} index_x: {index_x} index_y: {index_y} x: {x} y: {y}')
                # print(f'self.pulse_file_gaussian_filters[self.pulse_file_index]: {self.pulse_file_gaussian_filters[self.pulse_file_index]}')
                self.Grid.pulse_volume[x,y] = pulse_m3_per_s*self.grid_size_m_to_scaled_grid_size_m**3 * self.pulse_file_gaussian_filters[self.pulse_file_index][index_x][index_y]
        
    def set_active_pulses_gaussian_kernel_upc(self,center_x,center_y,radius,active_value: int):
        pulse_m3_per_s = 20503000.125
        bbox_min_x = center_x - radius
        bbox_min_y = center_y - radius
        bbox_max_x = center_x + radius
        bbox_max_y = center_y + radius

        for index_y, y in enumerate(range(bbox_min_y,bbox_max_y),start=0):
            for index_x, x in enumerate(range(bbox_min_x,bbox_max_x),start=0):
                self.Grid.is_active[x,y] = active_value
                self.Grid.is_active_ui[x,y] = 0
                # print(f'pulse_file_index: {self.pulse_file_index} index_x: {index_x} index_y: {index_y} x: {x} y: {y}')
                # print(f'self.pulse_file_gaussian_filters[self.pulse_file_index]: {self.pulse_file_gaussian_filters[self.pulse_file_index]}')
                self.Grid.pulse_volume[x,y] = pulse_m3_per_s*self.grid_size_m_to_scaled_grid_size_m**3 * self.pulse_file_gaussian_filters[0][index_x][index_y]
    
    def set_active_pulses_file(self,simulation_time,substeps):
        if(self.pulse_file_status != PulseFileStatus.END):
            if(simulation_time >= self.pulse_file_end_time[self.pulse_file_index]):
                self.pulse_file_index += 1

                # print(f'self.pulse_file_index: {self.pulse_file_index} simulation_time: {simulation_time}')
                if(self.pulse_file_index >= self.pulse_file_len):
                    self.pulse_file_status = PulseFileStatus.END
                    return
                else:
                    self.pulse_file_status = PulseFileStatus.INACTIVE
            if(simulation_time >= self.pulse_file_init_time[self.pulse_file_index]):
                if(self.pulse_file_status == PulseFileStatus.INACTIVE):
                    self.pulse_file_status = PulseFileStatus.ACTIVE
            if(self.pulse_file_status == PulseFileStatus.ACTIVE):
                self.set_active_pulses_gaussian_kernel(self.pulse_file_vent_x[self.pulse_file_index],self.pulse_file_vent_y[self.pulse_file_index],self.pulse_file_radius[self.pulse_file_index],substeps)
        self.set_active_pulses_gaussian_kernel_upc(
            400-156,
            66,
            self.pulse_file_radius[0],
            substeps)
        self.set_active_pulses_gaussian_kernel_upc(
            138,
            98,
            self.pulse_file_radius[0],
            substeps)
        # self.set_active_pulses_gaussian_kernel_upc(
        #     258,
        #     127,
        #     self.pulse_file_radius[0],
        #     substeps)
        self.set_active_pulses_gaussian_kernel_upc(
            400-180,
            150,
            self.pulse_file_radius[0],
            substeps)
        self.set_active_pulses_gaussian_kernel_upc(
            149,
            172,
            self.pulse_file_radius[0],
            substeps)
        self.set_active_pulses_gaussian_kernel_upc(
            235,
            246,
            self.pulse_file_radius[0],
            substeps)
        self.set_active_pulses_gaussian_kernel_upc(
            140,
            298,
            self.pulse_file_radius[0],
            substeps)
        self.set_active_pulses_gaussian_kernel_upc(
            250,
            334,
            self.pulse_file_radius[0],
            substeps)
        
    def add_dem(self,center_x,center_y,radius,brush_strength):
        height = 10.0*brush_strength/max_brush_strength_factor
        radius_grid = math.floor(radius/self.Grid.grid_size_to_km)
        bbox_min_x = center_x - radius_grid
        bbox_min_y = center_y - radius_grid
        bbox_max_x = center_x + radius_grid
        bbox_max_y = center_y + radius_grid

        for y in range(bbox_min_y,bbox_max_y):
            for x in range(bbox_min_x,bbox_max_x):
                u = (center_x-x)**2 + (center_y-y)**2
                if(u<radius_grid*radius_grid):
                    self.Grid.dem_elev[x,y] += height * cubicSmooth(u,radius_grid*radius_grid)
    
    def remove_dem(self,center_x,center_y,radius,brush_strength):
        height = 10.0*brush_strength/max_brush_strength_factor
        radius_grid = math.floor(radius/self.Grid.grid_size_to_km)
        bbox_min_x = center_x - radius_grid
        bbox_min_y = center_y - radius_grid
        bbox_max_x = center_x + radius_grid
        bbox_max_y = center_y + radius_grid

        for y in range(bbox_min_y,bbox_max_y):
            for x in range(bbox_min_x,bbox_max_x):
                u = (center_x-x)**2 + (center_y-y)**2
                if(u<radius_grid*radius_grid):
                    self.Grid.dem_elev[x,y] -= height * cubicSmooth(u,radius_grid*radius_grid)
    
    def add_heat(self,center_x,center_y,radius,brush_strength):
        height = 10e12*brush_strength/max_brush_strength_factor
        radius_grid = math.floor(radius/self.Grid.grid_size_to_km)
        bbox_min_x = center_x - radius_grid
        bbox_min_y = center_y - radius_grid
        bbox_max_x = center_x + radius_grid
        bbox_max_y = center_y + radius_grid

        for y in range(bbox_min_y,bbox_max_y):
            for x in range(bbox_min_x,bbox_max_x):
                u = (center_x-x)**2 + (center_y-y)**2
                if(u<radius_grid*radius_grid and self.Grid.lava_thickness[x,y] > self.Grid.update_heat_quantity_lava_height_minimum_m[None]):
                    self.Grid.heat_quantity[x,y] += height * cubicSmooth(u,radius_grid*radius_grid)
    
    def remove_heat(self,center_x,center_y,radius,brush_strength):
        height = 10e10*brush_strength/max_brush_strength_factor
        radius_grid = math.floor(radius/self.Grid.grid_size_to_km)
        bbox_min_x = center_x - radius_grid
        bbox_min_y = center_y - radius_grid
        bbox_max_x = center_x + radius_grid
        bbox_max_y = center_y + radius_grid

        for y in range(bbox_min_y,bbox_max_y):
            for x in range(bbox_min_x,bbox_max_x):
                u = (center_x-x)**2 + (center_y-y)**2
                if(u<radius_grid*radius_grid and self.Grid.lava_thickness[x,y] > self.Grid.update_heat_quantity_lava_height_minimum_m[None]):
                    # print(f'before self.Grid.heat_quantity[x,y]: {self.Grid.heat_quantity[x,y]}')
                    self.Grid.heat_quantity[x,y] -= height * cubicSmooth(u,radius_grid*radius_grid)
                    if(self.Grid.heat_quantity[x,y] < 0.0):
                        self.Grid.heat_quantity[x,y] = 0.0
                    # print(f'after self.Grid.heat_quantity[x,y]: {self.Grid.heat_quantity[x,y]}')