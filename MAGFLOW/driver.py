import math
import numpy as np
from MAGFLOW.initialize import initialize, Vent
from MAGFLOW.magflow_input_file import configureParams
from MAGFLOW.heightmap import Heightmap
from MAGFLOW.grid import Grid

np.random.seed(42)

def gennor(av,sd):
    return sd*np.random.normal() + av

def genunf(low,high):
    return low + (high-low)*np.random.random()

def cubicSmooth(x,r):
    return (1.0-x/r)*(1.0-x/r)*(1.0-x/r)

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
        # Read in the DEM using the gdal library
        self.Heightmap = Heightmap(heightmap_path,dim,hm_elev_min_m,hm_elev_max_m)
        self.Grid = Grid(n_grid,dim,self.Heightmap)
        self.set_flow_params()
        # Initialize the lava flow data structures and initialize vent cell
        self.CAList = self.init_flow()
        self.CAListSize = 0
        self.current_vent = -1
        self.pulseCount = 0
        self.ActiveCounter = 0

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

    def set_active_pulses(self,center_x,center_y,radius,active_value: int):
        height = 0.0001
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
                    self.Grid.pulse_volume[x,y] += height * cubicSmooth(u,radius_grid*radius_grid)
        
    def add_dem(self,center_x,center_y,radius):
        height = 10.0
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
    
    def remove_dem(self,center_x,center_y,radius):
        height = 10.0
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
    
    def add_heat(self,center_x,center_y,radius):
        height = 10.0
        radius_grid = math.floor(radius/self.Grid.grid_size_to_km)
        bbox_min_x = center_x - radius_grid
        bbox_min_y = center_y - radius_grid
        bbox_max_x = center_x + radius_grid
        bbox_max_y = center_y + radius_grid

        for y in range(bbox_min_y,bbox_max_y):
            for x in range(bbox_min_x,bbox_max_x):
                u = (center_x-x)**2 + (center_y-y)**2
                if(u<radius_grid*radius_grid):
                    self.Grid.heat_quantity[x,y] += height * cubicSmooth(u,radius_grid*radius_grid)
    
    def remove_heat(self,center_x,center_y,radius):
        height = 10.0
        radius_grid = math.floor(radius/self.Grid.grid_size_to_km)
        bbox_min_x = center_x - radius_grid
        bbox_min_y = center_y - radius_grid
        bbox_max_x = center_x + radius_grid
        bbox_max_y = center_y + radius_grid

        for y in range(bbox_min_y,bbox_max_y):
            for x in range(bbox_min_x,bbox_max_x):
                u = (center_x-x)**2 + (center_y-y)**2
                if(u<radius_grid*radius_grid):
                    self.Grid.heat_quantity[x,y] -= height * cubicSmooth(u,radius_grid*radius_grid)