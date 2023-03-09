import math
import numpy as np
from MOLASSES.initialize import initialize, Vent
from MOLASSES.molasses_input_file import configureParams
from MOLASSES.pulse import pulse
from MOLASSES.distribute import distribute
from MOLASSES.heightmap import Heightmap
from MOLASSES.grid import Grid

np.random.seed(42)

def gennor(av,sd):
    return sd*np.random.normal() + av

def genunf(low,high):
    return low + (high-low)*np.random.random()


class Neighbor:
    def __init__(self):
        self.row = 0
        self.col = 0
        self.run = 0.0
        self.elev_diff = 0.0

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
        # Initialize the remaining volume to be the volume of lava to erupt.
        self.volumeRemaining = self.active_flow.volumeToErupt
        self.CAListSize = 0
        self.current_vent = -1
        self.pulseCount = 0
        self.NeighborList = []
        for _ in range(8):
            self.NeighborList.append(Neighbor())
        self.ActiveCounter = 0
    
    def load_vent_data(self):
        self.active_flow.num_vents = 1
        self.active_flow.source = []
        for _ in range(self.active_flow.num_vents):
            self.active_flow.source.append(Vent())
        self.active_flow.source[0].easting = 20.0
        self.active_flow.source[0].northing = 20.0
        self.active_flow.source[0].easting = 18.52
        self.active_flow.source[0].northing = 18.52
        # self.active_flow.source[1].easting = 20.0
        # self.active_flow.source[1].northing = 20.0
        # self.active_flow.source[2].easting = 20.0
        # self.active_flow.source[2].northing = 20.0
        # self.active_flow.source[3].easting = 20.0
        # self.active_flow.source[3].northing = 20.0
        # self.active_flow.source[4].easting = 20.0
        # self.active_flow.source[4].northing = 20.0
        # self.active_flow.source[5].easting = 20.0
        # self.active_flow.source[5].northing = 20.0
        print(f'self.active_flow.source[0].easting: {self.active_flow.source[0].easting} self.active_flow.source[0].now: {self.active_flow.source[0].row}')
    
    def set_flow_params(self):
        print(f'self.inParams.min_residual: {self.inParams.max_residual} self.inParams.max_residual: {self.inParams.min_residual}')
        if(self.inParams.min_residual > 0 and self.inParams.max_residual > 0 and self.inParams.max_residual >= self.inParams.min_residual):
            if (self.inParams.log_mean_residual > 0 and self.inParams.log_std_residual > 0):
                log_min = math.log(self.inParams.min_residual,10)
                log_max = math.log(self.inParams.max_residual,10)
                self.active_flow.residual = gennor(self.inParams.log_mean_residual, self.inParams.log_std_residual)
                while(self.active_flow.residual > log_max or self.active_flow.residual < log_min):
                    self.active_flow.residual = gennor(self.inParams.log_mean_residual, self.inParams.log_std_residual)
                self.active_flow.residual = math.pow(self.active_flow.residual,10)
            else:
                self.active_flow.residual = genunf(self.inParams.min_residual, self.inParams.max_residual)
        else:
            self.active_flow.residual = self.inParams.residual
        
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
        
        print(f'self.active_flow.residual: {self.active_flow.residual} self.active_flow.volumeToErupt: {self.active_flow.volumeToErupt} self.active_flow.pulsevolume: {self.active_flow.pulsevolume}')
        
        # Write residual value into 2D Global Data Array
        self.Grid.fill_residual(self.active_flow.residual)
    
    def init_flow(self):
        maxCellsPossible = self.Grid.info[4] * self.Grid.info[2]
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
            self.active_flow.source[i].row = int((self.active_flow.source[i].northing - self.Grid.info[3]) / self.Grid.info[5])   # Row (Y) of vent cell
            self.active_flow.source[i].col = int((self.active_flow.source[i].easting - self.Grid.info[0]) / self.Grid.info[1])   # Col (X) of vent cell

        return local_CAList


    def step(self,render,camera,window,scene,canvas,show_options,gui):
        if(self.volumeRemaining > 0.0):
            # vent cell gets a new pulse of lava to distribute
            self.current_vent = (self.current_vent + 1) % (self.active_flow.num_vents)
            
            self.CAList[self.current_vent].row = self.active_flow.source[self.current_vent].row
            self.CAList[self.current_vent].col = self.active_flow.source[self.current_vent].col
            
            if (not(self.pulseCount % 100)):
                print(f'[Driver] Vent: {self.active_flow.source[self.current_vent].easting} {self.active_flow.source[self.current_vent].northing} Active Cells: {self.ActiveCounter} Volume Remaining: {self.volumeRemaining} Pulse Count: {self.pulseCount}')
                print(f'[Driver] self.Grid.eff_elev[215,215]: {self.Grid.eff_elev[215,215]} self.ActiveCounter: {self.ActiveCounter}')
            
            
            self.volumeRemaining = pulse(
            # &ActiveCounter,     (type= unsigned int*) Active list current cell count
            self.CAList[self.current_vent],            # (type=ActiveList*) 1D Active Cells List
            self.active_flow,       # (type=Lava_flow*) Lava_flow Data structure
            self.Grid,              # (type=DataCell**)  2D Data Grid
            self.volumeRemaining,   # (type=double) Lava volume not yet erupted
            self.Grid.info)         # (type=double*) Metadata array
            # print(f'[Driver] self.volumeRemaining: {self.volumeRemaining} active_flow.currentvolume: {self.active_flow.currentvolume}')

            self.pulseCount+=1

            # Distribute lava to active cells and their 8 neighbors.
            ret,self.ActiveCounter = distribute(
            self.Grid,              # (type=DataCell**)  2D Data Grid
            self.CAList,            # (type=ActiveList*) 1D Active Cells List
            self.CAListSize,        # (type=unsigned int) Max size of Active list
            self.ActiveCounter,     # (type=unsigned int*) Active list current cell count
            self.NeighborList,      # (type=Neighbor*) 8 element list of cell-neighbors info
            self.Grid.info,         # (type=double*) Metadata array
            # &ActiveFlow.source      (type=Lava_flow*) Lava_flow Data structure 
            render,camera,window,scene,canvas,self.Heightmap,show_options,gui)

            if (ret):
                print( "[MAIN Error returned from [DISTRIBUTE].ret=", ret)
                if (ret < 0): 
                    print( "[MAIN Error returned from [DISTRIBUTE].ret=", ret)
                    volumeRemaining = 0.0