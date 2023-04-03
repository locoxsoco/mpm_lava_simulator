class SpatialDensity:
    def __init__(self):
        self.easting = 0.0
        self.northing = 0.0
        self.prob = 0.0

class Vent:
    def __init__(self):
        self.northing = 0.0
        self.easting = 0.0
        self.row = 0
        self.col = 0

class Lava_flow:
    def __init__(self):
        self.source = None
        self.num_vents = 0
        self.volumeToErupt = 0
        self.currentvolume = 0        
        self.pulsevolume = 0
        self.spd_grd = SpatialDensity()

class InParams:
    def __init__(self):
        self.config_file = None
        self.dem_file = None
        self.vents_file = None        
        self.slope_map = None
        self.uncert_map = None
        self.elev_uncert = 0
        self.spd_file = None
        self.num_grids = 0
        self.spd_grid_spacing = 0
        self.cell_size = 0
        self.min_pulse_volume = 0
        self.max_pulse_volume = 0
        self.min_total_volume = 0
        self.max_total_volume = 0
        self.log_mean_volume = 0
        self.log_std_volume = 0
        self.runs = 1
        self.flows = 1
        self.parents = 0
        self.flow_field = 0

class OutParams:
    def __init__(self):
        self.ascii_flow_file = ""
        self.ascii_hits_file = ""
        self.raster_hits_file = ""
        self.raster_flow_file = ""
        self.raster_post_dem_file = ""
        self.raster_pre_dem_file = ""
        self.stats_file = ""

def initialize():
    # Initialize vent parameters
    active_flow = Lava_flow()
    # Initialize input parameters
    inParams = InParams()
    # Initialize output parmaeters
    outParams = OutParams()
    return active_flow, inParams, outParams
    