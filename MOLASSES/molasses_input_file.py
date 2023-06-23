def configureParams(inParams, outParams):
    ############################################################################
    # MOLASSES (MOdular LAva Simulation Software for the Earth Sciences) 
    # The MOLASSES model relies on a cellular automata algorithm to 
    # estimate the area inundated by lava flows.
    #
    #    Copyright (C) 2015-2021  
    #    Laura Connor (lconnor@usf.edu)
    #    Jacob Richardson 
    #    Charles Connor
    #
    #    This program is free software: you can redistribute it and/or modify
    #    it under the terms of the GNU General Public License as published by
    #    the Free Software Foundation, either version 3 of the License, or
    #    any later version.
    #
    #    This program is distributed in the hope that it will be useful,
    #    but WITHOUT ANY WARRANTY; without even the implied warranty of
    #    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    #    GNU General Public License for more details.
    #
    #    You should have received a copy of the GNU General Public License
    #    along with this program.  If not, see <http://www.gnu.org/licenses/>.
    ###########################################################################

    # MOLASSES Configuration File (with comments)
    # The comments can be removed for a simpler file.
    # The file name can be changed.

    ##############
    # Inputs
    ##############
    # ASCII file of vent locations, one easting northing pair per line.
    # Vent locations can be listed more than once to indicate a larger volume
    # of lava being erupted from a specific vent.
    inParams.vents_file = "inputs/vents_Fagradalsfjall.utm"
    #
    ######################################################################
    #DEM (digital elevation model) file in gdal readable format.
    inParams.dem_file = "inputs/dem.grd"
    #########################################################################
    # A grid cell model using a parent-child relationship prevents 
    # backward motion of the lava flow, choose PARENTS=Y. Currently, 'Y'
    # is required for multi-vent model.
    inParams.parents = 1 # Y
    #
    ##########################################################################
    # Integer value indicating uncertanty in DEM elevations
    # Default: 0 (no uncertainty)
    # OR a fileneame for a grid of uncertainties that varies from cell to cell
    inParams.elev_uncert = 0
    #
    #########################################################################
    # The RESIDUAL is the thickness of lava (in meters) required to accumulate
    # in a cell before that cell can release excess lava (thickness > RESIDUAL)
    # to its neighbors. It represents resistence to flow. 
    # A high RESIDUAL would be used for very viscous lavas,
    # low viscosity lavas would have lower RESIDUAL values.
    # For a hazard map, 
    # if the LOG_MEAN_RESIDUAL and LOG_STD_DEV_RESIDUAL are set,
    # the residual value is chosen between
    # a min and max value, based on a log-normal distribution.
    # The MIN_RESIDUAL and MAX_RESIDUAL are required;
    # the log(mean) and log(std_dev) are optional. If not set,
    # the residual is chosed from a uniform random distribution between
    # the min and max value. To apply a single residual value set
    # both MIN_ and MAX_ to the same value.
    # High viscosity
    inParams.min_residual = 30.0
    inParams.max_residual = 30.0
    # Low viscosity
    inParams.min_residual = 25.0
    inParams.max_residual = 25.0
    # inParams.min_residual = 0.3
    # inParams.max_residual = 0.3
    #LOG_MEAN_RESIDUAL = 0.38
    #LOG_STD_DEV_RESIDUAL = 0.1
    #
    ############################################################################
    # The TOTAL_VOLUME (total erupted lava volume, units=cubic meters) is
    # the volume of lava erupted from the active vent(s), specified for a single flow. 
    # For a hazard map, 
    # if the LOG_MEAN_TOTAL_VOLUME and LOG_STD_DEV_TOTAL_VOLUME are set,
    # the total lava volume is chosen between
    # a min and max value, and based on a log-normal distribution,
    # The min_volume and max_volume are required;
    # the LOG_MEAN_TOTAL_VOLUME and LOG_STD_DEV_TOTAL_VOLUME are optional. If not set,
    # the volume is chosed from a uniform random distribution between
    # the MIN_ and MAX_ value. To apply a single volume, set
    # both MIN_ and MAX_ to the same value.
    #LOG_MEAN_TOTAL_VOLUME = 6.5
    #LOG_STD_DEV_TOTAL_VOLUME = 0.2
    inParams.min_total_volume = 300000
    inParams.max_total_volume = 300000
    inParams.min_total_volume = 300.0
    inParams.max_total_volume = 300.0
    inParams.min_total_volume = 979200.0
    inParams.max_total_volume = 979200.0
    inParams.min_total_volume = 1181821696.0
    inParams.max_total_volume = 1181821696.0
    #
    ################################################################################
    # The PULSE_VOLUME is the fraction of the total lava volume that gets erupted
    # and distributed sequentially during code execution. Compare a garden hose
    # to a fire hydrant, the garden hose exhibits a smaller pulse volume that a fire hydrant.
    # Asuming equal total volumes, 
    # A lower PULSE_VOLUME would take longer to spread
    # than a higher PULSE_VOLUME.
    # For a hazard map, 
    # the PULSE_VOLUME is chosen randomely between a MIN_ and MAX_ value.
    # To apply a single PULSE_VOLUME, set both the MIN_ and MAX_ to the same value.
    inParams.min_pulse_volume = 100
    inParams.max_pulse_volume = 100
    inParams.min_pulse_volume = 164141.90
    inParams.max_pulse_volume = 164141.90
    #

    ########################
    # Simulation Parameters
    ########################
    # Number of lava flows to erupt per simulation
    inParams.flows = 1
    #
    # Number of simulation runs
    inParams.runs = 1
    #
    #############################
    # OUTPUTS
    ############################
    # grid cells inundated with lava (xyz: easting northing lava_thickness (m) )
    outParams.ascii_flow_file = "flow"
    #
    # grid cells inundated by lava (xyz: easting northing hit_count) 
    # ASCII_HIT_MAP = hits
    # tiff (raster) format output maps
    # RASTER_FLOW_MAP = 
    # RASTER_HIT_MAP =
    # RASTER_POST_DEM =