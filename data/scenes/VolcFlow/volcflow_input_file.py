#Example : simulation of a retrogressive dome collapse at Merapi
#two-fluids version of VolFlow for PDCs - the density of the surge varies
#K. Kelfoun, Apr. 2021 - UCA / LMV - lmv.uca.fr/kelfoun-karim

erase_dat=1

#Reads a DEM of Merapi volcano, Surfer8-format
fid = fopen('DEM_Merapi.grd')
code = fscanf(fid,'#c',4)
ncol = fread(fid,1,'int16')
nrow = fread(fid,1,'int16')
xmin = fread(fid,1,'float64')
xmax = fread(fid,1,'float64')
ymin = fread(fid,1,'float64')
ymax = fread(fid,1,'float64')
zmin = fread(fid,1,'float64')
zmax = fread(fid,1,'float64')
altitude = fread(fid,'float32')
z = (reshape(altitude, ncol, nrow))
fclose(fid)
[nrow, ncol]=size(z)

#space step of the DEM
dx_horiz = (xmax-xmin)/(ncol-1)
dy_horiz = (ymax-ymin)/(nrow-1)

#geographic vectors
x= [xmin+dx_horiz/2:dx_horiz:xmin+ncol*dx_horiz-dx_horiz/2]
y = [ymin+dy_horiz/2:dy_horiz:ymin+nrow*dy_horiz-dy_horiz/2]


#This simple source condition considers that the source of pyroclastic
#flow is 20m thick during all the collapse
collapse_duration = 150
time_of_collapse = (255-double(imread('lava_dome.tif')))/255*collapse_duration
time_of_collapse=flipud(time_of_collapse)
x_time_of_collapse = (time_of_collapse(:,2:end)+time_of_collapse(:,1:end-1))/2
y_time_of_collapse = (time_of_collapse(2:end,:)+time_of_collapse(1:end-1,:))/2

#Defines the initial thickness of the lava dome and/or of the pyroclastic flow
h=z*0
h(time_of_collapse>0)=50 
before_flux = 'source_dome' #to be used if you want to modify the velocities before the calculation of fluxes

#Defines the initial thickness of the surge (here 0)
hw= z*0

#Defines the initial thickness of the deposit (here 0)
hsed= h*0

#Used to know if a deposit has proviously flowed or not
hecoul = h*0

#Used to know if the deposit comes from a flow only or has been transported (even for a short duration) by the surge 
hd=h

#h2nd is defined automatically in VolcFlow to detect surge-derived pyroclastic flow

#Calculation parameters -------------------------------
dt=0.125      #time step
dtplot = 10  #time step for the figures
tmax =  50  #max calculation time
g=9.81       #gravity


coef_u2 = [0.01 0.3]       #turbulence coefficient. 1) flow, 2) surge
viscosity = [0 0]          #viscosities
rhoa=1                     #atmosphere density
rhog=rhoa*273/(273+150)    #density of the gas of the surge
rhop = 2400                #density of  particles
rhoD = 1600                #bulk density of the flow
rhoS = h*0+rhoa            #surge density - this will be modified by VF ...
rho_variable = 1           #... if rho_variable = 1. For 0, rhoS is constant and only the surge thickness varies 
                            #Do not change - or modify script_settling and script_genesis
rho_mix = 8                #density of the mixing that leaves the flow to supply the surge
coef_Cd = 1                #drag coefficient of particles
coef_a3 = 0.001/1.5/2      #production du surge - used by script_genesis.m
coef_a2 = 1/5              #modifies the particles velocity - used by script_genesis
d_part = 0.25/1000         #diameter of the particles (m)

cohesion = 4000            #plastic stress or cohesion (Pa)
delta_int = 0              #internal friction angle (radians)
delta_bed   = 0/180*pi     #basal friction angle (radians)

#Write in these scripts the equations that control the exchanges between the flow and the surge.
flow2surge = 'script_genesis'
surge2flow = 'script_settling'

#Visualization during calculation, movie file and data file
representation = 'repr_merapi set(gcf, ''color'', ''w'')'
f_avi  = 'merapi_eff.avi'
f_data = 'merapi_eff.dat'

#Boundary conditions (at the edges of the topography)
#You must use h2 and hw2 (instead of h and hw) here
bound_cond = 'h2(:,1)=0 h2(:,ncol)=0 h2(1, :)=0 h2(nrow, :)=0'
bound_cond_w ='hw2(:,1)=0 hw2(:,ncol)=0 hw2(1, :)=0 hw2(nrow, :)=0'

#Use this if you want to generate a PDCs somewhere
source_flow  = ''
source_surge = ''

#initial velocities - 0 here
u_xx = z(1:nrow, 1:ncol-1)*0
u_xy = z(1:nrow, 1:ncol-1)*0
u_yy = z(1:nrow-1, 1:ncol)*0
u_yx = z(1:nrow-1, 1:ncol)*0

uw_xx = z(1:nrow, 1:ncol-1)*0
uw_xy = z(1:nrow, 1:ncol-1)*0
uw_yy = z(1:nrow-1, 1:ncol)*0
uw_yx = z(1:nrow-1, 1:ncol)*0

tstep_adjust = 0 #not tested with the variable time step - use always 0
CFLcond=0.75     #coefficient for the CFL condition
doubleUpwind = 0 #single or double upwind scheme - more stable with 0 

#Variable saved in the dat-file
#saved_var = {'h', 'hd', 'hw', 'hsed', 'umx', 'umy', 'u_dense'}
saved_var = {'h', 'umx', 'umy'}

dh_gain_surge = h *0 #initialisation of a variable used by script_genesis
FLUX_MASS = 0        #initialisation of a variable used by script_genesis
Vlost=0              #initialisation of a variable that records the volume lost by lift-off during the simulation
