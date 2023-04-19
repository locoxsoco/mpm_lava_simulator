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
# Heightmap color
cube_colors_list_lvl0 = np.array([
    [54.0/256.0, 47.0/256.0, 54.0/256.0, 1.0],
    [54.0/256.0, 47.0/256.0, 54.0/256.0, 1.0],
    [54.0/256.0, 47.0/256.0, 54.0/256.0, 1.0],
    [54.0/256.0, 47.0/256.0, 54.0/256.0, 1.0],
    [54.0/256.0, 47.0/256.0, 54.0/256.0, 1.0],
    [54.0/256.0, 47.0/256.0, 54.0/256.0, 1.0],
    [54.0/256.0, 47.0/256.0, 54.0/256.0, 1.0],
    [54.0/256.0, 47.0/256.0, 54.0/256.0, 1.0]
], dtype=np.float32)
# Lava color
# cube_colors_list_lvl1 = np.array([
#     [169.0/256.0, 24.0/256.0, 35.0/256.0, 1.0],
#     [169.0/256.0, 24.0/256.0, 35.0/256.0, 1.0],
#     [169.0/256.0, 24.0/256.0, 35.0/256.0, 1.0],
#     [169.0/256.0, 24.0/256.0, 35.0/256.0, 1.0],
#     [169.0/256.0, 24.0/256.0, 35.0/256.0, 1.0],
#     [169.0/256.0, 24.0/256.0, 35.0/256.0, 1.0],
#     [169.0/256.0, 24.0/256.0, 35.0/256.0, 1.0],
#     [169.0/256.0, 24.0/256.0, 35.0/256.0, 1.0]
# ], dtype=np.float32)
cube_colors_list_lvl1 = np.array([
    [207.0/256.0, 16.0/256.0, 32.0/256.0, 1.0],
    [207.0/256.0, 16.0/256.0, 32.0/256.0, 1.0],
    [207.0/256.0, 16.0/256.0, 32.0/256.0, 1.0],
    [207.0/256.0, 16.0/256.0, 32.0/256.0, 1.0],
    [207.0/256.0, 16.0/256.0, 32.0/256.0, 1.0],
    [207.0/256.0, 16.0/256.0, 32.0/256.0, 1.0],
    [207.0/256.0, 16.0/256.0, 32.0/256.0, 1.0],
    [207.0/256.0, 16.0/256.0, 32.0/256.0, 1.0]
], dtype=np.float32)
# Crust color
cube_colors_list_lvl2 = np.array([
    [93.0/256.0, 40.0/256.0, 39.0/256.0, 1.0],
    [93.0/256.0, 40.0/256.0, 39.0/256.0, 1.0],
    [93.0/256.0, 40.0/256.0, 39.0/256.0, 1.0],
    [93.0/256.0, 40.0/256.0, 39.0/256.0, 1.0],
    [93.0/256.0, 40.0/256.0, 39.0/256.0, 1.0],
    [93.0/256.0, 40.0/256.0, 39.0/256.0, 1.0],
    [93.0/256.0, 40.0/256.0, 39.0/256.0, 1.0],
    [93.0/256.0, 40.0/256.0, 39.0/256.0, 1.0]
], dtype=np.float32)
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
    def __init__(self,n_grid,dim,heightmap):
        self.grid_size_to_km = heightmap.hm_height_px*heightmap.px_to_km/n_grid
        self.km_to_m = 1000.0
        print(f'self.grid_size_to_km: {self.grid_size_to_km} self.km_to_m: {self.km_to_m}')

        self.cube_positions = ti.Vector.field(dim, ti.f32, 8)
        self.cube_positions.from_numpy(cube_verts_list)
        self.cube_positions2 = ti.Vector.field(dim, ti.f32, 8)
        self.cube_positions2.from_numpy(cube_verts_list)
        self.cube_positions3 = ti.Vector.field(dim, ti.f32, 8)
        self.cube_positions3.from_numpy(cube_verts_list)
        self.cube_indices = ti.field(ti.i32, shape=len(cube_faces_list))
        self.cube_indices.from_numpy(cube_faces_list)
        self.cube_normals = ti.Vector.field(dim, ti.f32, 8)
        self.cube_normals.from_numpy(cube_face_normals_list)
        self.cube_colors_lvl0 = ti.Vector.field(4, ti.f32, 8)
        self.cube_colors_lvl0.from_numpy(cube_colors_list_lvl0)
        self.cube_colors_lvl1 = ti.Vector.field(4, ti.f32, 8)
        self.cube_colors_lvl1.from_numpy(cube_colors_list_lvl1)
        self.cube_colors_lvl2 = ti.Vector.field(4, ti.f32, 8)
        self.cube_colors_lvl2.from_numpy(cube_colors_list_lvl2)
        self.curr_cube_positions = ti.Vector.field(dim, ti.f32, 8)

        self.m_transforms_lvl0 = ti.Matrix.field(4,4,dtype=ti.f32,shape=n_grid*n_grid)
        self.m_transforms_lvl1 = ti.Matrix.field(4,4,dtype=ti.f32,shape=n_grid*n_grid)
        self.m_transforms_lvl2 = ti.Matrix.field(4,4,dtype=ti.f32,shape=n_grid*n_grid)

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
        neighDistances = np.array([1.0*self.grid_size_to_km*self.km_to_m,
                                   1.0*self.grid_size_to_km*self.km_to_m,
                                   1.0*self.grid_size_to_km*self.km_to_m,
                                   1.0*self.grid_size_to_km*self.km_to_m,
                                   sqrt2*self.grid_size_to_km*self.km_to_m,
                                   sqrt2*self.grid_size_to_km*self.km_to_m,
                                   sqrt2*self.grid_size_to_km*self.km_to_m,
                                   sqrt2*self.grid_size_to_km*self.km_to_m], dtype=np.float32)

        self.dem_elev = ti.field(ti.f32, (n_grid, ) * (dim-1))
        self.lava_thickness = ti.field(ti.f32, (n_grid, ) * (dim-1))
        self.solid_lava_thickness = ti.field(ti.f32, (n_grid, ) * (dim-1))
        self.heat_quantity = ti.field(ti.f32, (n_grid, ) * (dim-1))
        self.temperature = ti.field(ti.f32, (n_grid, ) * (dim-1))
        self.delta_time = ti.field(ti.f32, (n_grid, ) * (dim-1))
        self.lava_flux = ti.field(ti.f32, shape=(n_grid,n_grid,8))
        self.global_delta_time = ti.field(ti.f32, shape=())
        
        self.is_active = ti.field(ti.i32, shape=(n_grid,n_grid))
        self.pulse_volume = ti.field(ti.f32, shape=(n_grid,n_grid))

        # Etna's lava parameters
        self.lava_density = 2600.0
        self.specific_heat_capacity = 1150.0
        self.emissivity = 0.9
        self.ambient_temperature = 298.15
        self.solidification_temperature = 950.0
        self.extrusion_temperature = 1400.0
        self.H2O = 0.06325
        self.gravity = 9.81
        self.delta_time_c = 0.99
        self.cell_area = (self.grid_size_to_km*self.km_to_m)**2
        print(f'self.grid_size_to_km: {self.grid_size_to_km} self.cell_area: {self.cell_area}')
        self.global_delta_time = 0.0
        self.c_v = self.specific_heat_capacity
        self.max_lava_thickness = 250.0

        self.nRows = ti.field(ti.i32, 8)
        self.nCols = ti.field(ti.i32, 8)
        self.neighCodes = ti.field(ti.i32, 8)
        self.neighDistances = ti.field(ti.f32, 8)
        self.nRows.from_numpy(nRows)
        self.nCols.from_numpy(nCols)
        self.neighCodes.from_numpy(neighCodes)
        self.neighDistances.from_numpy(neighDistances)


        self.active = ti.field(ti.i32, (n_grid, ) * (dim-1))
        self.init_values(heightmap)
        self.calculate_m_transforms_lvl0()
        self.calculate_m_transforms_lvl1()

    @ti.kernel
    def init_values(self,heightmap: ti.template()):
        for i,j in self.dem_elev:
            self.dem_elev[i,j] = heightmap.heightmap_positions[int((i/self.n_grid+1.0/(2.0*self.n_grid))*heightmap.hm_height_px)*heightmap.hm_width_px+int((j/self.n_grid+1.0/(2.0*self.n_grid))*heightmap.hm_width_px)][1]*self.km_to_m
            self.parentcodes[i,j] = 0
            self.active[i,j] = -1
            self.lava_thickness[i,j] = 0.0
            self.solid_lava_thickness[i,j] = 0.0
            self.heat_quantity[i,j] = 0.0
            self.is_active[i,j] = 0
            self.pulse_volume[i,j] = 0.0

            self.temperature[i,j] = self.ambient_temperature
            self.delta_time[i,j] = 0.0

            # if(i==200 and j==200):
            #     self.lava_thickness[i,j] = 50.0

    @ti.kernel
    def calculate_m_transforms_lvl0(self):
        for idx in self.m_transforms_lvl0:
            i = idx//self.n_grid
            k = idx%self.n_grid
            self.m_transforms_lvl0[idx] = ti.Matrix.identity(float,4)
            self.m_transforms_lvl0[idx] *= self.grid_size_to_km
            self.m_transforms_lvl0[idx][1,1] = 1.0
            self.m_transforms_lvl0[idx][1,1] *= self.dem_elev[i,k]/self.km_to_m
            self.m_transforms_lvl0[idx][0,3] = i*self.grid_size_to_km + self.grid_size_to_km
            # self.m_transforms_lvl0[idx][1,3] = self.dem_elev[i,k] + self.grid_size_to_km
            self.m_transforms_lvl0[idx][2,3] = k*self.grid_size_to_km + self.grid_size_to_km
            self.m_transforms_lvl0[idx][3,3] = 1

    @ti.kernel
    def calculate_m_transforms_lvl1(self):
        for idx in self.m_transforms_lvl1:
            i = idx//self.n_grid
            k = idx%self.n_grid
            thickness = self.lava_thickness[i,k]
            if thickness > 1.0:
                self.m_transforms_lvl1[idx] = ti.Matrix.identity(float,4)
                self.m_transforms_lvl1[idx] *= self.grid_size_to_km
                self.m_transforms_lvl1[idx][1,1] = 1.0
                self.m_transforms_lvl1[idx][1,1] *= thickness/self.km_to_m
                self.m_transforms_lvl1[idx][0,3] = i*self.grid_size_to_km + self.grid_size_to_km
                self.m_transforms_lvl1[idx][1,3] = self.dem_elev[i,k]/self.km_to_m
                self.m_transforms_lvl1[idx][2,3] = k*self.grid_size_to_km + self.grid_size_to_km
                self.m_transforms_lvl1[idx][3,3] = 1
    
    @ti.kernel
    def calculate_m_transforms_lvl2(self,anchor_i: int,anchor_k: int):
        for idx in self.m_transforms_lvl2:
            i = int(idx//self.n_grid)
            k = int(idx%self.n_grid)
            if i==anchor_i and k==anchor_k:
                self.m_transforms_lvl2[idx] = ti.Matrix.identity(float,4)
                self.m_transforms_lvl2[idx] *= self.grid_size_to_km
                self.m_transforms_lvl2[idx][1,1] = 1.0
                self.m_transforms_lvl2[idx][1,1] *= (self.dem_elev[i,k]+self.lava_thickness[i,k]+100.0)/self.km_to_m
                self.m_transforms_lvl2[idx][0,3] = i*self.grid_size_to_km + self.grid_size_to_km
                # self.m_transforms_lvl2[idx][1,3] = self.dem_elev[i,k]/self.km_to_m
                self.m_transforms_lvl2[idx][2,3] = k*self.grid_size_to_km + self.grid_size_to_km
                self.m_transforms_lvl2[idx][3,3] = 1
            else:
                self.m_transforms_lvl2[idx] = ti.Matrix.identity(float,4)
                self.m_transforms_lvl2[idx][2,3] = 654654654

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
            rho = self.lava_density
            c_v = self.c_v
            delta_Q_t_m *= rho * c_v * self.global_delta_time
            
            epsilon = self.emissivity
            A = self.cell_area
            # Stefan–Boltzmann
            sigma = 5.68 * 10**(-4)
            delta_Q_t_r = epsilon * A * sigma * self.temperature[i,k]**4 * self.global_delta_time

            self.heat_quantity[i,k] += delta_Q_t_m - delta_Q_t_r
            if(i==215 and k==215):
                print(f'self.heat_quantity[i,k]: {self.heat_quantity[i,k]} delta_Q_t_m: {delta_Q_t_m} delta_Q_t_r: {delta_Q_t_r}')

    @ti.kernel
    def updateTemperature(self):
        for i,k in self.temperature:
            if(self.heat_quantity[i,k]>0):
                rho = self.lava_density
                c_v = self.c_v
                h_t_dt = self.lava_thickness[i,k]
                A = self.cell_area
                self.temperature[i,k] = self.heat_quantity[i,k] / (rho * c_v * h_t_dt * A)
            else:
                self.heat_quantity[i,k] = 0.0
            if(i==215 and k==215):
                print(f'self.temperature[i,k]: {self.temperature[i,k]}')

    @ti.kernel
    def computeNewLavaThickness(self):
        for i,k in self.dem_elev:
            q_tot = 0.0
            for n in ti.static(range(8)):
                q_tot += self.lava_flux[i,k,n]
            delta_lava_thickness = q_tot*self.global_delta_time/self.cell_area
            self.lava_thickness[i,k] += delta_lava_thickness
    
    @ti.kernel
    def computeTimeSteps(self):
        for i,k in self.dem_elev:
            c = self.delta_time_c
            h = self.lava_thickness[i,k]
            A = self.cell_area
            q_tot = 0.0
            for n in ti.static(range(8)):
                q_tot += self.lava_flux[i,k,n]
            self.delta_time[i,k] = 9999.9
            if h>0 and not (ti.math.isnan(q_tot) or ti.math.isinf(q_tot)) and q_tot<0:
                self.delta_time[i,k] = c*h*A/ti.abs(q_tot)
    
    @ti.kernel
    def computeGlobalTimeStep(self) -> ti.f32:
        global_delta_time = 9999999.9
        for i,k in self.dem_elev:
            ti.atomic_min(global_delta_time, self.delta_time[i,k])
        return global_delta_time

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
                if((delta_z+delta_h) > 0):
                    h = self.lava_thickness[i_n,k_n]
                    T = self.temperature[i_n,k_n]
                
                if (h<=0):
                    self.lava_flux[i,k,n] = 0.0
                else:
                    rho = self.lava_density
                    g = self.gravity
                    delta_x_sign = 1
                    if((delta_z+delta_h) < 0):
                        delta_x_sign = -1
                    delta_x = delta_x_sign * self.neighDistances[n]
                    S_y = 10.0**(13.00997 - 0.0089*T)
                    eta = 10.0**(-4.643 + (5812.44 - 427.04*self.H2O)/(T - 499.31 + 28.74*ti.log(self.H2O)))
                    h_cr = S_y * (ti.math.sqrt(delta_z**2 + delta_x**2)) / (rho*g*delta_x_sign*(delta_z+delta_h))
                    a = h/h_cr
                    if(h>h_cr):
                        q = (S_y * h_cr**2 * delta_x)/(3.0*eta) * (a**3 - 3.0/2.0*a**2 + 1.0/2.0)
                        self.lava_flux[i,k,n] = q
                    else:
                        self.lava_flux[i,k,n] = 0.0
    
    @ti.kernel
    def pulse(self):
        for i,k in self.dem_elev:
            if(self.is_active[i,k]==1):
                pulsevolume = self.pulse_volume[i,k]
                pulsevolume *= self.km_to_m**3
                pulseThickness = pulsevolume / self.cell_area
                new_lava_thickness = self.lava_thickness[i,k] + pulseThickness
                if (new_lava_thickness > self.max_lava_thickness):
                    pulseThickness = self.max_lava_thickness - self.lava_thickness[i,k]
                    pulsevolume = pulseThickness * self.cell_area
                self.lava_thickness[i,k] += pulseThickness
                self.is_active[i,k] = 0
                self.pulse_volume[i,k] = 0.0
                if(pulseThickness>0):
                    rho = self.lava_density
                    c_v = self.c_v
                    A = self.cell_area
                    h_t_dt = self.lava_thickness[i,k]
                    self.heat_quantity[i,k] += pulseThickness*A*self.extrusion_temperature*rho*c_v
                    self.temperature[i,k] = self.heat_quantity[i,k]/(rho*c_v*h_t_dt*A)


    def bboxIntersect(self,rayPosition,rayDirection):
        epsilon = 1.0e-5
        tmin = -1e-16
        tmax = 1e16

        a = [0,0]
        b = [self.n_grid, self.n_grid]
        p = rayPosition
        d = rayDirection - rayPosition
        # print(f'd: {d} p: {p}')

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
            # print(f't1: {t} tmin: {tmin} tmax: {tmax}')
            if (t < tmin):
                return False,0.0,0.0
            if (t <= tmax):
                tmax = t
            t = (a[0] - p[0]) / d[0]
            # print(f't2: {t} tmin: {tmin} tmax: {tmax}')
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
            # print(f't3: {t} tmin: {tmin} tmax: {tmax}')
            if (t < tmin):
                return False,0.0,0.0
            if (t <= tmax):
                tmax = t
            t = (a[1] - p[1]) / d[1]
            # print(f't4: {t} tmin: {tmin} tmax: {tmax}')
            if (t >= tmin):
                if(t > tmax):
                    return False,0.0,0.0
                tmin = t
        elif (p[1]<a[1] or p[1]>b[1]):
            return False,0.0,0.0
        
        return True,tmin,tmax

    def Intersect(self,rayPosition,rayDirection):
        rayPosList = [rayPosition[0],rayPosition[2]]
        rayPositionGrid = np.array(rayPosList)
        rayDirGridList = [rayDirection[0],rayDirection[2]]
        rayDirectionGrid = np.array(rayDirGridList)
        # normRayDirectionGrid = np.linalg.norm(rayDirectionGrid)
        # rayDirectionGrid /= normRayDirectionGrid
        # rayDirectionGrid /= self.grid_size_to_km
        # print(f'rayPositionGrid: {rayPositionGrid} rayDirectionGrid: {rayDirectionGrid}')
        # Check the intersection with the bounding box
        isBboxIntersected,ta,tb = self.bboxIntersect(rayPositionGrid,rayDirectionGrid)
        print(f'ta: {ta} tb: {tb}')
        if(isBboxIntersected):
            # Ray marching
            t = ta + 0.0001
            if (ta < 0.0):
                t = 0.0
            
            while (t < tb):
                #  Point along the ray
                p = rayPosition/self.grid_size_to_km + t*rayDirection
                # print(f'p_curr: {p} p[1]*self.km_to_m: {p[1]*self.km_to_m}')
                # return
                h = self.dem_elev[int(p[0]),int(p[2])] + self.lava_thickness[int(p[0]),int(p[2])] + self.solid_lava_thickness[int(p[0]),int(p[2])]
                print(f'curr_p: {p} h: {h} p[1]*self.grid_size_to_km*self.km_to_m: {p[1]*self.grid_size_to_km*self.km_to_m}')
                if (h > p[1]*self.grid_size_to_km*self.km_to_m):
                    # print(f'p[0]: {p[0]} p[2]: {p[2]} h: {h} p[1]*self.km_to_m: {p[1]*self.km_to_m} p: {p}')
                    return True,int(p[0]),int(p[2])
                else:
                    t += 1.0
            
            return False,-1,-1