import taichi as ti
import numpy as np

ti.init(arch=ti.vulkan)
if ti.lang.impl.current_cfg().arch != ti.vulkan:
    raise RuntimeError("Vulkan is not available.")

# # north, south, east, west, north-east, north-west, south-east, south-west
# npRows = np.array([+1,-1,+0,+0,+1,+1,-1,-1], dtype=np.int32)
# npCols = np.array([+0,+0,+1,-1,+1,-1,+1,-1], dtype=np.int32)
# npNeighCodes = np.array([1,2,4,8,16,32,64,128], dtype=np.int32)
# sqrt2 = ti.math.sqrt(2)
# npNeighDistances = np.array([1.0,1.0,1.0,1.0,sqrt2,sqrt2,sqrt2,sqrt2], dtype=np.float32)

# nRows = ti.ndarray(ti.i32, 8)
# nCols = ti.ndarray(ti.i32, 8)
# neighCodes = ti.ndarray(ti.i32, 8)
# neighDistances = ti.ndarray(ti.f32, 8)

# nRows.from_numpy(npRows)
# nCols.from_numpy(npCols)
# neighCodes.from_numpy(npNeighCodes)
# neighDistances.from_numpy(npNeighDistances)


# ##################################################################
# n_grid = 100
# lava_flux = ti.ndarray(ti.f32, shape=(n_grid,n_grid,8))
# dem_elev = ti.ndarray(ti.f32, shape=(n_grid,n_grid))
# solid_lava_thickness = ti.ndarray(ti.f32, shape=(n_grid,n_grid))
# lava_thickness = ti.ndarray(ti.f32, shape=(n_grid,n_grid))
# temperature = ti.ndarray(ti.f32, shape=(n_grid,n_grid))
# new_temperature = ti.ndarray(ti.f32, shape=(n_grid,n_grid))
# heat_quantity = ti.ndarray(ti.f32, shape=(n_grid,n_grid))
# delta_time = ti.ndarray(ti.f32, shape=(n_grid,n_grid))
# scaled_grid_size_m = 1
# quality_tolerance = 1
# flux_height_minimum_m = 1
# delta_total_height_min = 1
# delta_total_height_max = 1
# lava_density = 1
# gravity = 1
# H2O = 1
# delta_time_c = 1
# cell_area_m = 1
# global_delta_time_maximum_s = 1
# flux_height_minimum_m = 1
# lava_density = 1
# solidification_temperature = 900
# ambient_temperature = 295
# max_temperature = 1000
# c_v = 1
# emissivity = 1
# cooling_accelerator_factor = 1
# stefan_boltzmann_constant = 1
# update_heat_quantity_lava_height_minimum_m = 1
# ##################################################################

@ti.kernel
def computeFluxTransfers(
    lava_flux: ti.types.ndarray(dtype=ti.f32, ndim=3),
    dem_elev: ti.types.ndarray(dtype=ti.f32, ndim=2),
    solid_lava_thickness: ti.types.ndarray(dtype=ti.f32, ndim=2),
    lava_thickness: ti.types.ndarray(dtype=ti.f32, ndim=2),
    temperature: ti.types.ndarray(dtype=ti.f32, ndim=2),
    neighRows: ti.types.ndarray(dtype=ti.i32, ndim=1),
    neighCols: ti.types.ndarray(dtype=ti.i32, ndim=1),
    neighDistances: ti.types.ndarray(dtype=ti.f32, ndim=1),
    n_grid: ti.i32,
    scaled_grid_size_m: ti.f32,
    quality_tolerance: ti.f32,
    flux_height_minimum_m: ti.f32,
    delta_total_height_min: ti.f32,
    delta_total_height_max: ti.f32,
    lava_density: ti.f32,
    gravity: ti.f32,
    H2O: ti.f32
):
    # height, width, neighbors = lava_flux.shape[0], lava_flux.shape[1], lava_flux.shape[2]
    for i,k,n in lava_flux:
        cur_cell_total_height = dem_elev[i,k] + solid_lava_thickness[i,k]
        i_n,k_n = int(i+neighRows[n]), int(k+neighCols[n])
        if(i_n < 0 or k_n < 0 or i_n >= n_grid or k_n >= n_grid):
            lava_flux[i,k,n] = 0.0
        else:
            neigh_cell_total_height = dem_elev[i_n,k_n] + solid_lava_thickness[i_n,k_n]
            delta_z = neigh_cell_total_height - cur_cell_total_height
            delta_h = lava_thickness[i_n,k_n] - lava_thickness[i,k]
            h = lava_thickness[i,k]
            T = temperature[i,k]
            delta_x_sign = -1
            if((delta_z+delta_h) > 0):
                h = lava_thickness[i_n,k_n]
                T = temperature[i_n,k_n]
                delta_x_sign = 1
            
            delta_total_height_minimum_m = (quality_tolerance-1)/9.0*(delta_total_height_max-delta_total_height_min)+delta_total_height_min
            if (h<=flux_height_minimum_m or ti.abs(delta_z+delta_h) < delta_total_height_minimum_m):
                lava_flux[i,k,n] = 0.0
            else:
                rho = lava_density
                g = gravity
                delta_x = delta_x_sign * neighDistances[n] * scaled_grid_size_m
                S_y = 10.0**(13.00997 - 0.0089*T)
                eta = 10.0**(-4.643 + (5812.44 - 427.04*H2O)/(T - 499.31 + 28.74*ti.log(H2O)))
                h_cr = S_y * (ti.math.sqrt(delta_z**2 + delta_x**2)) / (rho*g*delta_x_sign*(delta_z+delta_h))
                a = h/h_cr
                if(h>h_cr):
                    q = (S_y * h_cr**2 * delta_x)/(3.0*eta) * (a**3 - 3.0/2.0*a**2 + 1.0/2.0)
                    lava_flux[i,k,n] = q
                else:
                    lava_flux[i,k,n] = 0.0

@ti.kernel
def computeTimeSteps(
    lava_flux: ti.types.ndarray(dtype=ti.f32, ndim=3),
    lava_thickness: ti.types.ndarray(dtype=ti.f32, ndim=2),
    delta_time: ti.types.ndarray(dtype=ti.f32, ndim=2),
    delta_time_c: ti.f32,
    cell_area_m: ti.f32,
    global_delta_time_maximum_s: ti.f32,
    flux_height_minimum_m: ti.f32
):
    for i,k in delta_time:
        c = delta_time_c
        h = lava_thickness[i,k]
        A = cell_area_m
        q_tot = 0.0
        for n in ti.static(range(8)):
            q_tot += lava_flux[i,k,n]
        if q_tot<0.0 and h>flux_height_minimum_m:
            delta_time[i,k] = c*h*A/ti.abs(q_tot)
        else:
            delta_time[i,k] = global_delta_time_maximum_s

@ti.kernel
def computeGlobalTimeStep(
    delta_time: ti.types.ndarray(dtype=ti.f32, ndim=2),
    global_delta_time_maximum_s: ti.f32
):
    global_delta_time = global_delta_time_maximum_s
    for i,k in delta_time:
        ti.atomic_min(global_delta_time, delta_time[i,k])
    delta_time[0,0] = global_delta_time

@ti.kernel
def computeNewLavaThickness(
    lava_flux: ti.types.ndarray(dtype=ti.f32, ndim=3),
    lava_thickness: ti.types.ndarray(dtype=ti.f32, ndim=2),
    global_delta_time: ti.f32,
    cell_area_m: ti.f32
):
    for i,k in lava_thickness:
        q_tot = 0.0
        for n in ti.static(range(8)):
            q_tot += lava_flux[i,k,n]
        delta_lava_thickness = q_tot*global_delta_time/cell_area_m
        lava_thickness[i,k] += delta_lava_thickness
        if(lava_thickness[i,k]<=0):
            lava_thickness[i,k] = 0.0

@ti.kernel
def computeHeatRadiationLoss(
    lava_flux: ti.types.ndarray(dtype=ti.f32, ndim=3),
    lava_thickness: ti.types.ndarray(dtype=ti.f32, ndim=2),
    temperature: ti.types.ndarray(dtype=ti.f32, ndim=2),
    new_temperature: ti.types.ndarray(dtype=ti.f32, ndim=2),
    heat_quantity: ti.types.ndarray(dtype=ti.f32, ndim=2),
    neighRows: ti.types.ndarray(dtype=ti.i32, ndim=1),
    neighCols: ti.types.ndarray(dtype=ti.i32, ndim=1),
    global_delta_time: ti.f32,
    lava_density: ti.f32,
    ambient_temperature: ti.f32,
    max_temperature: ti.f32,
    c_v: ti.f32,
    emissivity: ti.f32,
    cooling_accelerator_factor: ti.f32,
    cell_area_m: ti.f32,
    stefan_boltzmann_constant: ti.f32,
    update_heat_quantity_lava_height_minimum_m: ti.f32
):
    for i,k in lava_thickness:
        delta_Q_t_m = 0.0
        for n in ti.static(range(8)):
            i_n,k_n = int(i+neighRows[n]), int(k+neighCols[n])
            q_i = lava_flux[i,k,n]
            if(q_i>0):
                delta_Q_t_m += q_i*temperature[i_n,k_n]
            else:
                delta_Q_t_m += q_i*temperature[i,k]
        rho = lava_density
        delta_Q_t_m *= rho * c_v * global_delta_time
        
        epsilon = emissivity
        cooling_factor = cooling_accelerator_factor
        A = cell_area_m
        # Stefan–Boltzmann
        sigma = stefan_boltzmann_constant
        delta_Q_t_r = 0.0
        if lava_thickness[i,k] > update_heat_quantity_lava_height_minimum_m:
            delta_Q_t_r = epsilon * A * sigma * temperature[i,k]**4 * global_delta_time * 1.8**cooling_factor

        heat_quantity[i,k] += delta_Q_t_m - delta_Q_t_r
        
        if(heat_quantity[i,k]<0):
            heat_quantity[i,k] = 0.0
            new_temperature[i,k] = ambient_temperature
        else:
            h_t_dt = lava_thickness[i,k]
            curr_temperature = heat_quantity[i,k] / (rho * c_v * h_t_dt * A)
            new_temperature[i,k] = ti.max(ambient_temperature,ti.min(curr_temperature,max_temperature))

@ti.kernel
def updateTemperature(
    temperature: ti.types.ndarray(dtype=ti.f32, ndim=2),
    new_temperature: ti.types.ndarray(dtype=ti.f32, ndim=2)
):
    for i,k in temperature:
        temperature[i,k] = new_temperature[i,k]

@ti.kernel
def computeLavaSolidification(
    solid_lava_thickness: ti.types.ndarray(dtype=ti.f32, ndim=2),
    lava_thickness: ti.types.ndarray(dtype=ti.f32, ndim=2),
    temperature: ti.types.ndarray(dtype=ti.f32, ndim=2),
    heat_quantity: ti.types.ndarray(dtype=ti.f32, ndim=2),
    ambient_temperature: ti.f32,
    solidification_temperature: ti.f32,
    max_temperature: ti.f32,
    emissivity: ti.f32,
    stefan_boltzmann_constant: ti.f32,
    lava_density: ti.f32,
    c_v: ti.f32,
    cell_area_m: ti.f32,
    global_delta_time: ti.f32
):
    for i,k in temperature:
        if (temperature[i,k] < solidification_temperature):
            epsilon = emissivity
            sigma = stefan_boltzmann_constant
            rho = lava_density
            new_solid_lava = (epsilon * sigma * solidification_temperature**3 * global_delta_time) / (rho * c_v)
            if(lava_thickness[i,k] > new_solid_lava):
                solid_lava_thickness[i,k] += new_solid_lava
                lava_thickness[i,k] -= new_solid_lava
                h_t_dt = lava_thickness[i,k]
                A = cell_area_m
                curr_temperature = heat_quantity[i,k] / (rho * c_v * h_t_dt * A)
                temperature[i,k] = ti.max(ambient_temperature,ti.min(curr_temperature,max_temperature))
            else:
                solid_lava_thickness[i,k] += lava_thickness[i,k]
                lava_thickness[i,k] = 0.0
                temperature[i,k] = ambient_temperature

# computeFluxTransfers(
#     lava_flux,
#     dem_elev,
#     solid_lava_thickness,
#     lava_thickness,
#     temperature,
#     nRows,
#     nCols,
#     neighDistances,
#     n_grid,
#     scaled_grid_size_m,
#     quality_tolerance,
#     flux_height_minimum_m,
#     delta_total_height_min,
#     delta_total_height_max,
#     lava_density,
#     gravity,
#     H2O
# )
# computeTimeSteps(
#     lava_flux,
#     dem_elev,
#     lava_thickness,
#     delta_time,
#     delta_time_c,
#     cell_area_m,
#     global_delta_time_maximum_s,
#     flux_height_minimum_m
# )
# global_delta_time = computeGlobalTimeStep(
#     delta_time,
#     global_delta_time_maximum_s
# )
# computeNewLavaThickness(
#     lava_flux,
#     dem_elev,
#     lava_thickness,
#     global_delta_time,
#     cell_area_m
# )
# computeHeatRadiationLoss(
#     lava_flux,
#     dem_elev,
#     lava_thickness,
#     temperature,
#     new_temperature,
#     heat_quantity,
#     nRows,
#     nCols,
#     global_delta_time,
#     lava_density,
#     ambient_temperature,
#     max_temperature,
#     c_v,
#     emissivity,
#     cooling_accelerator_factor,
#     cell_area_m,
#     stefan_boltzmann_constant,
#     update_heat_quantity_lava_height_minimum_m
# )
# updateTemperature(
#     dem_elev,
#     temperature,
#     new_temperature
# )
# computeLavaSolidification(
#     solid_lava_thickness,
#     lava_thickness,
#     temperature,
#     heat_quantity,
#     ambient_temperature,
#     solidification_temperature,
#     max_temperature,
#     emissivity,
#     stefan_boltzmann_constant,
#     lava_density,
#     c_v,
#     cell_area_m,
#     global_delta_time
# )

mod = ti.aot.Module(ti.vulkan)
mod.add_kernel(computeFluxTransfers)
mod.add_kernel(computeTimeSteps)
mod.add_kernel(computeGlobalTimeStep)
mod.add_kernel(computeNewLavaThickness)
mod.add_kernel(computeHeatRadiationLoss)
mod.add_kernel(updateTemperature)
mod.add_kernel(computeLavaSolidification)
mod.archive("module.tcm")