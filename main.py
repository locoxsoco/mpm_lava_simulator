import numpy as np
import taichi as ti
import imageio

arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
# ti.init(debug=True)
ti.init(arch=arch)

########################### Setup variables ###########################
# dim, n_grid, steps, dt = 3, 32, 25, 4e-4
dim, n_grid, steps, dt = 3, 64, 25, 2e-4
#dim, n_grid, steps, dt = 3, 256, 5, 1e-4
#dim, n_grid, steps, dt = 3, 128, 25, 1e-4
dx = 1 / n_grid
n_particles = n_grid**dim // 2**(dim - 1) # // 128
# n_particles = 4194304
# n_particles = 100
print(f'Number of particles: {n_particles}')
#######################################################################

######################### Heightmap variables #########################
#heightmap_file = './data/heightmaps/fuji.png'
heightmap_file = './data/heightmaps/fuji_scoped.png'
hm_im = imageio.imread(heightmap_file)
# Size of heightmap that will define size of cube
hm_height, hm_width = hm_im.shape
# Conversion heightmap pixel to km => 1 km == 10.8 px
px_to_km = 1.0/10.8
hm_max_value = 65535.0
hm_elev_min_km = -7.8/1000.0
hm_elev_max_km = 3776.0/1000.0
y_scale = ((hm_elev_max_km-hm_elev_min_km)/(hm_width*px_to_km))
heightmap_image = ti.field(dtype=ti.f32, shape=(hm_height, hm_width))
heightmap_image.from_numpy(hm_im)

heightmap_positions = ti.Vector.field(dim, ti.f32, hm_height*hm_width)
heightmap_normals = ti.Vector.field(dim, ti.f32, hm_height*hm_width)
heightmap_distances = ti.field(ti.f32, hm_height*hm_width)
heightmap_colors = ti.Vector.field(3, ti.f32, hm_height*hm_width)
heightmap_indices = ti.field(ti.i32, shape=((hm_height-1)*(hm_width-1)*2)*3)

verts = ti.Vector.field(3, dtype=ti.f32, shape = 2*hm_width*hm_height)
#######################################################################

######################### Particle Parameters #########################
# From Setup
p_vol = (dx * 0.5)**2
neighbour = (3, ) * dim
# Customizable
p_rho = 1 # kg/m^3
bound = 3
E = 1000  # Young's modulus
nu = 0.2  #  Poisson's ratio
GRAVITY = [0, -9.8, 0]
k = 1 # Heat Conductivity
T_0 = 37.5 # Temperature at initial state
p_c = 1 # Heat Capacity per unit mass
# From Customizable
p_mass = p_vol * p_rho
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / (
    (1 + nu) * (1 - 2 * nu))  # Lame parameters for stress-strain coeff
#######################################################################

######################### Debug Parameters #########################
normal_line_column = 0
#######################################################################

############################# MPM Fields ##############################
F_x = ti.Vector.field(dim, float, n_particles) # Position
F_v = ti.Vector.field(dim, float, n_particles) # Velocity
F_C = ti.Matrix.field(dim, dim, float, n_particles) # Angular Momentum (APIC)
F_dg = ti.Matrix.field(3, 3, dtype=float, shape=n_particles) # Deformation Gradient
F_Jp = ti.field(float, n_particles) # Plastic Deformation Gradient Determinant (Plasticity)

F_colors = ti.Vector.field(4, float, n_particles)
F_colors_random = ti.Vector.field(4, float, n_particles)
F_materials = ti.field(int, n_particles)
F_grid_v = ti.Vector.field(dim, float, (n_grid, ) * dim)
F_grid_m = ti.field(float, (n_grid, ) * dim)
F_grid_level = ti.field(int, (n_grid, ) * dim)
F_used = ti.field(int, n_particles)
#######################################################################

############################# Materials ###############################
WATER = 0
JELLY = 1
SNOW = 2
LAVA = 3
TERRAIN = 4
#######################################################################

############################## Cubemap ################################
cube_verts_list = np.array([
    [-1,-1,-1],
    [-1,-1, 1],
    [-1, 1,-1],
    [-1, 1, 1],
    [1,-1,-1],
    [1,-1, 1],
    [1, 1,-1],
    [1, 1, 1]
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
    [-1, 0, 0],
    [1, 0, 0],
    [0,-1, 0],
    [0, 1, 0],
    [0, 0,-1],
    [0, 0, 1]
], dtype=np.float32)
cube_colors_list_lvl0 = np.array([
    [0, 0.5, 0, 0.1],
    [0, 0.5, 0, 0.1],
    [0, 0.5, 0, 0.1],
    [0, 0.5, 0, 0.1],
    [0, 0.5, 0, 0.1],
    [0, 0.5, 0, 0.1],
    [0, 0.5, 0, 0.1],
    [0, 0.5, 0, 0.1]
], dtype=np.float32)
cube_colors_list_lvl1 = np.array([
    [0.5, 0, 0, 0.1],
    [0.5, 0, 0, 0.1],
    [0.5, 0, 0, 0.1],
    [0.5, 0, 0, 0.1],
    [0.5, 0, 0, 0.1],
    [0.5, 0, 0, 0.1],
    [0.5, 0, 0, 0.1],
    [0.5, 0, 0, 0.1]
], dtype=np.float32)

cube_positions = ti.Vector.field(dim, ti.f32, 8)
cube_positions.from_numpy(cube_verts_list)

cube_indices = ti.field(ti.i32, shape=len(cube_faces_list))
cube_indices.from_numpy(cube_faces_list)

cube_normals = ti.Vector.field(dim, ti.f32, 8)
cube_normals.from_numpy(cube_face_normals_list)

cube_colors_lvl0 = ti.Vector.field(4, ti.f32, 8)
cube_colors_lvl0.from_numpy(cube_colors_list_lvl0)
cube_colors_lvl1 = ti.Vector.field(4, ti.f32, 8)
cube_colors_lvl1.from_numpy(cube_colors_list_lvl1)

curr_cube_positions = ti.Vector.field(dim, ti.f32, 8)
#m_transforms = ti.Matrix.field(4,4,dtype=ti.f32,shape=(n_grid, ) * dim)
m_transforms_lvl0 = ti.Matrix.field(4,4,dtype=ti.f32,shape=n_grid*n_grid*n_grid)
m_transforms_lvl1 = ti.Matrix.field(4,4,dtype=ti.f32,shape=n_grid*n_grid*n_grid)
#######################################################################

class CubeVolume:
    def __init__(self, minimum, size, material):
        self.minimum = minimum
        self.size = size
        self.volume = self.size.x * self.size.y * self.size.z
        self.material = material

class Heightmap:
    def __init__(self, position, material):
        self.position = position
        self.material = materials


# @ti.kernel
# def calculate_m_transforms_lvl0():
#     for i, j, k in m_transforms:
#         m_transforms[i,j,k] = ti.Matrix.identity(float,4)
#         m_transforms[i,j,k] /= 2*n_grid
#         m_transforms[i,j,k][0,3] = i*dx + dx/2
#         m_transforms[i,j,k][1,3] = j*dx + dx/2
#         m_transforms[i,j,k][2,3] = k*dx + dx/2
#         m_transforms[i,j,k][3,3] = 1
@ti.kernel
def calculate_m_transforms_lvl0():
    for idx in m_transforms_lvl0:
        i = idx//(n_grid*n_grid)
        j = (idx-n_grid*n_grid*i)//n_grid
        k = idx%n_grid
        if (F_grid_level[i,j,k]==0):
            m_transforms_lvl0[idx] = ti.Matrix.identity(float,4)
            m_transforms_lvl0[idx] /= 2*n_grid
            m_transforms_lvl0[idx][0,3] = i*dx + dx/2
            m_transforms_lvl0[idx][1,3] = j*dx + dx/2
            m_transforms_lvl0[idx][2,3] = k*dx + dx/2
            m_transforms_lvl0[idx][3,3] = 1
        else:
            m_transforms_lvl0[idx] = ti.Matrix.identity(float,4)
            m_transforms_lvl0[idx] /= 2*n_grid
            m_transforms_lvl0[idx][0,3] = 324534654
            m_transforms_lvl0[idx][1,3] = 324534654
            m_transforms_lvl0[idx][2,3] = 324534654
            m_transforms_lvl0[idx][3,3] = 1

@ti.kernel
def calculate_m_transforms_lvl1():
    for idx in m_transforms_lvl1:
        i = idx//(n_grid*n_grid)
        j = (idx-n_grid*n_grid*i)//n_grid
        k = idx%n_grid
        if (F_grid_level[i,j,k]==1):
            m_transforms_lvl1[idx] = ti.Matrix.identity(float,4)
            m_transforms_lvl1[idx] /= 2*n_grid
            m_transforms_lvl1[idx][0,3] = i*dx + dx/2
            m_transforms_lvl1[idx][1,3] = j*dx + dx/2
            m_transforms_lvl1[idx][2,3] = k*dx + dx/2
            m_transforms_lvl1[idx][3,3] = 1
        else:
            m_transforms_lvl1[idx] = ti.Matrix.identity(float,4)
            m_transforms_lvl1[idx] /= 2*n_grid
            m_transforms_lvl1[idx][0,3] = 324534654
            m_transforms_lvl1[idx][1,3] = 324534654
            m_transforms_lvl1[idx][2,3] = 324534654
            m_transforms_lvl1[idx][3,3] = 1

@ti.kernel
def substep(p_mass: float, mu_0: float, lambda_0: float, g_x: float, g_y: float, g_z: float):
    # 0. Reset Grid values
    for I in ti.grouped(F_grid_m):
        F_grid_v[I] = ti.zero(F_grid_v[I])
        F_grid_m[I] = 0
    # 1. Particle to grid transfer (P2G)
    ti.loop_config(block_dim=n_grid)
    for p in F_x:
        # Only the particles in the experiment
        if F_used[p] == 0:
            continue
        # Get grid cell and weight of corresponding particle
        Xp = F_x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        # Update Deformation Gradient
        F_dg[p] = (ti.Matrix.identity(float, 3) + dt * F_C[p]) @ F_dg[p]
        # Hardening coefficient h: snow gets harder when compressed
        h = 0.0
        if F_materials[p] == WATER or F_materials[p] == SNOW or F_materials[p] == LAVA:
            h = ti.exp(10 * (1.0 - F_Jp[p]))
        elif F_materials[p] == JELLY:  # jelly, make it softer
            h = 0.3
        # Lame Parameters
        mu, lambd = mu_0 * h, lambda_0 * h
        if F_materials[p] == WATER or F_materials[p] == LAVA:  # liquid
            mu = 0.0

        U, sig, V = ti.svd(F_dg[p])
        J = 1.0 # Deformation Gradient Determinant
        # Calculate J
        for d in ti.static(range(3)):
            new_sig = sig[d, d]
            if F_materials[p] == SNOW:
                new_sig = ti.min(ti.max(sig[d, d], 1 - 2.5e-2),
                              1 + 4.5e-3)  # Plasticity allows to break
            F_Jp[p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig
        if F_materials[p] == WATER or F_materials[p] == LAVA:
            # Reset deformation gradient to avoid numerical instability
            new_F = ti.Matrix.identity(float, 3)
            new_F[0, 0] = J
            F_dg[p] = new_F
        elif F_materials[p] == SNOW:
            # Reconstruct elastic deformation gradient after plasticity
            F_dg[p] = U @ sig @ V.transpose()
        # Stress Tensor = 2 * mu * strain_tensor + I * lambda * trace function of strain_tensor
        stress = 2 * mu * (F_dg[p] - U @ V.transpose()) @ F_dg[p].transpose(
        ) + ti.Matrix.identity(float, 3) * lambd * J * (J - 1)
        stress = (-dt * p_vol * 4) * stress / dx**2
        affine = stress + p_mass * F_C[p]

        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            F_grid_v[base +
                     offset] += weight * (p_mass * F_v[p] + affine @ dpos)
            F_grid_m[base + offset] += weight * p_mass
    # Resolve velocity on grid
    for i, j, k in F_grid_m: #for I in ti.grouped(F_grid_m):
        # 2. Compute grid velocities
        if F_grid_m[i, j, k] > 0:
            F_grid_v[i, j, k] /= F_grid_m[i, j, k]
        F_grid_v[i, j, k] += dt * ti.Vector([g_x, g_y, g_z])
        # Heightmap collision
        # if(F_grid_level[i-1, j, k] == 0 and F_grid_v[i, j, k][0] < 0):
        #     F_grid_v[i, j, k][0] = 0
        # elif(F_grid_level[i+1, j, k] == 0 and F_grid_v[i, j, k][0] > 0):
        #     F_grid_v[i, j, k][0] = 0
        # if(F_grid_level[i, j-1, k] == 0 and F_grid_v[i, j, k][1] < 0):
        #     F_grid_v[i, j, k][1] = 0
        # elif(F_grid_level[i, j+1, k] == 0 and F_grid_v[i, j, k][1] > 0):
        #     F_grid_v[i, j, k][1] = 0
        # if(F_grid_level[i, j, k-1] == 0 and F_grid_v[i, j, k][2] < 0):
        #     F_grid_v[i, j, k][2] = 0
        # elif(F_grid_level[i, j, k+1] == 0 and F_grid_v[i, j, k][2] > 0):
        #     F_grid_v[i, j, k][2] = 0
        # Boundary Collision
        if i < bound and F_grid_v[i, j, k][0] < 0:
            F_grid_v[i, j, k][0] = 0  # Boundary conditions
        if i > n_grid - bound and F_grid_v[i, j, k][0] > 0:
            F_grid_v[i, j, k][0] = 0
        if j < bound and F_grid_v[i, j, k][1] < 0:
            F_grid_v[i, j, k][1] = 0
        if j > n_grid - bound and F_grid_v[i, j, k][1] > 0:
            F_grid_v[i, j, k][1] = 0
        if k < bound and F_grid_v[i, j, k][2] < 0:
            F_grid_v[i, j, k][2] = 0
        if k > n_grid - bound and F_grid_v[i, j, k][2] > 0:
            F_grid_v[i, j, k][2] = 0
    # Transfer velocity back to particles
    ti.loop_config(block_dim=n_grid)
    # 7. Grid to particle transfer
    for p in F_x:
        if F_used[p] == 0:
            continue
        Xp = F_x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.zero(F_v[p])
        new_C = ti.zero(F_C[p])
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            g_v = F_grid_v[base + offset]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) / dx**2
        F_v[p] = new_v
        F_C[p] = new_C
        # 8. Particle advention
        # F_x[p] += dt * F_v[p]
        # Heightmap collision
        expected_F_x = F_x[p] + dt * F_v[p]
        hm_x = int(expected_F_x[0]*hm_height)
        hm_z = int(expected_F_x[2]*hm_width)
        planeN = heightmap_normals[hm_z*hm_width + hm_x]
        planeD = heightmap_distances[hm_z*hm_width + hm_x]
        if((ti.math.dot(planeN,expected_F_x)+planeD)*(ti.math.dot(planeN,F_x[p])+planeD)<=0):
            expected_F_x = expected_F_x - (1.0+1.0)*(planeN.dot(expected_F_x)+planeD)*planeN
            velElastic = - (1.0+1.0)*ti.math.dot(planeN,F_v[p])*planeN
            F_v[p] = F_v[p] + velElastic
            velT = F_v[p] - ti.math.dot(planeN,F_v[p])*planeN
            F_v[p] = F_v[p] - (0.01)*velT
            # F_v[p] *= 0.0
            F_x[p] = expected_F_x
        else:
            F_x[p] = expected_F_x

@ti.kernel
def init_cube_vol(first_par: int, last_par: int, x_begin: float,
                  y_begin: float, z_begin: float, x_size: float, y_size: float,
                  z_size: float, material: int):
    for i in range(first_par, last_par):
        F_x[i] = ti.Vector([ti.random() for i in range(dim)]) * ti.Vector(
            [x_size, y_size, z_size]) + ti.Vector([x_begin, y_begin, z_begin])
        F_Jp[i] = 1
        F_dg[i] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        F_v[i] = ti.Vector([(ti.random()-0.5)*2, ti.random(), (ti.random()-0.5)*2])
        F_materials[i] = material
        F_colors_random[i] = ti.Vector(
            [ti.random(), ti.random(),
             ti.random(), ti.random()])
        F_used[i] = 1

@ti.kernel
def set_all_unused():
    for p in F_used:
        F_used[p] = 0
        # basically throw them away so they aren't rendered
        F_x[p] = ti.Vector([533799.0, 533799.0, 533799.0])
        F_Jp[p] = 1
        F_dg[p] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        F_C[p] = ti.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        F_v[p] = ti.Vector([0.0, 0.0, 0.0])

@ti.kernel
def fill_heightmap():
    # Fill the image
    for row,column in heightmap_image:
        # heightmap_positions[(hm_height-1-row)*hm_width+(hm_width-1-column)] = ti.Vector([
        #     (hm_height-1-row)/hm_height,
        #     heightmap_image[hm_width-1-row,column]/hm_max_value*y_scale+dx*(bound+1),
        #     (hm_width-1-column)/hm_width
        # ])
        heightmap_positions[row*hm_width+column] = ti.Vector([
            row/hm_height,
            heightmap_image[hm_width-1-row,column]/hm_max_value*y_scale+dx*(bound+4)*0,
            column/hm_width
        ])
        heightmap_colors[row*hm_width+column] = ti.Vector([
            heightmap_image[hm_width-1-row,column]/hm_max_value,
            heightmap_image[hm_width-1-row,column]/hm_max_value,
            heightmap_image[hm_width-1-row,column]/hm_max_value
        ])
        if(row<(hm_width-1) and column<(hm_height-1)):
            heightmap_indices[(row*(hm_width-1)+column)*6+0] = row*hm_width+column
            heightmap_indices[(row*(hm_width-1)+column)*6+1] = row*hm_width+column+1
            heightmap_indices[(row*(hm_width-1)+column)*6+2] = (row+1)*hm_width+column
            heightmap_indices[(row*(hm_width-1)+column)*6+3] = row*hm_width+column+1
            heightmap_indices[(row*(hm_width-1)+column)*6+4] = (row+1)*hm_width+column+1
            heightmap_indices[(row*(hm_width-1)+column)*6+5] = (row+1)*hm_width+column

@ti.kernel
def set_levels():
    # Fill the image
    for i,j,k in F_grid_level:
        # if grid height is inside terrain
        if(j/n_grid <= heightmap_positions[int((i/n_grid+dx/2.0)*hm_height)*hm_width+int((k/n_grid+dx/2.0)*hm_width)][1]):
            F_grid_level[i,j,k] = 0
        else:
            F_grid_level[i,j,k] = 1

@ti.kernel
def fill_normalmap():
    # Fill the image
    for i,j in heightmap_image:
        if(i==0):
            if(j==0):
                heightmap_normals[i*hm_height+j] = ti.Vector([
                    0.0,
                    1.0*dx,
                    0.0
                ]).normalized()
            elif(j==hm_width-1):
                heightmap_normals[i*hm_height+j] = ti.Vector([
                    0.0,
                    1.0*dx,
                    0.0
                ]).normalized()
            else:
                heightmap_normals[i*hm_height+j] = ti.Vector([
                    0.0,
                    1.0*dx,
                    (heightmap_positions[i*hm_height+j-1][1]-heightmap_positions[i*hm_height+j+1][1])/(2.0*dx)
                ]).normalized()
        elif(i==hm_height-1):
            if(j==0):
                heightmap_normals[i*hm_height+j] = ti.Vector([
                    0.0,
                    1.0*dx,
                    0.0
                ]).normalized()
            elif(j==hm_width-1):
                heightmap_normals[i*hm_height+j] = ti.Vector([
                    0.0,
                    1.0*dx,
                    0.0
                ]).normalized()
            else:
                heightmap_normals[i*hm_height+j] = ti.Vector([
                    0.0,
                    1.0*dx,
                    (heightmap_positions[i*hm_height+j-1][1]-heightmap_positions[i*hm_height+j+1][1])/(2.0*dx)
                ]).normalized()
        else:
            if(j==0):
                heightmap_normals[i*hm_height+j] = ti.Vector([
                    (heightmap_positions[(i-1)*hm_height+j][1]-heightmap_positions[(i+1)*hm_height+j][1])/(2.0*dx),
                    1.0*dx,
                    0.0
                ]).normalized()
            elif(j==hm_width-1):
                heightmap_normals[i*hm_height+j] = ti.Vector([
                    (heightmap_positions[(i-1)*hm_height+j][1]-heightmap_positions[(i+1)*hm_height+j][1])/(2.0*dx),
                    1.0*dx,
                    0.0
                ]).normalized()
            else:
                heightmap_normals[i*hm_height+j] = ti.Vector([
                    (heightmap_positions[(i-1)*hm_height+j][1]-heightmap_positions[(i+1)*hm_height+j][1])/(2.0*dx),
                    1.0*y_scale,
                    (heightmap_positions[i*hm_height+j-1][1]-heightmap_positions[i*hm_height+j+1][1])/(2.0*dx)
                ]).normalized()                
        heightmap_distances[i*hm_height+j] = -ti.math.dot(heightmap_normals[i*hm_height+j],heightmap_positions[i*hm_height+j])
        if (i==200 and j==200):
            print(f'i-1: {heightmap_positions[(i-1)*hm_height+j][1]}')
            print(f'i+1: {heightmap_positions[(i+1)*hm_height+j][1]}')
            print(f'j-1: {heightmap_positions[i*hm_height+j-1][1]}')
            print(f'j+1: {heightmap_positions[i*hm_height+j+1][1]}')
            print('N', heightmap_normals[i*hm_height+j])
            print('P', heightmap_positions[i*hm_height+j])
            print('D', heightmap_distances[i*hm_height+j])
            print('dx', dx)
            print('y_scale', y_scale)
        # if(i*hm_height+j < 150):
        #     print(f'i: {i} j: {j} heightmap_distances: {heightmap_distances[i*hm_height+j]} heightmap_positions: {heightmap_positions[i*hm_height+j]} heightmap_normals: {heightmap_normals[i*hm_height+j]}')

@ti.kernel
def init_lines():
    for i in range(hm_width*hm_height):
        verts[2*i] = heightmap_positions[i]
        verts[2*i+1] = heightmap_positions[i] + heightmap_normals[i]*0.01

@ti.kernel
def set_color_by_material(mat_color: ti.types.ndarray()):
    for i in range(n_particles):
        mat = F_materials[i]
        F_colors[i] = ti.Vector(
            [mat_color[mat, 0], mat_color[mat, 1], mat_color[mat, 2], 1.0])


def init_vols(vols):
    set_all_unused()
    total_vol = 0
    for v in vols:
        total_vol += v.volume

    next_p = 0
    for i, v in enumerate(vols):
        v = vols[i]
        if isinstance(v, CubeVolume):
            par_count = int(v.volume / total_vol * n_particles)
            if i == len(
                    vols
            ) - 1:  # this is the last volume, so use all remaining particles
                par_count = n_particles - next_p
            init_cube_vol(next_p, next_p + par_count, *v.minimum, *v.size,
                          v.material)
            next_p += par_count
        else:
            raise Exception("???")

    if heightmap_file:
        fill_heightmap()
        set_levels()
        calculate_m_transforms_lvl0()
        calculate_m_transforms_lvl1()
        fill_normalmap()
        init_lines()

def init():
    global paused
    global p_mass
    global mu_0, lambda_0
    global p_rho
    global bound
    global E
    global nu
    p_mass = p_vol * p_rho
    mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / (
        (1 + nu) * (1 - 2 * nu))  # Lame parameters
    init_vols(presets[curr_preset_id])

def show_options():
    global use_random_colors
    global paused
    global particles_radius
    global curr_preset_id
    global p_rho
    global bound
    global E
    global nu
    global normal_line_column

    with gui.sub_window("Params", 0.05, 0.1, 0.2, 0.15) as w:
        p_rho = w.slider_float("Density", p_rho, 0, 1000)
        bound = w.slider_float("Bound", bound, 0, 10)
        E = w.slider_float("Young's modulus", E, 0, 1000)
        nu = w.slider_float("Poisson's ratio", nu, 0, 1)

    with gui.sub_window("Presets", 0.05, 0.1, 0.2, 0.15) as w:
        old_preset = curr_preset_id
        for i in range(len(presets)):
            if w.checkbox(preset_names[i], curr_preset_id == i):
                curr_preset_id = i
        if curr_preset_id != old_preset:
            init()
            paused = True

    with gui.sub_window("Gravity", 0.05, 0.3, 0.2, 0.1) as w:
        GRAVITY[0] = w.slider_float("x", GRAVITY[0], -10, 10)
        GRAVITY[1] = w.slider_float("y", GRAVITY[1], -10, 10)
        GRAVITY[2] = w.slider_float("z", GRAVITY[2], -10, 10)

    with gui.sub_window("Options", 0.05, 0.45, 0.2, 0.4) as w:
        use_random_colors = w.checkbox("use_random_colors", use_random_colors)
        if not use_random_colors:
            material_colors[WATER] = w.color_edit_3("water color",
                                                    material_colors[WATER])
            material_colors[SNOW] = w.color_edit_3("snow color",
                                                   material_colors[SNOW])
            material_colors[JELLY] = w.color_edit_3("jelly color",
                                                    material_colors[JELLY])
            set_color_by_material(np.array(material_colors, dtype=np.float32))
        particles_radius = w.slider_float("particles radius m",
                                          particles_radius*(hm_height*px_to_km*1000), 0.0, 100.0)/(hm_height*px_to_km*1000)
        if w.button("restart"):
            init()
        if paused:
            if w.button("Continue"):
                paused = False
        else:
            if w.button("Pause"):
                paused = True

    with gui.sub_window("Debug normal lines", 0.05, 0.1, 0.2, 0.15) as w:
        normal_line_column = w.slider_int("Column", normal_line_column, 0, 400)

def render():
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)

    scene.ambient_light((0, 0, 0))

    colors_used = F_colors_random if use_random_colors else F_colors
    scene.particles(F_x, per_vertex_color=colors_used, radius=particles_radius)
    scene.mesh(vertices=heightmap_positions, indices=heightmap_indices,per_vertex_color=heightmap_colors,normals=heightmap_normals)

    for i in range(150):
        scene.lines(verts, color = (0.28, 0.68, 0.99), width = 0.5, vertex_count = 2, vertex_offset = 4*(normal_line_column+hm_width*i))

    scene.mesh_instance(vertices=cube_positions, indices=cube_indices,per_vertex_color=cube_colors_lvl0, transforms=m_transforms_lvl0)
    #scene.mesh_instance(vertices=cube_positions, indices=cube_indices,per_vertex_color=cube_colors_lvl1, transforms=m_transforms_lvl1)

    scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.5, 0.5, 0.5))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.5, 0.5, 0.5))
    scene.point_light(pos=(0.0, 0.0, 0.0), color=(1.0, 1.0, 1.0))
    # scene.point_light(pos=(1.1, 0.15, 0.0), color=(1.0, 1.0, 1.0))

    canvas.scene(scene)

def main():
    while window.running:
        if not paused:
            for _ in range(steps):
                substep(p_mass,mu_0,lambda_0,*GRAVITY)
        render()
        show_options()
        window.show()


print("Loading presets...this might take a minute")

presets = [
    [
        # CubeVolume(ti.Vector([0.475, 0.2, 0.475]), ti.Vector([0.005, 0.05, 0.005]),
        CubeVolume(ti.Vector([0.4975, 0.25, 0.4975]), ti.Vector([0.05, 0.1, 0.05]),
                LAVA),
    ],
    [
        CubeVolume(ti.Vector([0.55, 0.05, 0.55]), ti.Vector([0.4, 0.4, 0.4]),
                WATER),
    ],
    [
        CubeVolume(ti.Vector([0.05, 0.05, 0.05]),
                    ti.Vector([0.3, 0.4, 0.3]), WATER),
        CubeVolume(ti.Vector([0.65, 0.05, 0.65]),
                    ti.Vector([0.3, 0.4, 0.3]), WATER),
    ],
    [
        CubeVolume(ti.Vector([0.6, 0.05, 0.6]),
                    ti.Vector([0.25, 0.25, 0.25]), WATER),
        CubeVolume(ti.Vector([0.35, 0.35, 0.35]),
                    ti.Vector([0.25, 0.25, 0.25]), SNOW),
        CubeVolume(ti.Vector([0.05, 0.6, 0.05]),
                    ti.Vector([0.25, 0.25, 0.25]), JELLY),
    ]
]
preset_names = [
    "Lava Dam Break",
    "Single Dam Break",
    "Double Dam Break",
    "Water Snow Jelly"
]

curr_preset_id = 0

paused = False

use_random_colors = False
particles_radius = 0.001

material_colors = [
    (0.1, 0.6, 0.9),
    (0.93, 0.33, 0.23),
    (1.0, 1.0, 1.0),
    (0.84765625, 0.015625, 0.16015625)
]

init()

res = (1080, 720)
window = ti.ui.Window("Real MPM 3D", res, vsync=True)

canvas = window.get_canvas()
canvas.set_background_color((0.16796875,0.17578125,0.2578125))
gui = window.GUI
scene = ti.ui.Scene()
camera = ti.ui.make_camera()
camera.position(0.0, 0.3, 1.0)
camera.lookat(0.5, 0.0, 0.5)
camera.fov(55)

if __name__ == '__main__':
    main()