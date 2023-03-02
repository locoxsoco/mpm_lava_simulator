import numpy as np
import taichi as ti
import imageio

arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
ti.init(debug=True)

heightmap_file = './data/heightmaps/fuji_scoped.png'
hm_im = imageio.imread(heightmap_file)
# Size of heightmap that will define size of cube
x_map_length, z_map_length = hm_im.shape

print(f'x_map_length: {x_map_length} z_map_length: {z_map_length}')

heightmap_image = ti.field(dtype=ti.f32, shape=(x_map_length, z_map_length))
heightmap_image.from_numpy(hm_im)

@ti.kernel
def fill_normalmap():
    # Fill the image
    for i,j in heightmap_image:
        print(f'i: {i} j: {j}')

fill_normalmap()
print(f'Done')