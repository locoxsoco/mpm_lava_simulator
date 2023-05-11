import taichi as ti
import imageio.v3 as iio

@ti.data_oriented
class Heightmap:
    def __init__(self, heightmap_file_path, dim, hm_elev_min_m, hm_elev_max_m):
        self.heightmap_file = heightmap_file_path
        hm_im = iio.imread(self.heightmap_file)
        # Size of heightmap that will define size of cube
        self.hm_height_px, self.hm_width_px = hm_im.shape
        # Conversion heightmap pixel to km => 1 km == 10.8 px
        # self.px_to_km = 1.0/10.8
        self.px_to_km = 1.0/200.0
        self.hm_max_value = 65535.0
        self.hm_elev_min_km = hm_elev_min_m/1000.0
        self.hm_elev_max_km = hm_elev_max_m/1000.0
        self.hm_elev_range_km = self.hm_elev_max_km-self.hm_elev_min_km
        self.heightmap_image = ti.field(dtype=ti.f32, shape=(self.hm_height_px, self.hm_width_px))
        self.heightmap_image.from_numpy(hm_im)

        self.heightmap_positions = ti.Vector.field(dim, ti.f32, self.hm_height_px*self.hm_width_px)
        self.heightmap_normals = ti.Vector.field(dim, ti.f32, self.hm_height_px*self.hm_width_px)
        self.heightmap_distances = ti.field(ti.f32, self.hm_height_px*self.hm_width_px)
        self.heightmap_colors = ti.Vector.field(3, ti.f32, self.hm_height_px*self.hm_width_px)
        self.heightmap_indices = ti.field(ti.i32, shape=((self.hm_height_px-1)*(self.hm_width_px-1)*2)*3)

        self.verts = ti.Vector.field(3, dtype=ti.f32, shape = 2*self.hm_width_px*self.hm_height_px)
        self.fill_heightmap()
        self.fill_normalmap()
        self.init_lines()

    @ti.kernel
    def fill_heightmap(self):
        # Fill the image
        for row,column in self.heightmap_image:
            self.heightmap_positions[row*self.hm_width_px+column] = ti.Vector([
                row*self.px_to_km,
                self.heightmap_image[self.hm_width_px-1-row,column]/self.hm_max_value*self.hm_elev_range_km,
                column*self.px_to_km
            ])
            self.heightmap_colors[row*self.hm_width_px+column] = ti.Vector([
                self.heightmap_image[self.hm_width_px-1-row,column]/self.hm_max_value,
                self.heightmap_image[self.hm_width_px-1-row,column]/self.hm_max_value,
                self.heightmap_image[self.hm_width_px-1-row,column]/self.hm_max_value
            ])
            if(row<(self.hm_width_px-1) and column<(self.hm_height_px-1)):
                self.heightmap_indices[(row*(self.hm_width_px-1)+column)*6+0] = row*self.hm_width_px+column
                self.heightmap_indices[(row*(self.hm_width_px-1)+column)*6+1] = row*self.hm_width_px+column+1
                self.heightmap_indices[(row*(self.hm_width_px-1)+column)*6+2] = (row+1)*self.hm_width_px+column
                self.heightmap_indices[(row*(self.hm_width_px-1)+column)*6+3] = row*self.hm_width_px+column+1
                self.heightmap_indices[(row*(self.hm_width_px-1)+column)*6+4] = (row+1)*self.hm_width_px+column+1
                self.heightmap_indices[(row*(self.hm_width_px-1)+column)*6+5] = (row+1)*self.hm_width_px+column
    
    @ti.kernel
    def fill_normalmap(self):
        # Fill the image
        for i,j in self.heightmap_image:
            if(i==0):
                if(j==0):
                    self.heightmap_normals[i*self.hm_height_px+j] = ti.Vector([
                        0.0,
                        1.0,
                        0.0
                    ]).normalized()
                elif(j==self.hm_width_px-1):
                    self.heightmap_normals[i*self.hm_height_px+j] = ti.Vector([
                        0.0,
                        1.0,
                        0.0
                    ]).normalized()
                else:
                    self.heightmap_normals[i*self.hm_height_px+j] = ti.Vector([
                        0.0,
                        1.0,
                        (self.heightmap_positions[i*self.hm_height_px+j-1][1]-self.heightmap_positions[i*self.hm_height_px+j+1][1])/(2.0*self.px_to_km)
                    ]).normalized()
            elif(i==self.hm_height_px-1):
                if(j==0):
                    self.heightmap_normals[i*self.hm_height_px+j] = ti.Vector([
                        0.0,
                        1.0,
                        0.0
                    ]).normalized()
                elif(j==self.hm_width_px-1):
                    self.heightmap_normals[i*self.hm_height_px+j] = ti.Vector([
                        0.0,
                        1.0,
                        0.0
                    ]).normalized()
                else:
                    self.heightmap_normals[i*self.hm_height_px+j] = ti.Vector([
                        0.0,
                        1.0,
                        (self.heightmap_positions[i*self.hm_height_px+j-1][1]-self.heightmap_positions[i*self.hm_height_px+j+1][1])/(2.0*self.px_to_km)
                    ]).normalized()
            else:
                if(j==0):
                    self.heightmap_normals[i*self.hm_height_px+j] = ti.Vector([
                        (self.heightmap_positions[(i-1)*self.hm_height_px+j][1]-self.heightmap_positions[(i+1)*self.hm_height_px+j][1])/(2.0*self.px_to_km),
                        1.0,
                        0.0
                    ]).normalized()
                elif(j==self.hm_width_px-1):
                    self.heightmap_normals[i*self.hm_height_px+j] = ti.Vector([
                        (self.heightmap_positions[(i-1)*self.hm_height_px+j][1]-self.heightmap_positions[(i+1)*self.hm_height_px+j][1])/(2.0*self.px_to_km),
                        1.0,
                        0.0
                    ]).normalized()
                else:
                    self.heightmap_normals[i*self.hm_height_px+j] = ti.Vector([
                        (self.heightmap_positions[(i-1)*self.hm_height_px+j][1]-self.heightmap_positions[(i+1)*self.hm_height_px+j][1])/(2.0*self.px_to_km),
                        1.0,
                        (self.heightmap_positions[i*self.hm_height_px+j-1][1]-self.heightmap_positions[i*self.hm_height_px+j+1][1])/(2.0*self.px_to_km)
                    ]).normalized()                
            self.heightmap_distances[i*self.hm_height_px+j] = -ti.math.dot(self.heightmap_normals[i*self.hm_height_px+j],self.heightmap_positions[i*self.hm_height_px+j])

    @ti.kernel
    def init_lines(self):
        for i in range(self.hm_width_px*self.hm_height_px):
            self.verts[2*i] = self.heightmap_positions[i]
            self.verts[2*i+1] = self.heightmap_positions[i] + self.heightmap_normals[i]*0.2